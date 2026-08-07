"""The 4-stage classifier: kernel -> failure category.

Categories (first matching stage wins):
  syntactic  - code fails to build
  runtime    - crashes, trips compute-sanitizer, or is too slow to evaluate
               on standard input
  functional - runs, but wrong output on the standard input
  silent     - passes the standard check, wrong under the stress battery
  pass       - correct on standard input and survives the battery

Pure classification: no API calls, no persistence. Weight sync mirrors
KernelBench, seeding before get_init_inputs(), before constructing each model,
and before every input draw. Without it, problems with learnable parameters
give the two models different weights.
"""

from kfault.battery import silent_battery
from kfault.benchmark import benchmark
from kfault.comparison import clone_tree, compare_outputs, map_tensors
from kfault.config import FORWARD_TIME_LIMIT_S
from kfault.sanitizer import run_sanitizer


def classify(
    generated_code: str,
    problem_code: str,
    *,
    use_sanitizer: bool = True,
    seed: int = 42,
    forward_time_limit: float | None = FORWARD_TIME_LIMIT_S,
):
    """Run Stages 1-4 and return (category, detail, extra).

    ``forward_time_limit`` is the per-forward wall-clock budget in seconds;
    pass ``None`` to disable the check.
    """
    import torch

    extra: dict = {}

    def failure(category, stage, error):
        detail = str(error)[:1000]
        error_type = type(error).__name__ if isinstance(error, BaseException) else None
        extra["failure"] = {
            "stage": stage,
            "error_type": error_type,
            "message": detail,
        }
        return category, detail, extra

    # Stage 1: SYNTACTIC. Does the generated code build?
    kernel_ns: dict = {}
    problem_ns: dict = {}
    try:
        exec(problem_code, problem_ns)
        Model = problem_ns["Model"]
        get_init_inputs = problem_ns.get("get_init_inputs", lambda: [])
        get_inputs = problem_ns.get("get_inputs", lambda: [])
        get_stress_inputs = problem_ns.get("get_stress_inputs")
        torch.manual_seed(seed)
        init_inputs = get_init_inputs()
    except Exception as e:
        # a broken problem is our bug, not the model's
        raise RuntimeError(f"reference problem failed to load: {e}") from e

    # distinct but reproducible draws: incrementing seed offset per call
    _draw = [0]

    def fresh_inputs():
        _draw[0] += 1
        torch.manual_seed(seed + _draw[0])
        return map_tensors(get_inputs(), lambda tensor: tensor.cuda())

    torch.manual_seed(seed)
    ref_model = Model(*clone_tree(init_inputs)).cuda().eval()

    try:
        compiled_code = compile(generated_code, "<generated_kernel>", "exec")
    except Exception as error:
        return failure("syntactic", "python_parse", error)

    try:
        exec(compiled_code, kernel_ns)  # load_inline compiles here
    except Exception as error:
        return failure("syntactic", "generated_code_execution_or_build", error)

    try:
        ModelNew = kernel_ns["ModelNew"]
    except Exception as error:
        return failure("syntactic", "model_lookup", error)

    try:
        torch.manual_seed(seed)  # same weights as ref_model
        kernel_model = ModelNew(*clone_tree(init_inputs)).cuda().eval()
    except Exception as error:
        return failure("runtime", "model_initialization", error)

    # Stage 2a: RUNTIME. Does it crash on standard input?
    std_inputs = fresh_inputs()
    try:
        with torch.no_grad():
            ref_out = ref_model(*clone_tree(std_inputs))
            kernel_out = kernel_model(*clone_tree(std_inputs))
        torch.cuda.synchronize()  # surface async kernel faults here
    except Exception as error:
        return failure("runtime", "standard_forward", error)

    # Stage 2a': RUNTIME. Is a single forward call affordable?
    # Everything after this calls the kernel ~90 more times, so a kernel needing
    # tens of seconds per call cannot finish and produces no row at all. The
    # first call pays lazy module load and CUDA setup, so time a warm one.
    if forward_time_limit is not None:
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                kernel_model(*clone_tree(std_inputs))
                end.record()
            end.synchronize()
            forward_seconds = start.elapsed_time(end) / 1000.0
        except Exception as error:
            return failure("runtime", "standard_forward", error)
        extra["forward_seconds"] = round(forward_seconds, 4)
        if forward_seconds > forward_time_limit:
            return failure(
                "runtime",
                "forward_time_limit",
                f"one warm forward took {forward_seconds:.1f}s, over the "
                f"{forward_time_limit:.1f}s budget; the battery and benchmark "
                f"would need ~90 such calls",
            )

    # Stage 2b: RUNTIME. Memory errors under compute-sanitizer?
    if use_sanitizer:
        sanitizer = run_sanitizer(problem_code, generated_code, seed)
        extra["sanitizer"] = sanitizer
        # Compatibility field retained for existing result analysis scripts.
        extra["sanitizer_errors"] = sanitizer["errors"]
        if sanitizer["status"] in {"violation", "kernel_error"}:
            detail = sanitizer["detail"]
            if sanitizer["log_excerpt"]:
                detail += f": {sanitizer['log_excerpt']}"
            return failure("runtime", f"sanitizer_{sanitizer['status']}", detail)
        if sanitizer["status"] != "clean":
            extra.setdefault("validation_warnings", []).append(
                f"sanitizer_{sanitizer['status']}"
            )

    # Stage 3: FUNCTIONAL. Correct on the standard input?
    outputs_agree, mismatch = compare_outputs(ref_out, kernel_out)
    if not outputs_agree:
        return failure("functional", "standard_output_comparison", mismatch)

    custom_stress_inputs = None
    if get_stress_inputs is not None:

        def custom_stress_inputs():
            torch.manual_seed(seed + 10_000)
            variants = get_stress_inputs()
            if isinstance(variants, dict):
                items = variants.items()
            else:
                items = enumerate(variants)
            return {
                str(name): map_tensors(inputs, lambda tensor: tensor.cuda())
                for name, inputs in items
            }

    # Stage 4: SILENT. Wrong under the stress battery?
    # A battery crash is a harness failure, not evidence about the kernel, so it
    # must not be labelled silent. It also must not escape: a torch exception
    # cannot be deserialized by a caller without torch, hiding the real cause.
    try:
        core, robust, battery_diagnostics = silent_battery(
            ref_model,
            kernel_model,
            fresh_inputs,
            init_inputs,
            Model,
            seed,
            custom_stress_inputs=custom_stress_inputs,
        )
    except Exception as error:
        return failure("error", "stress_battery_harness", error)
    extra["robustness_flags"] = robust
    extra["battery_diagnostics"] = battery_diagnostics

    # Only correct kernels are meaningful to benchmark. Timing after all probes
    # also prevents repeated calls from changing state observed by validation.
    if core:
        extra["timing"] = None
        extra["timing_skipped"] = "kernel failed the stress battery"
    else:
        try:
            extra["timing"] = benchmark(ref_model, kernel_model, std_inputs)
        except Exception as error:
            extra["timing"] = None
            extra["timing_error"] = {
                "error_type": type(error).__name__,
                "message": str(error)[:1000],
            }

    if core:
        return failure("silent", "stress_battery", ", ".join(core))

    detail = "survived core battery" + (
        f" (robustness flags: {', '.join(robust)})" if robust else ""
    )
    warnings = extra.get("validation_warnings", [])
    if warnings:
        detail += f" (warnings: {', '.join(warnings)})"
    return "pass", detail, extra
