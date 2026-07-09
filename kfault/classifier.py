"""The 4-stage classifier: kernel -> failure category.

Categories (first matching stage wins):
  syntactic  - code fails to build
  runtime    - crashes or trips compute-sanitizer on standard input
  functional - runs, but wrong output on the standard input
  silent     - passes the standard check, wrong under the stress battery
  pass       - correct on standard input and survives the battery

Pure classification: no API calls, no persistence. Weight sync mirrors
KernelBench: torch.manual_seed(seed) before get_init_inputs(), before
constructing each model, and before every input draw. Without this,
problems with learnable params give the two models different weights.
"""

from kfault.battery import silent_battery
from kfault.benchmark import benchmark
from kfault.sanitizer import run_sanitizer


def classify(
    generated_code: str,
    problem_code: str,
    *,
    use_sanitizer: bool = True,
    seed: int = 42,
):
    """Run Stages 1-4 and return (category, detail, extra)."""
    import torch

    extra: dict = {}

    # Stage 1: SYNTACTIC. Does the generated code build?
    kernel_ns: dict = {}
    problem_ns: dict = {}
    try:
        exec(problem_code, problem_ns)
        Model = problem_ns["Model"]
        get_init_inputs = problem_ns.get("get_init_inputs", lambda: [])
        get_inputs = problem_ns.get("get_inputs", lambda: [])
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
        return [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()
        ]

    torch.manual_seed(seed)
    ref_model = Model(*init_inputs).cuda().eval()

    try:
        exec(generated_code, kernel_ns)  # load_inline compiles here
        ModelNew = kernel_ns["ModelNew"]
        torch.manual_seed(seed)  # same weights as ref_model
        kernel_model = ModelNew(*init_inputs).cuda().eval()
    except Exception as e:
        return "syntactic", str(e)[:500], extra

    # Stage 2a: RUNTIME. Does it crash on standard input?
    std_inputs = fresh_inputs()
    try:
        with torch.no_grad():
            kernel_out = kernel_model(*std_inputs)
            ref_out = ref_model(*std_inputs)
        torch.cuda.synchronize()  # surface async kernel faults here
    except Exception as e:
        return "runtime", str(e)[:500], extra

    # kernel executes: time it against PyTorch on the standard input
    try:
        extra["timing"] = benchmark(ref_model, kernel_model, std_inputs)
    except Exception:
        extra["timing"] = None

    # Stage 2b: RUNTIME. Memory errors under compute-sanitizer?
    if use_sanitizer:
        errors, detail = run_sanitizer(problem_code, generated_code, seed)
        extra["sanitizer_errors"] = errors
        if errors:
            return "runtime", f"compute-sanitizer: {errors} error(s) {detail}", extra

    # Stage 3: FUNCTIONAL. Correct on the standard input?
    if ref_out.shape != kernel_out.shape:
        return (
            "functional",
            f"shape mismatch {tuple(kernel_out.shape)} vs {tuple(ref_out.shape)}",
            extra,
        )
    if not torch.allclose(ref_out.float(), kernel_out.float(), atol=1e-3, rtol=1e-3):
        return "functional", "mismatch on standard input", extra

    # Stage 4: SILENT. Wrong under the stress battery?
    core, robust = silent_battery(
        ref_model, kernel_model, fresh_inputs, init_inputs, Model, seed
    )
    extra["robustness_flags"] = robust
    if core:
        return "silent", ", ".join(core), extra

    detail = "survived core battery" + (
        f" (robustness flags: {', '.join(robust)})" if robust else ""
    )
    return "pass", detail, extra
