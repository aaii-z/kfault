"""Run one generated-kernel invocation under compute-sanitizer memcheck."""

DRIVER = r"""
import os
import torch

seed = int(os.environ.get("KFAULT_SEED", "42"))
problem = open(os.environ["KFAULT_PROBLEM"]).read()
kernel = open(os.environ["KFAULT_KERNEL"]).read()

pns = {}
exec(problem, pns)
get_init_inputs = pns.get("get_init_inputs", lambda: [])
get_inputs = pns.get("get_inputs", lambda: [])
torch.manual_seed(seed)
init_inputs = get_init_inputs()

kns = {}
exec(kernel, kns)
ModelNew = kns["ModelNew"]
torch.manual_seed(seed)
kernel_model = ModelNew(*init_inputs).cuda().eval()

torch.manual_seed(seed + 1)
def map_tensors(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: map_tensors(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [map_tensors(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(item, transform) for item in value)
    return value

inputs = map_tensors(get_inputs(), lambda tensor: tensor.cuda())
with torch.no_grad():
    kernel_model(*inputs)
torch.cuda.synchronize()
"""

_UNAVAILABLE_MARKERS = (
    "Device not supported",
    "not supported by",
    "Failed to initialize",
    "does not have permission",
    "No attachable process found",
)

_VIOLATION_MARKERS = (
    "Invalid ",
    "misaligned",
    "out-of-bounds",
    "Race reported",
    "Memory leak",
    "leaked",
)

_KERNEL_ERROR_MARKERS = (
    "CUDA error",
    "RuntimeError",
    "Traceback (most recent call last)",
    "an illegal memory access was encountered",
    "misaligned address",
    "launch failed",
)


def _result(status, detail, *, errors=None, returncode=None, log_excerpt=""):
    return {
        "status": status,
        "errors": errors,
        "detail": detail,
        "returncode": returncode,
        "log_excerpt": log_excerpt,
    }


def _excerpt(output: str, limit: int = 1200) -> str:
    """Return the most useful bounded section of sanitizer output."""
    starts = [
        output.find(marker)
        for marker in _VIOLATION_MARKERS + _KERNEL_ERROR_MARKERS
        if output.find(marker) >= 0
    ]
    start = min(starts) if starts else max(0, len(output) - limit)
    return output[start : start + limit].strip().replace("\n", " ")


def parse_sanitizer_output(output: str, returncode: int):
    """Classify completed sanitizer output separately from process launching."""
    import re

    excerpt = _excerpt(output)
    for marker in _UNAVAILABLE_MARKERS:
        if marker in output:
            return _result(
                "unavailable",
                f"compute-sanitizer unavailable: {marker}",
                returncode=returncode,
                log_excerpt=excerpt,
            )

    match = re.search(r"ERROR SUMMARY:\s+(\d+)\s+error", output)
    errors = int(match.group(1)) if match else None
    has_violation = any(marker in output for marker in _VIOLATION_MARKERS)

    if has_violation or (errors is not None and errors > 0):
        return _result(
            "violation",
            "compute-sanitizer detected a memory violation",
            errors=errors if errors is not None else 1,
            returncode=returncode,
            log_excerpt=excerpt,
        )

    if returncode == 0:
        return _result(
            "clean",
            "compute-sanitizer reported no memory errors",
            errors=errors if errors is not None else 0,
            returncode=0,
        )

    if any(marker in output for marker in _KERNEL_ERROR_MARKERS):
        return _result(
            "kernel_error",
            "kernel process failed under compute-sanitizer",
            errors=errors,
            returncode=returncode,
            log_excerpt=excerpt,
        )

    return _result(
        "tool_error",
        "compute-sanitizer exited without a recognizable diagnostic",
        errors=errors,
        returncode=returncode,
        log_excerpt=excerpt,
    )


def run_sanitizer(problem_code: str, kernel_code: str, seed: int = 42):
    """Return a structured sanitizer result.

    Status is one of ``clean``, ``violation``, ``kernel_error``, ``timeout``,
    ``unavailable``, or ``tool_error``. Only violations and kernel errors are
    attributed to the generated kernel by the classifier.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    sanitizer = shutil.which("compute-sanitizer") or (
        "/usr/local/cuda/bin/compute-sanitizer"
        if os.path.exists("/usr/local/cuda/bin/compute-sanitizer")
        else None
    )
    if sanitizer is None:
        return _result("unavailable", "compute-sanitizer not found")

    with tempfile.TemporaryDirectory() as tmp:
        prob_path = os.path.join(tmp, "problem.py")
        kern_path = os.path.join(tmp, "kernel.py")
        drv_path = os.path.join(tmp, "driver.py")
        for path, content in (
            (prob_path, problem_code),
            (kern_path, kernel_code),
            (drv_path, DRIVER),
        ):
            with open(path, "w") as file:
                file.write(content)

        env = dict(
            os.environ,
            KFAULT_PROBLEM=prob_path,
            KFAULT_KERNEL=kern_path,
            KFAULT_SEED=str(seed),
        )
        try:
            proc = subprocess.run(
                [
                    sanitizer,
                    "--tool",
                    "memcheck",
                    "--error-exitcode",
                    "86",
                    "--target-processes",
                    "all",
                    sys.executable,
                    drv_path,
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or "") + (error.stderr or "")
            return _result(
                "timeout",
                "compute-sanitizer exceeded 300 seconds",
                log_excerpt=_excerpt(output),
            )
        except Exception as error:
            return _result(
                "tool_error",
                f"sanitizer launch failed: {type(error).__name__}: {error}"[:500],
            )

        output = (proc.stdout or "") + (proc.stderr or "")
        return parse_sanitizer_output(output, proc.returncode)
