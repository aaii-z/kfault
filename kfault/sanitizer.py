"""Stage 2b: run a kernel once under compute-sanitizer memcheck.

The torch extension build is cached in TORCH_EXTENSIONS_DIR, so the kernel
compiled by Stage 1 is reused here. This is one cheap forward pass under
memory checking, in a subprocess.
"""

# Driver executed under compute-sanitizer. Seeded like the classifier.
DRIVER = r'''
import os
import torch

seed = int(os.environ.get("KFAULT_SEED", "42"))

problem = open(os.environ["KFAULT_PROBLEM"]).read()
kernel = open(os.environ["KFAULT_KERNEL"]).read()

pns = {}
exec(problem, pns)
Model = pns["Model"]
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
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]
with torch.no_grad():
    kernel_model(*inputs)
torch.cuda.synchronize()
'''

# Tool-level failure markers: sanitizer could not run at all (e.g. Modal T4
# rejects it with "Device not supported"). Not kernel errors.
_UNAVAILABLE_MARKERS = (
    "Device not supported",
    "not supported by",
    "Failed to initialize",
    "does not have permission",
    "Unsupported",
)


def run_sanitizer(problem_code: str, kernel_code: str, seed: int = 42):
    """Return (errors, detail). errors is None when the tool could not run;
    the caller then skips Stage 2b instead of mislabelling the kernel."""
    import os
    import re
    import shutil
    import subprocess
    import tempfile

    sanitizer = shutil.which("compute-sanitizer") or (
        "/usr/local/cuda/bin/compute-sanitizer"
        if os.path.exists("/usr/local/cuda/bin/compute-sanitizer")
        else None
    )
    if sanitizer is None:
        return None, "compute-sanitizer not found"

    with tempfile.TemporaryDirectory() as tmp:
        prob_path = os.path.join(tmp, "problem.py")
        kern_path = os.path.join(tmp, "kernel.py")
        drv_path = os.path.join(tmp, "driver.py")
        with open(prob_path, "w") as f:
            f.write(problem_code)
        with open(kern_path, "w") as f:
            f.write(kernel_code)
        with open(drv_path, "w") as f:
            f.write(DRIVER)

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
                    "--tool", "memcheck",
                    "--error-exitcode", "1",
                    "--target-processes", "all",
                    "python", drv_path,
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception as e:
            return None, f"sanitizer launch failed: {str(e)[:200]}"

        out = (proc.stdout or "") + (proc.stderr or "")

        for marker in _UNAVAILABLE_MARKERS:
            if marker in out:
                return None, f"sanitizer unavailable ({marker})"

        m = re.search(r"ERROR SUMMARY:\s+(\d+)\s+error", out)
        if m:
            errors = int(m.group(1))
        else:
            # no summary line: non-zero exit only when --error-exitcode fired
            errors = 1 if proc.returncode != 0 else 0

        # trust a positive count only with a real violation block in the log
        violation = any(
            k in out
            for k in ("Invalid ", "misaligned", "out-of-bounds", "Fatal", "leaked")
        )
        if errors and not violation:
            return None, "sanitizer reported non-memory error; ignoring"

        snippet = ""
        idx = out.find("Invalid ")
        if idx == -1:
            idx = out.find("========= ")
        if idx != -1:
            snippet = out[idx:idx + 800].replace("\n", " ")
        return errors, snippet
