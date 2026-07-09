"""Kernel vs PyTorch timing on the standard input, using CUDA events."""


def benchmark(ref_model, kernel_model, inputs, warmup: int = 3, iters: int = 10):
    """Return {ref_ms, kernel_ms, speedup} with median times over `iters`."""
    import torch

    def median_ms(model):
        with torch.no_grad():
            for _ in range(warmup):
                model(*inputs)
            torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(*inputs)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        times.sort()
        return times[len(times) // 2]

    ref_ms = median_ms(ref_model)
    kernel_ms = median_ms(kernel_model)
    return {
        "ref_ms": round(ref_ms, 4),
        "kernel_ms": round(kernel_ms, 4),
        "speedup": round(ref_ms / kernel_ms, 3) if kernel_ms else None,
    }
