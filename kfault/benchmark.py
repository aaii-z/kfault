"""Interleaved CUDA-event timing for the reference and generated model."""


def benchmark(ref_model, kernel_model, inputs, warmup: int = 5, iters: int = 30):
    """Return median, spread, and speedup from interleaved measurements."""
    import random
    import statistics

    import torch

    from kfault.comparison import clone_tree

    if warmup < 0 or iters < 2:
        raise ValueError("warmup must be non-negative and iters must be at least 2")

    models = {"ref": ref_model, "kernel": kernel_model}
    samples = {"ref": [], "kernel": []}
    rng = random.Random(0)

    def invoke(model):
        model(*clone_tree(inputs))

    def measure(model):
        call_inputs = clone_tree(inputs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(*call_inputs)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    with torch.no_grad():
        for _ in range(warmup):
            order = list(models)
            rng.shuffle(order)
            for name in order:
                invoke(models[name])
        torch.cuda.synchronize()

        for _ in range(iters):
            order = list(models)
            rng.shuffle(order)
            for name in order:
                samples[name].append(measure(models[name]))

    def percentile(values, fraction):
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def summary(values):
        return {
            "median_ms": round(statistics.median(values), 4),
            "p10_ms": round(percentile(values, 0.10), 4),
            "p90_ms": round(percentile(values, 0.90), 4),
            "stdev_ms": round(statistics.pstdev(values), 4),
        }

    ref_summary = summary(samples["ref"])
    kernel_summary = summary(samples["kernel"])
    ref_ms = ref_summary["median_ms"]
    kernel_ms = kernel_summary["median_ms"]
    return {
        # Retain old names for result-analysis compatibility.
        "ref_ms": ref_ms,
        "kernel_ms": kernel_ms,
        "speedup": round(ref_ms / kernel_ms, 3) if kernel_ms else None,
        "ref": ref_summary,
        "kernel": kernel_summary,
        "warmup": warmup,
        "iterations": iters,
        "order_seed": 0,
    }
