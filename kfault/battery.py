"""Stage 4: fixed stress battery for kernels that pass the standard check.

Core checks (fp32, must pass, else the kernel is a silent failure):
  fresh_seed       - 3 new random input draws
  non_contig       - non-contiguous inputs
  nondeterministic - same input twice must give the same output
  extreme          - inputs scaled by 100

Robustness checks (reported separately, not counted as silent):
  fp16             - half precision

Keep this battery small and fixed.
"""


def silent_battery(ref_model, kernel_model, fresh_inputs, init_inputs, Model, seed):
    """Return (core_failures, robustness_failures) as lists of check names."""
    import torch

    core = []
    robust = []

    def agree(inputs, ref_m, ker_m, atol=1e-3, rtol=1e-3):
        with torch.no_grad():
            r = ref_m(*inputs)
            k = ker_m(*inputs)
        return torch.allclose(r.float(), k.float(), atol=atol, rtol=rtol)

    # fresh random draws
    try:
        for _ in range(3):
            if not agree(fresh_inputs(), ref_model, kernel_model):
                core.append("fresh_seed")
                break
    except Exception:
        core.append("fresh_seed_crash")

    # non-contiguous inputs: same values, broken strides
    def make_noncontig(inputs):
        out = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x = torch.stack([x, x], dim=-1)[..., 0]
            out.append(x)
        return out

    try:
        if not agree(make_noncontig(fresh_inputs()), ref_model, kernel_model):
            core.append("non_contig")
    except Exception:
        core.append("non_contig_crash")

    # same input twice must agree with itself (reference-free check for
    # uninitialised output, missing syncs, races)
    try:
        det_inputs = fresh_inputs()
        with torch.no_grad():
            k1 = kernel_model(*det_inputs)
            k2 = kernel_model(*det_inputs)
        if not torch.allclose(k1.float(), k2.float(), atol=1e-3, rtol=1e-3):
            core.append("nondeterministic")
    except Exception:
        core.append("nondeterministic_crash")

    # fp16: .half() mutates kernel_model and .float() does not restore the
    # exact fp32 weights, so snapshot and restore the state dict
    _saved = {k: v.detach().clone() for k, v in kernel_model.state_dict().items()}
    try:
        torch.manual_seed(seed)  # same weights as the fp32 reference
        ref_h = Model(*init_inputs).cuda().half().eval()
        ker_h = kernel_model.half()
        half_inputs = [
            x.half() if isinstance(x, torch.Tensor) else x for x in fresh_inputs()
        ]
        if not agree(half_inputs, ref_h, ker_h, atol=1e-2, rtol=1e-2):
            robust.append("fp16")
    except Exception:
        robust.append("fp16_crash")
    finally:
        try:
            kernel_model.float()
            kernel_model.load_state_dict(_saved)
        except Exception:
            pass

    # extreme magnitude inputs (still fp32)
    def scale(inputs, c=100.0):
        return [x * c if isinstance(x, torch.Tensor) else x for x in inputs]

    try:
        if not agree(scale(fresh_inputs()), ref_model, kernel_model):
            core.append("extreme")
    except Exception:
        core.append("extreme_crash")

    return core, robust
