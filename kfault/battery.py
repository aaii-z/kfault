"""Small fixed stress battery for kernels that pass the standard check."""


class _ReferenceProbeError(Exception):
    """The reference rejected a generated stress input; the probe is invalid."""


class _KernelProbeError(Exception):
    """The generated model crashed on an input accepted by the reference."""


def silent_battery(
    ref_model,
    kernel_model,
    fresh_inputs,
    init_inputs,
    Model,
    seed,
    *,
    custom_stress_inputs=None,
):
    """Return ``(core_failures, robustness_failures, diagnostics)``."""
    import torch

    from kfault.comparison import clone_tree, compare_outputs, map_tensors

    core = []
    robust = []
    diagnostics = {}

    def error_detail(error):
        return {
            "error_type": type(error).__name__,
            "message": str(error)[:1000],
        }

    def agree(inputs, ref_m, ker_m, atol=1e-3, rtol=1e-3):
        with torch.no_grad():
            try:
                reference = ref_m(*clone_tree(inputs))
            except Exception as error:
                raise _ReferenceProbeError(str(error)) from error
            try:
                candidate = ker_m(*clone_tree(inputs))
            except Exception as error:
                raise _KernelProbeError(str(error)) from error
        return compare_outputs(reference, candidate, atol=atol, rtol=rtol)

    def check(name, inputs, *, ref_m=ref_model, ker_m=kernel_model, target=core):
        try:
            matches, detail = agree(inputs, ref_m, ker_m)
        except _ReferenceProbeError as error:
            diagnostics[name] = {
                "status": "skipped_reference_error",
                **error_detail(error),
            }
            return None
        except _KernelProbeError as error:
            target.append(f"{name}_crash")
            diagnostics[name] = {"status": "kernel_error", **error_detail(error)}
            return False
        except Exception as error:
            target.append(f"{name}_harness_error")
            diagnostics[name] = {"status": "harness_error", **error_detail(error)}
            return False
        if not matches:
            target.append(name)
            diagnostics[name] = {"status": "mismatch", "detail": detail}
            return False
        diagnostics[name] = {"status": "pass"}
        return True

    # Three same-shape random draws catch hardcoded or lucky outputs.
    for index in range(3):
        outcome = check(f"fresh_seed_{index + 1}", fresh_inputs(), target=[])
        if outcome is False:
            core.append("fresh_seed")
            break

    # Preserve values while presenting non-standard dense strides.
    def noncontiguous(tensor):
        return torch.stack([tensor, tensor], dim=-1)[..., 0]

    check("non_contig", map_tensors(fresh_inputs(), noncontiguous))

    # Re-run from identical private inputs and protect the first output from a
    # model that reuses an internal output buffer.
    try:
        deterministic_inputs = fresh_inputs()
        with torch.no_grad():
            first = clone_tree(kernel_model(*clone_tree(deterministic_inputs)))
            second = kernel_model(*clone_tree(deterministic_inputs))
        deterministic, detail = compare_outputs(first, second)
        if deterministic:
            diagnostics["nondeterministic"] = {"status": "pass"}
        else:
            core.append("nondeterministic")
            diagnostics["nondeterministic"] = {
                "status": "mismatch",
                "detail": detail,
            }
    except Exception as error:
        core.append("nondeterministic_crash")
        diagnostics["nondeterministic"] = {
            "status": "kernel_error",
            **error_detail(error),
        }

    def floating_only(tensor):
        return tensor.is_floating_point()

    base = fresh_inputs()
    check(
        "zeros",
        map_tensors(
            base,
            lambda tensor: torch.zeros_like(tensor)
            if floating_only(tensor)
            else tensor,
        ),
    )
    check(
        "small_magnitude",
        map_tensors(
            fresh_inputs(),
            lambda tensor: tensor * 1e-4 if floating_only(tensor) else tensor,
        ),
    )
    check(
        "extreme",
        map_tensors(
            fresh_inputs(),
            lambda tensor: tensor * 100.0 if floating_only(tensor) else tensor,
        ),
    )

    # Optional problem-defined variants cover shapes that cannot safely be
    # inferred generically (empty, odd, boundary, broadcasting, and so on).
    if custom_stress_inputs is not None:
        try:
            variants = custom_stress_inputs()
            for variant_name, inputs in variants.items():
                check(f"custom_{variant_name}", inputs)
        except Exception as error:
            diagnostics["custom_inputs"] = {
                "status": "provider_error",
                **error_detail(error),
            }

    # FP16 is a robustness signal, not part of the FP32 silent-failure label.
    saved_state = {
        name: value.detach().clone()
        for name, value in kernel_model.state_dict().items()
    }
    ref_half = None
    try:
        torch.manual_seed(seed)
        ref_half = Model(*clone_tree(init_inputs)).cuda().half().eval()
    except Exception as error:
        diagnostics["fp16"] = {
            "status": "skipped_reference_error",
            **error_detail(error),
        }

    if ref_half is not None:
        try:
            kernel_half = kernel_model.half()
            half_inputs = map_tensors(
                fresh_inputs(),
                lambda tensor: tensor.half() if floating_only(tensor) else tensor,
            )
            check(
                "fp16",
                half_inputs,
                ref_m=ref_half,
                ker_m=kernel_half,
                target=robust,
            )
        except Exception as error:
            robust.append("fp16_crash")
            diagnostics["fp16"] = {
                "status": "kernel_error",
                **error_detail(error),
            }
        finally:
            try:
                kernel_model.float()
                kernel_model.load_state_dict(saved_state)
            except Exception as error:
                robust.append("fp16_restore_error")
                diagnostics["fp16_restore"] = {
                    "status": "restore_error",
                    **error_detail(error),
                }

    return core, robust, diagnostics
