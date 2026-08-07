"""Small fixed stress battery for kernels that pass the standard check."""


class _ReferenceProbeError(Exception):
    """The reference rejected a generated stress input; the probe is invalid."""


class _KernelProbeError(Exception):
    """The generated model crashed on an input accepted by the reference."""


def _is_out_of_memory(error) -> bool:
    """True when a probe died from GPU memory pressure rather than a real fault.

    Checks the message rather than the type: the probe wrappers re-raise as
    _ReferenceProbeError / _KernelProbeError, so the original OutOfMemoryError
    class is no longer available by the time this runs.
    """
    return "out of memory" in str(error).lower()


def _is_dtype_unsupported(error) -> bool:
    """True when the kernel refused a dtype rather than computing it wrongly.

    An extension written against ``data_ptr<float>()`` raises before a thread
    runs: that is the declared FP32 contract, not a fault. Counting it as one
    flagged every passing kernel, and a probe with no variance measures the
    harness. Only a kernel that runs in half and returns wrong numbers earns a
    flag.
    """
    message = str(error).lower()
    if "data_ptr<float>" in message or "data_ptr<c10::half>" in message:
        return True

    # A kernel may enforce its own contract with TORCH_CHECK rather than let
    # ATen raise; those messages never mention "half". Only the fp16 probe
    # reaches here, so a complaint about float32 is a refusal by definition.
    if "float32" in message or "fp32" in message or "torch.float" in message:
        if any(
            word in message
            for word in ("must", "only", "expect", "require", "support", "invalid")
        ):
            return True

    if "half" not in message and "c10::half" not in message:
        return False
    return any(
        marker in message
        for marker in (
            "expected scalar type",  # expected scalar type Float but found Half
            "not implemented for",  # "..." not implemented for 'Half'
            "should be the same",  # Input type and bias type should be the same
            "same dtype",  # expected m1 and m2 to have the same dtype
            "expected all tensors to be on the same",
            "invalid argument",
            "incompatible function arguments",
        )
    )


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

    def release():
        """Drop cached blocks between probes.

        Nothing frees a probe's inputs and activations until silent_battery
        returns, so on a large problem they accumulate until the GPU OOMs
        mid-battery. Releasing keeps peak memory at roughly one probe.
        """
        torch.cuda.empty_cache()

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

    def check(
        name,
        inputs,
        *,
        ref_m=ref_model,
        ker_m=kernel_model,
        target=core,
        skip_if=None,
        skip_status="skipped",
    ):
        try:
            return _check(name, inputs, ref_m, ker_m, target, skip_if, skip_status)
        finally:
            # drop this probe's inputs before reclaiming, or the cached blocks
            # stay pinned and the next probe starts from the same high-water mark
            del inputs
            release()

    def _check(name, inputs, ref_m, ker_m, target, skip_if=None, skip_status="skipped"):
        try:
            matches, detail = agree(inputs, ref_m, ker_m)
        except _ReferenceProbeError as error:
            diagnostics[name] = {
                "status": "skipped_reference_error",
                **error_detail(error),
            }
            return None
        except _KernelProbeError as error:
            if _is_out_of_memory(error):
                diagnostics[name] = {
                    "status": "skipped_out_of_memory",
                    **error_detail(error),
                }
                return None
            if skip_if is not None and skip_if(error):
                diagnostics[name] = {"status": skip_status, **error_detail(error)}
                return None
            target.append(f"{name}_crash")
            diagnostics[name] = {"status": "kernel_error", **error_detail(error)}
            return False
        except Exception as error:
            # A harness fault is not evidence about the kernel. Blaming it here
            # silently inflates the silent-failure count, which is the study's
            # headline number: on a 14.5 GiB T4 an OOM during the probes turned
            # a working kernel into "silent, non_contig_harness_error".
            status = (
                "skipped_out_of_memory"
                if _is_out_of_memory(error)
                else "skipped_harness_error"
            )
            diagnostics[name] = {"status": status, **error_detail(error)}
            return None
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
        if _is_out_of_memory(error):
            diagnostics["nondeterministic"] = {
                "status": "skipped_out_of_memory",
                **error_detail(error),
            }
        else:
            core.append("nondeterministic_crash")
            diagnostics["nondeterministic"] = {
                "status": "kernel_error",
                **error_detail(error),
            }
    finally:
        # this probe keeps two full outputs alive at once, so it is the single
        # largest resident block in the battery; release it before continuing
        first = second = deterministic_inputs = None
        release()

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
    # A kernel that *refuses* half input is recorded as unsupported, not failed;
    # see _is_dtype_unsupported.
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
                skip_if=_is_dtype_unsupported,
                skip_status="skipped_dtype_unsupported",
            )
        except Exception as error:
            if _is_dtype_unsupported(error):
                # .half() or the input cast itself refused; same non-fault case
                diagnostics["fp16"] = {
                    "status": "skipped_dtype_unsupported",
                    **error_detail(error),
                }
            else:
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
