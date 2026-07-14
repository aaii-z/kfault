"""Helpers for isolating model inputs and comparing structured outputs."""


def clone_tree(value):
    """Clone every tensor in a nested Python container.

    Model inputs and outputs can contain tuples, lists, or dictionaries. Keeping
    the surrounding structure while cloning tensors prevents one model from
    mutating data that is later observed by the other model.
    """
    import torch

    if isinstance(value, torch.Tensor):
        return value.clone(memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return type(value)((key, clone_tree(item)) for key, item in value.items())
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        items = [clone_tree(item) for item in value]
        if hasattr(value, "_fields"):  # namedtuple
            return type(value)(*items)
        return tuple(items)
    return value


def map_tensors(value, transform):
    """Apply ``transform`` to every tensor in a nested container."""
    import torch

    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return type(value)(
            (key, map_tensors(item, transform)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [map_tensors(item, transform) for item in value]
    if isinstance(value, tuple):
        items = [map_tensors(item, transform) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items)
    return value


def compare_outputs(reference, candidate, *, atol=1e-3, rtol=1e-3, path="output"):
    """Recursively compare tensor/scalar outputs and return ``(ok, detail)``."""
    import cmath
    import math

    import torch

    if isinstance(reference, torch.Tensor) or isinstance(candidate, torch.Tensor):
        if not isinstance(reference, torch.Tensor) or not isinstance(
            candidate, torch.Tensor
        ):
            return False, f"{path}: tensor/non-tensor type mismatch"
        if reference.shape != candidate.shape:
            return (
                False,
                f"{path}: shape mismatch {tuple(candidate.shape)} vs "
                f"{tuple(reference.shape)}",
            )
        if reference.is_complex() or candidate.is_complex():
            if not reference.is_complex() or not candidate.is_complex():
                return False, f"{path}: real/complex dtype mismatch"
            values_match = torch.allclose(
                reference.to(torch.complex128),
                candidate.to(torch.complex128),
                atol=atol,
                rtol=rtol,
            )
        elif reference.is_floating_point() or candidate.is_floating_point():
            if not reference.is_floating_point() or not candidate.is_floating_point():
                return False, f"{path}: integer/floating dtype mismatch"
            values_match = torch.allclose(
                reference.to(torch.float64),
                candidate.to(torch.float64),
                atol=atol,
                rtol=rtol,
            )
        else:
            values_match = torch.equal(reference, candidate)
        if not values_match:
            return False, f"{path}: value mismatch"
        return True, ""

    if isinstance(reference, dict) or isinstance(candidate, dict):
        if not isinstance(reference, dict) or not isinstance(candidate, dict):
            return False, f"{path}: mapping type mismatch"
        if reference.keys() != candidate.keys():
            return False, f"{path}: mapping keys mismatch"
        for key in reference:
            ok, detail = compare_outputs(
                reference[key],
                candidate[key],
                atol=atol,
                rtol=rtol,
                path=f"{path}[{key!r}]",
            )
            if not ok:
                return ok, detail
        return True, ""

    sequence_types = (list, tuple)
    if isinstance(reference, sequence_types) or isinstance(candidate, sequence_types):
        if type(reference) is not type(candidate):
            return False, f"{path}: sequence type mismatch"
        if len(reference) != len(candidate):
            return False, f"{path}: sequence length mismatch"
        for index, (ref_item, candidate_item) in enumerate(
            zip(reference, candidate, strict=True)
        ):
            ok, detail = compare_outputs(
                ref_item,
                candidate_item,
                atol=atol,
                rtol=rtol,
                path=f"{path}[{index}]",
            )
            if not ok:
                return ok, detail
        return True, ""

    if isinstance(reference, (int, float, complex, bool)) and isinstance(
        candidate, (int, float, complex, bool)
    ):
        has_complex = isinstance(reference, complex) or isinstance(candidate, complex)
        isclose = cmath.isclose if has_complex else math.isclose
        if isclose(reference, candidate, abs_tol=atol, rel_tol=rtol):
            return True, ""
        return False, f"{path}: scalar value mismatch"

    try:
        equal = reference == candidate
    except Exception:
        equal = False
    if isinstance(equal, bool) and equal:
        return True, ""
    return False, f"{path}: value mismatch"
