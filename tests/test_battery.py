import unittest

from kfault.battery import _is_dtype_unsupported, _is_out_of_memory

# A kernel refusing half input is enforcing its declared contract, not failing.
# Batches 01/02 flagged fp16_crash on 100% of passing kernels because every
# refusal counted as a fault. Both framings appear: messages raised by aten,
# and messages the model wrote itself in a TORCH_CHECK, which never say "half".
REFUSALS = (
    "expected scalar type Float but found Half",
    "\"addmm_impl\" not implemented for 'Half'",
    "Input type (c10::Half) and bias type (float) should be the same",
    "fused_gemm_cuda(): incompatible function arguments; got Half tensor",
    "TypeError in data_ptr<float>",
    "Input must be float32",  # the observed DeepSeek L2 p81 message
    "x must be float32",
    "only float32 is supported",
    "expected float32 tensor",
    "kernel requires fp32 inputs",
    "dtype must be torch.float32",
    "invalid dtype, float32 only",
)

# Real faults that merely mention a dtype, and must not be excused as refusals.
REAL_FAILURES = (
    "expected m1 and m2 to have the same dtype",  # names no dtype
    "CUDA error: an illegal memory access",
    "output mismatch: max abs diff 0.5 in Half output",
    "CUDA out of memory",
    "illegal memory access while writing float32 output",
    "CUDA out of memory allocating float32 buffer",
)


class BatteryTests(unittest.TestCase):
    def test_dtype_refusal_is_distinguished_from_a_real_fault(self):
        for message in REFUSALS:
            with self.subTest(message=message):
                self.assertTrue(_is_dtype_unsupported(Exception(message)))
        for message in REAL_FAILURES:
            with self.subTest(message=message):
                self.assertFalse(_is_dtype_unsupported(Exception(message)))
        self.assertTrue(_is_out_of_memory(Exception("CUDA out of memory")))
        self.assertFalse(_is_out_of_memory(Exception("illegal memory access")))


if __name__ == "__main__":
    unittest.main()
