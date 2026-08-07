import unittest

from kfault.battery import _is_dtype_unsupported, _is_out_of_memory


class DtypeUnsupportedTests(unittest.TestCase):
    """A kernel refusing half input is not a robustness failure.

    Batches 01/02 flagged fp16_crash on 100% of passing kernels because every
    refusal counted as a fault. These cases pin the difference.
    """

    def assert_unsupported(self, message):
        self.assertTrue(_is_dtype_unsupported(Exception(message)), message)

    def assert_real_failure(self, message):
        self.assertFalse(_is_dtype_unsupported(Exception(message)), message)

    def test_aten_scalar_type_refusal(self):
        self.assert_unsupported("expected scalar type Float but found Half")

    def test_not_implemented_for_half(self):
        self.assert_unsupported("\"addmm_impl\" not implemented for 'Half'")

    def test_input_and_bias_type_mismatch(self):
        self.assert_unsupported(
            "Input type (c10::Half) and bias type (float) should be the same"
        )

    def test_pybind_signature_refusal(self):
        # the load_inline binding declared float*, so half never type-checks
        self.assert_unsupported(
            "fused_gemm_cuda(): incompatible function arguments; got Half tensor"
        )

    def test_data_ptr_mention(self):
        self.assert_unsupported("TypeError in data_ptr<float>")

    def test_dtype_word_alone_is_not_enough(self):
        # no dtype named: could be any pair of mismatched tensors, not a refusal
        self.assert_real_failure("expected m1 and m2 to have the same dtype")

    def test_illegal_memory_access_is_a_real_failure(self):
        self.assert_real_failure("CUDA error: an illegal memory access")

    def test_wrong_numbers_is_a_real_failure(self):
        self.assert_real_failure("output mismatch: max abs diff 0.5 in Half output")

    def test_out_of_memory_is_not_a_dtype_refusal(self):
        self.assert_real_failure("CUDA out of memory")
        self.assertTrue(_is_out_of_memory(Exception("CUDA out of memory")))


class CustomDtypeAssertionTests(unittest.TestCase):
    """Kernels that enforce fp32 with their own TORCH_CHECK.

    These messages are written by the model, so they never mention "half".
    DeepSeek's L2 p81 kernel said only "Input must be float32" and was wrongly
    counted as a robustness failure.
    """

    def assert_unsupported(self, message):
        self.assertTrue(_is_dtype_unsupported(Exception(message)), message)

    def assert_real_failure(self, message):
        self.assertFalse(_is_dtype_unsupported(Exception(message)), message)

    def test_the_observed_deepseek_message(self):
        self.assert_unsupported("Input must be float32")

    def test_common_phrasings(self):
        for message in (
            "x must be float32",
            "only float32 is supported",
            "expected float32 tensor",
            "kernel requires fp32 inputs",
            "dtype must be torch.float32",
            "invalid dtype, float32 only",
        ):
            self.assert_unsupported(message)

    def test_a_crash_mentioning_float32_is_not_a_refusal(self):
        # no assertion word: this is a computation failing, not a contract
        self.assert_real_failure("illegal memory access while writing float32 output")

    def test_out_of_memory_on_a_float32_buffer_is_not_a_refusal(self):
        self.assert_real_failure("CUDA out of memory allocating float32 buffer")


if __name__ == "__main__":
    unittest.main()
