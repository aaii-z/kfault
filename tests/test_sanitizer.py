import unittest

from kfault.sanitizer import parse_sanitizer_output


class SanitizerParsingTests(unittest.TestCase):
    def test_clean(self):
        result = parse_sanitizer_output("ERROR SUMMARY: 0 errors", 0)
        self.assertEqual(result["status"], "clean")
        self.assertEqual(result["errors"], 0)

    def test_memory_violation(self):
        output = "Invalid __global__ read\nERROR SUMMARY: 2 errors"
        result = parse_sanitizer_output(output, 86)
        self.assertEqual(result["status"], "violation")
        self.assertEqual(result["errors"], 2)

    def test_kernel_error(self):
        result = parse_sanitizer_output("RuntimeError: CUDA error", 1)
        self.assertEqual(result["status"], "kernel_error")

    def test_unavailable(self):
        result = parse_sanitizer_output("Device not supported", 1)
        self.assertEqual(result["status"], "unavailable")

    def test_unknown_nonzero_is_tool_error(self):
        result = parse_sanitizer_output("unknown failure", 2)
        self.assertEqual(result["status"], "tool_error")


if __name__ == "__main__":
    unittest.main()
