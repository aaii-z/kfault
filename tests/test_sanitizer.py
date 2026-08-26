import unittest

from kfault.sanitizer import parse_sanitizer_output

# (output, exit code, expected status, expected error count)
CASES = (
    ("ERROR SUMMARY: 0 errors", 0, "clean", 0),
    ("Invalid __global__ read\nERROR SUMMARY: 2 errors", 86, "violation", 2),
    ("RuntimeError: CUDA error", 1, "kernel_error", None),
    ("Device not supported", 1, "unavailable", None),
    ("unknown failure", 2, "tool_error", None),
)


class SanitizerParsingTests(unittest.TestCase):
    def test_status_and_error_count(self):
        for output, code, status, errors in CASES:
            with self.subTest(status=status):
                result = parse_sanitizer_output(output, code)
                self.assertEqual(result["status"], status)
                if errors is not None:
                    self.assertEqual(result["errors"], errors)


if __name__ == "__main__":
    unittest.main()
