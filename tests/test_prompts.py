import unittest

from kfault.prompts import strip_fences, strip_reasoning

# (input, expected) for the response parser. A truncated reasoning block must
# yield nothing rather than leaking reasoning text into the compiled kernel.
CASES = (
    (
        strip_reasoning,
        "<reasoning>hidden ```bad```</reasoning>\n```python\nclass ModelNew: pass\n```",
        "```python\nclass ModelNew: pass\n```",
    ),
    (strip_reasoning, "prefix<reasoning>truncated", ""),
    (strip_fences, "```python3\nclass ModelNew: pass\n```", "class ModelNew: pass"),
    (
        strip_fences,
        "```text\nexample\n```\n```py\nclass ModelNew: pass\n```",
        "class ModelNew: pass",
    ),
)


class PromptParsingTests(unittest.TestCase):
    def test_response_parsing(self):
        for function, text, expected in CASES:
            with self.subTest(function=function.__name__, text=text[:40]):
                self.assertEqual(function(text), expected)


if __name__ == "__main__":
    unittest.main()
