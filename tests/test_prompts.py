import unittest

from kfault.prompts import strip_fences, strip_reasoning


class PromptParsingTests(unittest.TestCase):
    def test_completed_reasoning_is_removed(self):
        text = (
            "<reasoning>hidden ```bad```</reasoning>\n"
            "```python\nclass ModelNew: pass\n```"
        )
        self.assertEqual(strip_reasoning(text), "```python\nclass ModelNew: pass\n```")

    def test_unclosed_reasoning_is_empty(self):
        self.assertEqual(strip_reasoning("prefix<reasoning>truncated"), "")

    def test_python3_fence_tag_is_removed(self):
        text = "```python3\nclass ModelNew: pass\n```"
        self.assertEqual(strip_fences(text), "class ModelNew: pass")

    def test_model_new_block_is_preferred(self):
        text = "```text\nexample\n```\n```py\nclass ModelNew: pass\n```"
        self.assertEqual(strip_fences(text), "class ModelNew: pass")


if __name__ == "__main__":
    unittest.main()
