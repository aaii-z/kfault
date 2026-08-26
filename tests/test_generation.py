import unittest

from kfault.generation import generation_config, generation_key, legacy_matches

BASE = dict(
    reasoning_effort="high",
    level=1,
    problem_id=3,
    sample=0,
    temperature=0.0,
    dataset_revision="rev",
    prompt="PROMPT",
)

LEGACY = {
    "model_id": "zai.glm-5",
    "arm": "think",
    "sample": 0,
    "temperature": 0.0,
    "prompt": "PROMPT",
    "raw_response": "code",
}
REQUEST = {"arm": "think", "sample": 0, "temperature": 0.0, "prompt": "PROMPT"}


class GenerationKeyTests(unittest.TestCase):
    def test_generation_keys_and_legacy_import(self):
        # pinned: adding repair must not invalidate already-paid generations
        self.assertEqual(
            generation_key(generation_config(**BASE)), "138d93d8dd21c3d40c57"
        )
        config = generation_config(**BASE, turn=0, feedback="ignored")
        self.assertNotIn("turn", config)
        self.assertNotIn("feedback_sha256", config)

        key = lambda **kw: generation_key(generation_config(**BASE, **kw))  # noqa: E731
        self.assertNotEqual(key(turn=1, feedback="a"), key(turn=1, feedback="b"))
        self.assertNotEqual(key(turn=1, feedback="a"), key(turn=2, feedback="a"))

        self.assertTrue(legacy_matches(LEGACY, **REQUEST))
        self.assertFalse(legacy_matches(LEGACY, **{**REQUEST, "prompt": "OTHER"}))
        self.assertFalse(legacy_matches({**LEGACY, "raw_response": None}, **REQUEST))


if __name__ == "__main__":
    unittest.main()
