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


class TestGenerationConfig(unittest.TestCase):
    def test_turn_zero_key_is_the_one_shot_key(self):
        """Repair support must not invalidate already-paid generations."""
        self.assertEqual(
            generation_key(generation_config(**BASE)),
            "138d93d8dd21c3d40c57",
        )

    def test_turn_zero_carries_no_repair_fields(self):
        config = generation_config(**BASE, turn=0, feedback="ignored")
        self.assertNotIn("turn", config)
        self.assertNotIn("feedback_sha256", config)

    def test_feedback_changes_the_key_within_a_turn(self):
        one = generation_key(generation_config(**BASE, turn=1, feedback="a"))
        two = generation_key(generation_config(**BASE, turn=1, feedback="b"))
        self.assertNotEqual(one, two)

    def test_turn_changes_the_key(self):
        one = generation_key(generation_config(**BASE, turn=1, feedback="a"))
        two = generation_key(generation_config(**BASE, turn=2, feedback="a"))
        self.assertNotEqual(one, two)


class TestLegacyMatches(unittest.TestCase):
    def legacy(self, **overrides):
        entry = {
            "model_id": "zai.glm-5",
            "arm": "think",
            "sample": 0,
            "temperature": 0.0,
            "prompt": "PROMPT",
            "raw_response": "code",
        }
        entry.update(overrides)
        return entry

    def request(self, **overrides):
        fields = {"arm": "think", "sample": 0, "temperature": 0.0, "prompt": "PROMPT"}
        fields.update(overrides)
        return fields

    def test_exact_match_is_imported(self):
        self.assertTrue(legacy_matches(self.legacy(), **self.request()))

    def test_different_prompt_is_rejected(self):
        self.assertFalse(legacy_matches(self.legacy(), **self.request(prompt="OTHER")))

    def test_missing_response_is_rejected(self):
        entry = self.legacy(raw_response=None)
        self.assertFalse(legacy_matches(entry, **self.request()))


if __name__ == "__main__":
    unittest.main()
