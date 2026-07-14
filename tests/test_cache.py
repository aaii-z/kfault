import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kfault.cache import cache_key, content_hash, write_json_atomic


class CacheKeyTests(unittest.TestCase):
    def test_dictionary_order_does_not_change_key(self):
        self.assertEqual(cache_key({"a": 1, "b": 2}), cache_key({"b": 2, "a": 1}))

    def test_configuration_change_changes_key(self):
        self.assertNotEqual(cache_key({"seed": 1}), cache_key({"seed": 2}))

    def test_content_hash_is_stable(self):
        self.assertEqual(content_hash("kernel"), content_hash("kernel"))

    def test_atomic_json_write(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json_atomic(str(path), {"category": "pass"})
            self.assertEqual(path.read_text(), '{\n  "category": "pass"\n}')


if __name__ == "__main__":
    unittest.main()
