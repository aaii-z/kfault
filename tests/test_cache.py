import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kfault.cache import cache_key, content_hash, write_json_atomic


class CacheTests(unittest.TestCase):
    def test_keys_are_stable_and_writes_are_atomic(self):
        # order must not matter, but any field that decides an outcome must
        self.assertEqual(cache_key({"a": 1, "b": 2}), cache_key({"b": 2, "a": 1}))
        self.assertNotEqual(cache_key({"seed": 1}), cache_key({"seed": 2}))
        self.assertEqual(content_hash("kernel"), content_hash("kernel"))
        self.assertNotEqual(content_hash("kernel"), content_hash("kernel "))

        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json_atomic(str(path), {"category": "pass"})
            self.assertEqual(path.read_text(), '{\n  "category": "pass"\n}')


if __name__ == "__main__":
    unittest.main()
