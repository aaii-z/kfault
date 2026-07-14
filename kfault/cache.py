"""Stable cache keys for generation and classification artifacts."""

import hashlib
import json
import os
import tempfile


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def cache_key(config: dict, *, length: int = 20) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def write_json_atomic(path: str, value) -> None:
    """Replace a JSON cache entry atomically after a complete write."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(value, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
