"""Generation artifacts: cache config, legacy import, and the paid API call."""

import json

from kfault.cache import cache_key, content_hash
from kfault.config import (
    BEDROCK_URL,
    GENERATION_CACHE_VERSION,
    MAX_TOKENS,
    MODEL_ID,
    OPENAI_VERSION,
)


def generation_config(
    *,
    reasoning_effort: str,
    level: int,
    problem_id: int,
    sample: int,
    temperature: float,
    dataset_revision: str,
    prompt: str,
    turn: int = 0,
    feedback: str = "",
) -> dict:
    config = {
        "schema": GENERATION_CACHE_VERSION,
        "model_id": MODEL_ID,
        "reasoning_effort": reasoning_effort,
        "level": level,
        "problem_id": problem_id,
        "sample": sample,
        "temperature": temperature,
        "dataset_revision": dataset_revision,
        "prompt_sha256": content_hash(prompt),
        "openai_version": OPENAI_VERSION,
    }
    # Turn 0 keeps the original key so one-shot generations stay cached.
    if turn:
        config["turn"] = turn
        config["feedback_sha256"] = content_hash(feedback)
    return config


def generation_key(config: dict) -> str:
    return cache_key(config)


def artifact_key(raw_response, generated_at) -> str:
    """Identity of the response itself, independent of the request config."""
    return content_hash(
        json.dumps(
            {"raw_response": raw_response, "generated_at": generated_at},
            sort_keys=True,
        )
    )


def legacy_matches(legacy: dict, *, arm, sample, temperature, prompt) -> bool:
    """Only import an old combined-cache entry on an exact request match."""
    return (
        legacy.get("model_id") == MODEL_ID
        and legacy.get("arm") == arm
        and legacy.get("sample", 0) == sample
        and legacy.get("temperature", 0.0) == temperature
        and legacy.get("prompt") == prompt
        and legacy.get("raw_response") is not None
    )


def from_legacy(legacy: dict, legacy_path: str, key: str, config: dict) -> dict:
    """Carry a paid response forward without its stale classification."""
    generated_at = legacy.get("generated_at") or legacy.get("labelled_at")
    generation = {
        "generation_key": key,
        "generation_config": config,
        "prompt": legacy["prompt"],
        "raw_response": legacy["raw_response"],
        "finish_reason": legacy.get("finish_reason"),
        "usage": legacy.get("usage"),
        "gen_seconds": legacy.get("gen_seconds"),
        "generated_at": generated_at,
        "migrated_from": legacy_path,
    }
    generation["generation_artifact_key"] = artifact_key(
        generation["raw_response"], generated_at
    )
    return generation


def request(
    prompt: str,
    *,
    api_key: str,
    reasoning_effort: str,
    temperature: float,
    key: str,
    config: dict,
) -> dict:
    """Call the model once and package the response as a generation artifact."""
    import time
    from datetime import datetime, timezone

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=BEDROCK_URL)
    started = time.monotonic()
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        extra_body={"reasoning_effort": reasoning_effort},
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    generation = {
        "generation_key": key,
        "generation_config": config,
        "prompt": prompt,
        "raw_response": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage.model_dump() if response.usage else None,
        "gen_seconds": round(time.monotonic() - started, 2),
        "generated_at": generated_at,
    }
    generation["generation_artifact_key"] = artifact_key(
        generation["raw_response"], generated_at
    )
    return generation
