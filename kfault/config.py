"""Run configuration, importable without Modal."""

import os

# Bedrock, OpenAI-compatible. Both arms share weights; only reasoning differs.
# Models needing a real on/off switch: zai.glm-5/4.7, qwen3-32b/235b,
# deepseek.v3.2. Rejected (reason regardless of the switch):
# qwen3-coder-480b, kimi-k2-thinking, minimax-m2.1, gpt-oss-*.
BEDROCK_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"

# Part of the generation cache key, so two models never share cached output.
# Must be forwarded into the image (pipeline.py .env): generation runs
# container-side and would otherwise fall back to the default.
MODEL_ID = os.environ.get("KFAULT_MODEL", "zai.glm-5")

ARMS = {
    "think": "high",  # reasoning ON
    "direct": "none",  # reasoning OFF
}

GPU = os.environ.get("KFAULT_GPU", "T4")
# amd64 digest of nvidia/cuda:12.4.0-devel-ubuntu22.04 (arm64 digests fail).
CUDA_IMAGE = (
    "nvidia/cuda@sha256:14c54fad24b376ab78a70e1ef6595a2b7c8cdbf187e4f9b76de99a"
    "926fb62460"
)
DATASET_REVISION = "ca1464e5d4723fd14f87b8e68306aaa1fe66b81b"

# Independent keys, so a classifier change never triggers a regeneration.
GENERATION_CACHE_VERSION = 1
CLASSIFIER_CACHE_VERSION = 4  # v4: clone_tree preserves strides (non_contig)

# A classification makes ~90 forward calls, so the container timeout is really
# a per-forward budget. Stage 2a' times one warm call and rejects slow kernels
# up front; without it they exhaust the container and produce no row at all.
DEFAULT_FORWARD_TIME_LIMIT_S = 5.0
FORWARD_TIME_LIMIT_S = float(
    os.environ.get("KFAULT_FORWARD_LIMIT_S", DEFAULT_FORWARD_TIME_LIMIT_S)
)

# Backstop for a kernel that hangs inside its first forward call, which
# FORWARD_TIME_LIMIT_S cannot catch. A hung CUDA kernel cannot be interrupted
# from Python, so the container timeout is the only lever and the pipeline
# turns it into a labelled row. Sized from data: the slowest legitimate
# classification observed was 544s, mostly nvcc compile time. 300s would have
# mislabelled 27 valid records.
CLASSIFIER_TIMEOUT_S = int(os.environ.get("KFAULT_CLASSIFIER_TIMEOUT_S", 900))

# Battery probes plus benchmark warmup and iterations.
FORWARD_CALLS_PER_CLASSIFICATION = 90

# The two limits are not independent, and slower hardware needs both raised.
# A T4 runs ~3.8x slower than an L4, so a fixed budget fails kernels for being
# on a slower device rather than for being wrong. Raising only the forward
# budget moves the failure to a container timeout, so refuse the pair rather
# than produce data that looks fine and is not.
_needed = FORWARD_TIME_LIMIT_S * FORWARD_CALLS_PER_CLASSIFICATION
if CLASSIFIER_TIMEOUT_S < _needed:
    raise ValueError(
        f"KFAULT_FORWARD_LIMIT_S={FORWARD_TIME_LIMIT_S}s allows "
        f"{FORWARD_CALLS_PER_CLASSIFICATION} forward calls = {_needed:.0f}s, but "
        f"KFAULT_CLASSIFIER_TIMEOUT_S={CLASSIFIER_TIMEOUT_S}s would kill the "
        f"container first. Raise the timeout to at least {_needed:.0f}s plus "
        f"compile time (try {int(_needed * 1.5)})."
    )

PINNED_PACKAGES = (
    "torch==2.6.0",
    "ninja==1.11.1.1",
    "openai==2.41.1",
    "datasets==5.0.0",
    "modal==1.5.0",
)
OPENAI_VERSION = "2.41.1"

RESULTS_DIR = "/results"
MAX_TOKENS = 16000  # the think arm spends tokens before the code
