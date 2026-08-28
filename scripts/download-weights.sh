#!/usr/bin/env bash
# Download the pinned NVFP4 checkpoint (~126 GiB) into the Hugging Face cache.
# Resumable — safe to re-run if the connection drops.
#
#   scripts/download-weights.sh
#
# Needs roughly 135 GB free on the filesystem holding ~/.cache/huggingface.
set -euo pipefail

MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
REVISION="${REVISION:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
IMAGE="${IMAGE:-qwen38-flash-dgx:1m}"        # or any image containing the `hf` CLI
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

echo ">> downloading $MODEL@$REVISION into $HF_CACHE (resumable)"
# HF_HUB_DISABLE_XET=1: the Xet backend stalled on some Spark setups; plain HTTPS
# is reliable and saturates the link.
docker run --rm --name qwen38-dl \
  -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 \
  -v "$HF_CACHE:/hf" --entrypoint bash "$IMAGE" \
  -c "hf download '$MODEL' --revision '$REVISION' --max-workers 8"

echo ">> done. Verify with:  scripts/serve.sh"
