#!/usr/bin/env bash
# Validated one-million-token profile for one 128 GB NVIDIA DGX Spark / ASUS GX10.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NAME="${NAME:-qwen38-flash-1m}"
export IMAGE="${IMAGE:-qwen38-flash-dgx:1m}"
export PORT="${PORT:-11002}"
export CTX="${CTX:-1000000}"
export YARN="${YARN:-1}"
export SEQS="${SEQS:-1}"
export GPU_MEM="${GPU_MEM:-0.905}"
export MTP="${MTP:-1}"
export KV_DTYPE="${KV_DTYPE:-auto}"
export PREWARM="${PREWARM:-0}"
export BATCH_TOKENS="${BATCH_TOKENS:-1024}"
export PLE_TRIM_MIB="${PLE_TRIM_MIB:-8192}"
export PLE_TRIM_MIN_ROWS="${PLE_TRIM_MIN_ROWS:-1024}"

exec "$script_dir/serve.sh"
