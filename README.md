---
license: apache-2.0
base_model:
- RadixArk/Qwen3.8-Flash-Next-NVFP4
library_name: vllm
pipeline_tag: text-generation
tags:
- qwen3.8
- dgx-spark
- gb10
- long-context
- 1m-context
- nvfp4
- vllm
- speculative-decoding
---

# 1,000,000 tokens. One 128 GB DGX Spark.

**Qwen3.8-Flash-Next, 989,801 tokens end to end, 5/5 retrieval, 26.7 tok/s
decode, and 92.7% HumanEval+ Mini—on one NVIDIA DGX Spark / ASUS GX10.**

The excellent upstream
[Qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
recipe proved 500K context and accurately called 1M out of reach with its
original memory policy. This repository changes that policy, not the model.

It keeps the giant PLE lookup table memory-mapped on NVMe, tells Linux that the
access pattern is random, and releases clean PLE pages before they crowd out the
growing KV cache. The result is a real, tested one-million-token serving profile
with native MTP speculative decoding still enabled.

> This repository contains serving code and benchmark artifacts, **not model
> weights**. The code is Apache-2.0. The Qwen/RadixArk checkpoint has separate
> terms; review the source model card before use.

## The proof, not the promise

| Test | Result |
|---|---:|
| Advertised context | 1,000,000 tokens |
| Allocated KV capacity | 1,095,163 tokens |
| Validated request | 989,734 prompt + 67 output = **989,801 tokens** |
| Distributed needle retrieval | **5/5** at 5%, 25%, 50%, 75%, 95% |
| Median decode, five coding languages | **26.712 tok/s** |
| Minimum decode | **26.630 tok/s** |
| Median TTFT | **0.257 s** |
| HumanEval | **156/164 (95.1%)** |
| HumanEval+ Mini | **152/164 (92.7%)** |
| Executable coding microbenchmark | **34/34** |
| Minimum available memory during 989K prefill | **3.512 GiB** |
| Swap-free change during 989K prefill | **-0.010 GiB** |

Hardware: NVIDIA GX10 / GB10 with 121.63 GiB usable unified memory. The model
service was the only large workload on the box.

## Quickstart

Requirements:

- DGX Spark, ASUS GX10, or compatible GB10 system with 128 GB unified memory
- NVIDIA container runtime and Docker
- roughly 130 GB of fast local storage; NVMe is strongly recommended
- patience for the initial checkpoint download and roughly 10-minute cold load

```bash
git clone https://github.com/sayyidfareed/qwen3.8-flash-next-dgx-spark-1m.git
cd qwen3.8-flash-next-dgx-spark-1m

docker build -t qwen38-flash-dgx:1m .
scripts/download-weights.sh
scripts/serve-1m.sh
docker logs -f qwen38-flash-1m
```

The server is ready when the log says `Application startup complete`. It exposes
an OpenAI-compatible API on port `11002` by default:

```bash
curl http://localhost:11002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role":"user","content":"Write a lock-free ring buffer in Rust."}],
    "temperature": 0,
    "max_tokens": 1024,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## The exact 1M profile

`scripts/serve-1m.sh` is only a pinned wrapper around the upstream serve script:

```text
CTX=1000000
YARN=1
SEQS=1
GPU_MEM=0.905
MTP=1
KV_DTYPE=auto
PREWARM=0
BATCH_TOKENS=1024
PLE_TRIM_MIB=8192
PLE_TRIM_MIN_ROWS=1024
```

Why each non-default matters:

- **YaRN factor 4** follows Qwen's published recipe for extending the native
  262,144-token window to one million tokens.
- **One sequence** reserves the KV budget for one near-million-token request.
- **1,024-token chunked prefill** bounds temporary activation memory.
- **MTP1** nearly doubles decode versus the MTP0 safety profile.
- **8 GiB PLE watermark** starts evicting clean, re-readable PLE pages before
  unified memory becomes critical.
- **1,024-row trim threshold** keeps the trim path out of single-token decode.

## The one idea that unlocked 1M

Flash-Next has an enormous n-gram/PLE lookup table. The upstream recipe already
made the crucial move: mmap the table instead of permanently loading it beside
the GPU weights.

At ultra-long context, Linux's page cache becomes the next bottleneck. PLE rows
are hash-selected, so normal sequential readahead is mostly waste. Meanwhile,
every cached PLE page competes with a KV cache that is growing toward 30 GiB.

This patch adds three controls:

1. `MADV_RANDOM` disables inappropriate sequential readahead.
2. Below a memory watermark, `MADV_DONTNEED` and `POSIX_FADV_DONTNEED` release
   clean PLE pages that can always be read again from NVMe.
3. Trimming only runs after large gathers, not during token-by-token decode.

The gather returns a fresh tensor copy before eviction, so released mmap pages
cannot change the current result. The included CPU regression verifies
bit-identical gathers from one through 131,072 rows, the placeholder FP8 path,
range checks, prewarming, and forced trimming.

## Quality: what did 1M cost?

We reran the complete short coding suite because this recipe uses the RadixArk
checkpoint, while the strongest earlier Flash-Next result used a different
`starkweatherdigital` export.

| Checkpoint / profile | HumanEval | HumanEval+ Mini | Microbench | Decode |
|---|---:|---:|---:|---:|
| `starkweatherdigital`, standard MTP1 | 157/164 | 155/164 | 34/34 | 27.633 tok/s |
| **RadixArk, 1M YaRN + MTP1** | **156/164** | **152/164** | **34/34** | **26.712 tok/s** |

That is a modest but measurable quality difference. Because both the checkpoint
and serving profile changed, it would be dishonest to blame YaRN alone. Treat
the 1M profile as a separate deployment target, not a drop-in benchmark alias.

## Reproduce the headline test

First run the cheap 5K smoke probe:

```bash
python3 benchmarks/long_context_probe.py \
  --base-url http://localhost:11002 \
  --target-tokens 5000 \
  --output results/smoke-5k.json
```

The near-limit proof takes about 17 minutes on the validated machine and sends a
multi-megabyte request. Run it only after the smoke test succeeds:

```bash
python3 benchmarks/long_context_probe.py \
  --base-url http://localhost:11002 \
  --target-tokens 985000 \
  --max-output-tokens 128 \
  --timeout 3000 \
  --output results/reproduction-985k.json
```

The prompt uses deterministic varied archive records and inserts five unique
needles at roughly 5%, 25%, 50%, 75%, and 95%. It is a retrieval and operational
stability test, not a comprehensive long-context reasoning benchmark.

## Limits you should understand

- **Prompt plus output must remain below one million.** The validated 989,801-
  token request leaves 10,199 tokens of headroom.
- **This is a single-sequence profile.** It is optimized for one huge context,
  not aggregate multi-user throughput.
- **Do not co-locate another large model.** CPU, page cache, GPU allocations, and
  KV all share the same 128 GB pool.
- **The PLE table stays on fast local storage.** Slow network storage will hurt
  page-fault latency.
- **The checkpoint is public but tagged as a candidate export.** Pin the tested
  revision instead of silently following future repository updates.
- Multimodal inputs were not part of this validation.

## Reproducibility

| Component | Pinned identity |
|---|---|
| Target checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Checkpoint revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Upstream runtime | `blazux/qwen3.8-Flash-DGX` |
| Upstream commit | `d2854bfff0a0b6f46984b0941ed1db6010031295` |
| Experimental image digest | `sha256:eb4e7977dbe296156c0132905312393fb95ac0d09305d0686f73ab8144b409a5` |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for methodology, memory traces,
artifact hashes, and the exact comparison profiles. See
[patches/dgx-spark-1m.patch](patches/dgx-spark-1m.patch) for the complete delta
from the pinned upstream commit.

## Credits

- Qwen team / Alibaba for Qwen3.8-Flash-Next.
- [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) for the
  NVFP4 checkpoint.
- [blazux](https://github.com/blazux/qwen3.8-Flash-DGX) for discovering and
  implementing the mmap-PLE approach that made Flash-Next practical on GB10.
- [jschmied](https://github.com/jschmied/qwen38-flash-next-gb10) for independent
  upstream reproduction and concurrency/offload investigation.
- vLLM and NVIDIA Model Optimizer for the serving and quantization stack.

If you reproduce 1M on another Spark, open an issue with your exact KV capacity,
minimum `MemAvailable`, storage model, and result JSON. One machine is a result;
multiple machines are a recipe.
