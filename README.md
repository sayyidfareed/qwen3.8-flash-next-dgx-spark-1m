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

# One million tokens on one 128 GB DGX Spark

On an ASUS GX10, this Qwen3.8-Flash-Next setup handled a 989,734-token prompt,
returned all five buried values, and generated 67 more tokens. Short prompts
decoded at a 26.7 tok/s median, and the same profile scored 92.7% on HumanEval+
Mini.

The upstream
[Qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
project got this model to 500K on the same class of machine. Its original memory
policy ran out of room before 1M. I kept the model and its mmap-based PLE
offload, then changed how those mapped pages are handled under memory pressure.

The PLE lookup table stays memory-mapped on NVMe. Linux is told to expect random
access, and clean PLE pages are released before they squeeze the growing KV
cache. That was enough to serve a near-limit request while keeping native MTP
speculative decoding enabled.

> This repository contains serving code and benchmark artifacts, **not model
> weights**. The code is Apache-2.0. The Qwen/RadixArk checkpoint has separate
> terms; review the source model card before use.

## Measured on the GX10

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
- about 130 GB of fast local storage; NVMe is strongly recommended
- roughly 10 minutes for a cold model load on the tested machine

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

## The 1M serving profile

`scripts/serve-1m.sh` is a pinned wrapper around the upstream serve script:

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

Why these values:

- **YaRN factor 4** follows Qwen's published recipe for extending the native
  262,144-token window to one million tokens.
- **One sequence** reserves the KV budget for one near-million-token request.
- **1,024-token chunked prefill** bounds temporary activation memory.
- **MTP1** nearly doubles decode versus the MTP0 safety profile.
- **8 GiB PLE watermark** starts evicting clean, re-readable PLE pages before
  unified memory becomes critical.
- **1,024-row trim threshold** keeps the trim path out of single-token decode.

## Where the extra memory came from

Flash-Next has an enormous n-gram/PLE lookup table. The upstream recipe already
memory-maps that table instead of permanently loading it beside the GPU
weights.

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

## Short-context quality check

We reran the complete short coding suite because this recipe uses the RadixArk
checkpoint, while the strongest earlier Flash-Next result used a different
`starkweatherdigital` export.

| Checkpoint / profile | HumanEval | HumanEval+ Mini | Microbench | Decode |
|---|---:|---:|---:|---:|
| `starkweatherdigital`, standard MTP1 | 157/164 | 155/164 | 34/34 | 27.633 tok/s |
| **RadixArk, 1M YaRN + MTP1** | **156/164** | **152/164** | **34/34** | **26.712 tok/s** |

The 1M run is slightly behind the earlier result. Both the checkpoint and the
serving profile changed, so this comparison does not isolate the effect of YaRN.
The two rows should be treated as separate deployment targets.

## Run the 989K test

Start with the 5K smoke probe:

```bash
python3 benchmarks/long_context_probe.py \
  --base-url http://localhost:11002 \
  --target-tokens 5000 \
  --output results/smoke-5k.json
```

The near-limit run takes about 17 minutes on the tested machine and sends a
multi-megabyte request. Run it after the smoke test succeeds:

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

## Operational limits

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

If you try this on another Spark, please open an issue with the KV capacity,
minimum `MemAvailable`, storage model, and result JSON. I would especially like
to see results from other NVMe drives and OEM GB10 systems.
