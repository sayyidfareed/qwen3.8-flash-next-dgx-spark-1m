# Qwen3.8-Flash-Next 1M Context on GX10

Last updated: 2026-08-28

## Outcome

The GX10 now serves Qwen3.8-Flash-Next with a practical one-million-token
window and native one-token MTP speculative decoding. The final profile:

- advertises `max_model_len=1,000,000`;
- allocates KV capacity for 1,095,163 tokens (1.10 concurrent 1M requests);
- retrieved all five needles from a 989,734-token prompt and generated 67
  answer tokens, for 989,801 total tokens;
- scored 156/164 HumanEval, 152/164 HumanEval+ Mini, and 34/34 on the coding
  microbenchmark;
- sustained 26.712 tok/s median decode across five 512-token coding prompts;
- used 3.512 GiB minimum available system memory during the near-1M run and
  changed swap-free by only -0.010 GiB;
- runs as a single-sequence long-context profile on port `11002` without
  changing the NVIDIA OEM service on port `11000`.

Qwen documents a native 262,144-token window and a 1,000,000-token YaRN
extension using factor 4. The upstream DGX project validated 500K but describes
larger configurations as memory-limited. This result extends that project by
controlling the memory-mapped PLE table's Linux page-cache behavior.

Sources:

- [Qwen ultra-long-text configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#processing-ultra-long-texts)
- [Upstream Qwen3.8-Flash-DGX repository](https://github.com/blazux/qwen3.8-Flash-DGX)
- [Upstream long-context limitations](https://github.com/blazux/qwen3.8-Flash-DGX/blob/main/docs/HOW-IT-WORKS.md#long-context-what-works-and-what-does-not)

## Model and runtime identity

| Component | Identity |
|---|---|
| Host | NVIDIA GX10 / GB10, 121.63 GiB usable unified memory |
| Target checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Checkpoint revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Checkpoint size | 126 GiB, 419 files, verified offline |
| Runtime source | `blazux/qwen3.8-Flash-DGX` |
| Runtime commit | `d2854bfff0a0b6f46984b0941ed1db6010031295` |
| Experimental image | `qwen38-flash-dgx:1m-exp2` |
| Image digest | `sha256:eb4e7977dbe296156c0132905312393fb95ac0d09305d0686f73ab8144b409a5` |
| Served model ID | `qwen3.8-flash-next` |
| Endpoint | `http://gx10:11002/v1` |

This is a separate RadixArk checkpoint/runtime validation. Its retrieval and
coding-quality scores were measured independently and must not be silently
substituted for the `starkweatherdigital` checkpoint's results.

## Tested configurations

| Profile | Context | YaRN | MTP draft tokens | Sequences | Memory utilization | Prefill batch | PLE trim watermark | Minimum trim gather | KV capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Upstream baseline | 500,000 | 4x | 2 | 8 | 0.850 | 8,192 | Disabled | — | 791,411 |
| 1M safety profile | 1,000,000 | 4x | 0 | 1 | 0.900 | 1,024 | 8 GiB | Every gather | 1,368,924 |
| **Recommended 1M profile** | **1,000,000** | **4x** | **1** | **1** | **0.905** | **1,024** | **8 GiB** | **1,024 rows** | **1,095,163** |

The final launch parameters are:

```text
YARN=1 CTX=1000000 SEQS=1 GPU_MEM=0.905 MTP=1 KV_DTYPE=auto
PREWARM=0 BATCH_TOKENS=1024 PLE_TRIM_MIB=8192 PLE_TRIM_MIN_ROWS=1024
```

## Long-context retrieval results

| Profile | Target | Actual prompt | Output | Total tokens | Needles | Wall time | Prompt-equivalent throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| 500K upstream baseline | 450K | 452,285 | 67 | 452,352 | 5/5 | 315.346 s | 1,434.25 tok/s |
| 1M safety, intermediate | 750K | 753,603 | 67 | 753,670 | 5/5 | 673.883 s | 1,118.30 tok/s |
| 1M safety, near limit | 985K | 989,734 | 67 | 989,801 | 5/5 | 949.506 s | 1,042.37 tok/s |
| **Recommended 1M + MTP1** | **985K** | **989,734** | **67** | **989,801** | **5/5** | **998.188 s** | **991.53 tok/s** |

“Prompt-equivalent throughput” is prompt tokens divided by total client wall
time. It is intentionally conservative: the denominator also includes the
67-token answer, request overhead, and response transfer. It is not the
server's isolated prefill-kernel throughput.

The harness generates deterministic but varied archive records and places five
unique values at approximately 5%, 25%, 50%, 75%, and 95% of the prompt. The
model must return all five values in order. Thinking is disabled, temperature
is 0, and only one request runs at a time.

## Coding quality

The standard suite was rerun because this RadixArk checkpoint differs from the
previously ranked `starkweatherdigital` Flash-Next checkpoint.

| Checkpoint / profile | HumanEval | HumanEval+ Mini | Microbenchmark | Generation failures | Generation wall time |
|---|---:|---:|---:|---:|---:|
| `starkweatherdigital`, standard MTP1 | 157/164 (95.7%) | 155/164 (94.5%) | 34/34 | 0 | 794 s |
| **RadixArk, 1M YaRN + MTP1** | **156/164 (95.1%)** | **152/164 (92.7%)** | **34/34** | **0** | **1,590 s** |

Both EvalPlus columns cover all 164 tasks with one sample per task. Generation
used temperature 0, top-p 0.95, a 768-token cap, `enable_thinking=false`, and
`tool_choice=none`. Three fixed clients submitted disjoint shards, but the 1M
server's one-sequence limit meant only one generation was active at a time. The
merged artifact contains 164 unique task IDs and no failed API generations.
Evaluation used the pinned `ganler/evalplus:v0.3.1` container with networking
disabled and the cached HumanEvalPlus Mini `v0.1.10` dataset.

The 1M checkpoint loses one base-test pass and three HumanEval+ Mini passes
relative to the other Flash-Next checkpoint. That modest quality cost is real
enough to keep the entries separate, although 92.7% HumanEval+ Mini and a
perfect executable microbenchmark remain strong.

At task level, RadixArk newly passes the base test for `HumanEval/83` but loses
the base passes for `HumanEval/103` and `HumanEval/140`. The three additional
HumanEval+ Mini losses are `HumanEval/103`, `HumanEval/124`, and
`HumanEval/140`; it gains no plus-only task over the other checkpoint.

## Short-prompt serving performance

The comparable speed screen uses five sequential, streaming, 512-token coding
completions in Python, Rust, TypeScript, CUDA C++, and Go.

| Profile | Median decode | Mean decode | Minimum decode | Median TTFT | First-trial TTFT | Median effective completion |
|---|---:|---:|---:|---:|---:|---:|
| 1M safety, MTP0 | 13.937 tok/s | 13.958 tok/s | 13.911 tok/s | 0.287 s | 0.294 s | 13.843 tok/s |
| **Recommended 1M, MTP1** | **26.712 tok/s** | **27.028 tok/s** | **26.630 tok/s** | **0.257 s** | 0.896 s | **26.352 tok/s** |

The recommended profile nearly doubles median decode speed. Its first request
has a larger warm-up penalty, while its median TTFT is slightly better.

## Memory behavior

| Profile | Loaded-model memory | Consumed before KV | Peak activation | KV allocation | Minimum available during near-1M | Swap-free change |
|---|---:|---:|---:|---:|---:|---:|
| 500K baseline | 79.92 GiB | 80.46 GiB | 1.78 GiB | 21.14 GiB | — | — |
| 1M safety, MTP0 | 74.04 GiB | 75.14 GiB | 1.88 GiB | 32.45 GiB | 4.405 GiB | +0.002 GiB |
| **Recommended 1M, MTP1** | **79.05 GiB** | **79.75 GiB** | **1.88 GiB** | **28.44 GiB** | **3.512 GiB** | **-0.010 GiB** |

The MTP1 memory trace has 207 five-second samples. Available memory ranged from
3.512 to 4.231 GiB and cached memory from 3.065 to 4.072 GiB. A 10 MiB swap-free
change is operationally negligible and, importantly, does not scale with the
near-million-token KV growth.

## Why the patch is needed

The model's large PLE lookup table is memory-mapped instead of permanently
resident with the GPU weights. Linux normally keeps recently read mmap pages in
its page cache and may perform readahead. PLE access is hash-selected and
effectively random, so those cached/readahead pages compete with the expanding
KV cache without providing enough reuse at ultra-long context.

The patch:

1. marks every PLE mmap as random access with `MADV_RANDOM`;
2. watches Linux `MemAvailable` after large PLE gathers;
3. below the configured watermark, releases clean PLE pages with
   `MADV_DONTNEED` and `POSIX_FADV_DONTNEED`;
4. skips trim checks for gathers smaller than 1,024 rows, preserving fast
   token-by-token decode;
5. lowers chunked prefill from 8,192 to 1,024 tokens and limits serving to one
   sequence so temporary activation memory stays bounded.

The returned tensors are fresh copies, so dropping the clean backing pages does
not change gathered values. The synthetic CPU test exercises sizes from one row
through 131,072 rows, the placeholder FP8 path, range checks, prewarming, and
forced trimming; every result remained bit-identical.

## Limits and interpretation

- The API's maximum is prompt plus generated tokens. The validated 989,801-token
  request leaves 10,199 tokens below the configured 1,000,000-token limit.
- This is a deterministic retrieval probe, not a comprehensive long-context
  reasoning benchmark. It proves allocation, execution, positional retrieval,
  and memory stability at the target scale.
- The profile is intentionally single-sequence. Concurrent long-context
  requests would need more KV memory than this machine has.
- MTP1 reduces prompt-equivalent long-prefill throughput by about 4.9% versus
  MTP0 in these two near-limit runs, while improving short decode by about 91.7%.
  That is the better interactive coding tradeoff.
- No other large model should be co-located while this 1M profile is active.

## Reproducibility artifacts

Artifacts are included under `results/` in this repository.

| Artifact | SHA-256 |
|---|---|
| `benchmarks/long_context_probe.py` | `ed6931c069a4cdbfe3098dd5272c00aaff3edebca69d6b0e393b12d6f5057e6d` |
| `baseline-500k-probe-450k.json` | `106c3d19e68749086e669d9e19b7eb42dacc6a7cfac562ccfb3f408802c31f36` |
| `experimental-1m-probe-750k.json` | `c8e5ff52c31ac69cc10826dc2116003f9bba323fefa6f69602ade91db3776766` |
| `experimental-1m-probe-985k.json` | `be69d03c0d1f268024f69ece1edb9e066e40bf487d0bb9c3270473b0506f59f8` |
| `experimental-1m-mtp1-probe-985k.json` | `241301d72c5031ca4cb389cdd697836372a61a4cea7e58f19ca86553973eb0c3` |
| `experimental-1m-streaming.json` | `72b91d818ad09e17977c84085b51c4c31964d1375d679e08b5ef9f7dde89dffc` |
| `experimental-1m-mtp1-streaming.json` | `3f2a3cb624e70e28657db6fd8ea7a13a784ef3fe520eebea06d56e4d809e7b4e` |
| `experimental-1m-mtp1-memory.csv` | `8904107226854d0c3a71859e7e9334a7090e003b28802e07fb9b1c6cb2dc4d75` |
| `qwen38-flash-next-dgx-1m.patch` | `567365cd9945c01760a2f86c38893563ccfe1865c8d16cf7a0cb49b492762042` |
| `evalplus-samples.jsonl` | `e20978d47e559c39d00a845dd7dee2a51bdfbf33e32ee1acfb3d7056f0848c29` |
| `evalplus-results.json` | `f2ea40b094e6330811291fa495ce2ced4039a223f8a7061d8c7b6a3bd804cfc2` |
| `microbench.json` | `1ef1d6ce79b3ad969c28fab0065cb4a47d706e71c0a490635b0198f109acc2ff` |
