# Benchmark findings: Ollama on Apple Silicon vs vLLM on NVIDIA T4

October 16–17, 2026. Raw tables: `results/sweeps/*/sweep.md`. Per-request rows, vLLM logs, `/metrics` snapshots and per-second GPU samples sit next to them.

## Setup

| Run | Engine | Model | Hardware |
|---|---|---|---|
| `14b-p1` | Ollama, `OLLAMA_NUM_PARALLEL=1` | Qwen2.5-14B Q4_K_M (GGUF) | Apple Silicon Mac |
| `14b-p4` | Ollama, `OLLAMA_NUM_PARALLEL=4` | same | same |
| `t4-3b-fp16` | vLLM 0.31.0 | Qwen2.5-3B-Instruct, FP16 | 1× T4 (Kaggle) |
| `t4-7b-awq` | vLLM 0.31.0 | Qwen2.5-7B-Instruct-AWQ, 4-bit | 1× T4 |
| `t4x2-7b-fp16-tp2` | vLLM 0.31.0, TP=2 | Qwen2.5-7B-Instruct, FP16 | 2× T4 over PCIe |

Workload: closed-loop client, input ≈ 150 / 430 / 1,550 prompt tokens, 128 output tokens, temperature 0. Each prompt starts with a random tag so prefix caching cannot skip prefill. vLLM ran with its defaults: chunked prefill and prefix caching on, Triton attention backend (T4 has no FlashAttention), CUDA graphs captured, `--max-model-len 4096`, `--gpu-memory-utilization 0.90`.

Client and server shared one machine. The Mac and the T4 runs use different models, so Mac-vs-T4 rows compare behaviour, not speed.

## 1. Without spare compute, parallel slots do not add throughput (Mac)

One Ollama slot is a FIFO queue. TTFT grows by one full request per extra user, and throughput stays at about 23 tok/s. Little's law fits: at c=4, 4 users ÷ 0.18 req/s ≈ 22 s, and E2E measured 21.9 s.

Four slots batch the decode, but on this Mac a 4-sequence step costs about 3.6× a 1-sequence step:

| 14B, 148-token input, c=4 | 1 slot | 4 slots |
|---|---|---|
| TTFT p95 | 18.0 s | **2.6 s** |
| TPOT p50 | 39 ms | 143 ms |
| Throughput | 22.7 tok/s | 24.8 tok/s |

Slots spread latency more fairly and add almost no capacity. At 1,550-token input and c=4, four slots made E2E p95 worse (59 s → 73 s).

## 2. On a GPU, batched decode is nearly free until it is not (T4)

| 3B FP16, 148-token input | TPOT p50 | Throughput |
|---|---|---|
| c=4 | 35.1 ms | 107 tok/s |
| c=8 | 35.5 ms | 206 tok/s |
| c=16 | 40.8 ms | 331 tok/s |
| c=32 | 56.2 ms | **490 tok/s** |

From c=4 to c=8, TPOT is flat and throughput doubles. Decode is memory-bound: one weight read serves the whole batch. The knee sits between c=16 and c=32.

## 3. Long prompts cancel most of the batching gain

| 3B FP16, c=4 → c=32 | TPOT growth | Throughput gain |
|---|---|---|
| 148-token input | 1.6× | 4.6× |
| 434-token input | 2.4× | 3.4× |
| 1,553-token input | 5.5× | **1.6×** |

At about 12 input tokens per output token, prefill dominates and shares steps with decode. TTFT p50 vs p95 at 1,553 tokens and c=32 is 2.8 s vs 27.5 s: new prompts wait in vLLM's per-step token budget. KV memory was not the limit (32 × 2.2k ≈ 70k tokens of 188k available).

## 4. Capacity depends on the workload, not just the GPU

SLO: TPOT p95 < 50 ms and TTFT p95 < 1 s, on 3B FP16, 1× T4.

| Workload | Highest passing concurrency | Per-GPU throughput |
|---|---|---|
| ~150-token prompts | c=16 | ~330 tok/s, 2.6 req/s |
| ~430-token prompts | c=4 | ~95 tok/s |
| ~1,550-token prompts | none | — |

With an E2E p95 ≤ 6 s SLO on 7B AWQ: c=8 for short prompts (1.63 req/s), and only c=1 for 1,550-token prompts (0.17 req/s, 4% headroom). The cheapest fix for long prompts is sending fewer tokens (retrieve only the relevant context) or caching a shared prefix.

## 5. Quantization saves memory everywhere, speed only with the right kernels

| 7B, 1× T4 vs 3B FP16, 1× T4, c=32 | 148-token | 434-token | 1,553-token |
|---|---|---|---|
| 3B FP16 throughput | 490 | 320 | 78 tok/s |
| 7B AWQ throughput | 364 | 135 | 54 tok/s |

AWQ shrinks 7B to about 5.5 GB, which makes it fit one T4. On Turing, vLLM uses AWQ kernels that dequantize on every step (`quantization=auto_awq`, no Marlin), so AWQ scales worse than FP16 once compute matters. At c=1 it gave the steadiest TPOT of any run (30 ms).

## 6. Tensor parallel helps long-context work and latency, replicas help throughput

| 7B, 1,550-token input | AWQ, 1 GPU | FP16 TP=2, 2 GPUs |
|---|---|---|
| TTFT p95, c=1 | 1.93 s | **1.03 s** |
| TTFT p95, c=32 | 54.0 s | **26.7 s** |
| TPOT p50, c=32 | 514 ms | **282 ms** |
| Throughput per GPU, c=32 | **54** | 50 tok/s |
| TPOT p50, c=1 | **30 ms** | 42 ms |

TP splits prefill compute, so long prompts gain the most. Per-layer all-reduce over PCIe costs latency at c=1. For 148-token prompts at c=32, two AWQ replicas (≈ 728 tok/s) beat one TP=2 server (471 tok/s). Rule: replicas for throughput, TP for latency under load or for models that do not fit one GPU.

## 7. KV-cache math matched vLLM's own numbers

`bytes/token = 2 × layers × kv_heads × head_dim × 2 (FP16)`

| Model | KV per token | Predicted pool | vLLM reported |
|---|---|---|---|
| 3B FP16 (36 L, 2 KV heads) | 36 KB | ~170k tokens | **187,744** (45.8× at 4k) |
| 7B AWQ (28 L, 4 KV heads) | 56 KB | ~125k tokens | **113,744** (27.8× at 4k) |
| 7B FP16 TP=2 | 56 KB, split | ~150–160k | **192,096** (46.9× at 4k) |

AWQ quantizes weights only; the KV cache stays FP16, which is why 7B AWQ holds fewer tokens than 3B FP16 despite smaller weights.

## 8. Measurement caveat: Kaggle T4 clocks were throttled

On FP16 runs, c=1 was slower than c=4 (3B: 110 ms vs 35 ms TPOT), which batching cannot explain. GPU samples aligned to each cell show those c=1 cells ran at **~305–320 MHz**, the floor, against a 1,590 MHz maximum, at 100% utilization and ~57 W (below the 70 W cap). Across all runs the SM clock averaged roughly 400–1,000 MHz.

- Every T4 number here is a lower bound for a T4 held at full clock.
- The cause of the low clock (thermal vs power vs driver policy) is not measured. The notebook's sampler now records `temperature.gpu`, `power.limit` and `clocks_event_reasons.active`.
- In production, lock clocks (`nvidia-smi -lgc`) or watch DCGM clock and throttle metrics before trusting a benchmark.

## Open questions

- Same model on Mac vs T4: run `3b-p4` on the Mac.
- c=1 rerun with clock reasons recorded, and with c=2 and c=3 to close the E2E-SLO gap.
- FP8 on L4 / L40S (Ada) vs AWQ on T4.
- Quality: FP16 vs AWQ on an eval set before choosing either for production.
