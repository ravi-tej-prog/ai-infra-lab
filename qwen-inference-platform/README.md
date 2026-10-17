# Qwen inference platform

Phase 0 and phase 1 of a production-style inference platform: repository layout, a vLLM serving profile for Qwen2.5 14B Instruct, a streaming client, and a benchmark harness.

The client runs here. vLLM on CUDA does not. This machine is an Apple Silicon Mac, and the current vLLM release does not ship a macOS CUDA wheel.

## What this phase includes

```text
Qwen2.5-14B-Instruct
        |
        v
vLLM serve profile          rendered, not started, on this Mac
        |
        v
OpenAI-compatible HTTP API  the API vLLM will expose
        |
        v
benchmark client            runs locally, against a mock server or Ollama
```

## What waits

FastAPI, React, Kubernetes, Docker packaging, Prometheus, Grafana, HPA, KEDA, GPU autoscaling, node affinity, taints, tensor-parallel runs, and multi-node inference. See `ARCHITECTURE.md`.

## Layout

```text
qwen-inference-platform/
  README.md
  ARCHITECTURE.md
  qwen_platform/          Python package: client, metrics, harness, renderer
  model-serving/          checkpoint notes and vLLM YAML profiles
  benchmark/              methodology and result files
  scripts/                wrappers around the Python commands
  docs/CAPACITY_PLANNING.md
  tests/
```

## Setup

Requires Python 3.11 or newer. This Mac is on Python 3.14, which is inside vLLM 0.31.0's declared range (`>=3.10,<3.15`) and is still unable to run the CUDA wheel.

```bash
cd qwen-inference-platform
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Direct dependencies, and why they are here:

- `httpx==0.28.1` reads the streaming response so the client can timestamp the first content chunk itself.
- `PyYAML==6.0.3` loads the serving profile and the benchmark methodology.
- `pytest==9.1.1` is a development dependency for the formula, config, and mock-server tests.

vLLM is not a dependency of this package. Installing it here would not create a CUDA device.

## Commands

```bash
python -m qwen_platform preflight
python -m qwen_platform model-math
python -m qwen_platform render-serve
pytest
```

Single request against a server. vLLM's default URL will fail on this Mac until a GPU host exists:

```bash
python -m qwen_platform infer --profile short --max-tokens 32
python -m qwen_platform infer --profile medium --max-tokens 64
python -m qwen_platform infer --profile long --max-tokens 128
python -m qwen_platform infer --profile short --max-tokens 16 --no-stream
```

The same client pointed at local Ollama. On this machine `ollama show qwen2.5:14b` reports `qwen2`, `Q4_K_M`, and a 32,768-token context. That call is a GGUF comparison, not the vLLM baseline:

```bash
python -m qwen_platform infer \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5:14b \
  --engine ollama \
  --profile short \
  --max-tokens 32
```

Benchmark. The methodology command writes `not_run` when port 8000 has no vLLM:

```bash
python -m qwen_platform bench \
  --config benchmark/configs/methodology.yaml \
  --output benchmark/results/runs/vllm-baseline.not-run.json
```

Ollama smoke, two short requests, concurrency 1:

```bash
python -m qwen_platform bench --config benchmark/configs/ollama-smoke.yaml
```

Shell wrappers with the same arguments live in `scripts/`.

Sweep, every input length at every concurrency level, one server setting per label:

```bash
python -m qwen_platform sweep --label 14b-p1
python -m qwen_platform sweep --label 3b-p4 --model qwen2.5:3b
python -m qwen_platform sweep-report benchmark/results/sweeps/*
```

Each request starts with a random tag (`unique_prompts: true`), so prefix caching cannot skip prefill and hide input-length cost.

## Metrics

Ollama has no `/metrics`, so the sweep exports client-observed TTFT, TPOT, E2E, tokens, and in-flight requests as Prometheus histograms:

```bash
docker compose -f observability/docker-compose.yml up -d
python -m qwen_platform sweep --label 14b-p4 --metrics-port 9101
```

Prometheus is at http://localhost:9090 and Grafana at http://localhost:3000 (dashboard "LLM benchmark — client view"). On a GPU node the same stack adds two scrape targets in `observability/prometheus/prometheus.yml`: vLLM's own `/metrics` and the DCGM exporter. Dashboard percentiles are bucket estimates. `sweep.md` holds the exact values.


## Concepts

**Prefill.** The model reads the prompt once and writes the KV cache for every prompt token. Cost grows with prompt length.

**Decode.** After the first output token, the model produces one new token at a time and reads the cache it just wrote. Cost grows with output length and with how many sequences share the step.

**TTFT, time to first token.** Clock time from sending the request until the first output token arrives. Prefill dominates it. Queueing and network sit in the same interval on a client measurement.

**TPOT, time per output token.** `(E2E - TTFT) / (output_tokens - 1)`. This is the average decode interval. It is undefined for a single output token, because there is no second token to measure against. It is not the total latency.

**ITL, inter-token latency.** The gap between successive output tokens. Mean ITL and mean TPOT describe the same decode stretch in different averages: ITL weights every gap, TPOT is computed per request and then averaged across requests. This client stores gaps between SSE chunks. Those gaps are token gaps only when the server puts one token in each chunk.

**E2E latency.** Clock time from sending the request until the stream finishes. It contains TTFT plus every later token. A long prompt can make E2E large even when decode itself is quick.

**Throughput.** Tokens or requests per second over a shared wall clock. Per-request tokens/second uses that request's own E2E, so prefill is in the denominator. Aggregate tokens/second uses the whole measured window, so overlapping requests count together.

**Concurrency.** How many requests are allowed in flight together. The harness is closed-loop: it keeps up to N requests outstanding until the planned count is finished.

TTFT moves with prefill because the first token cannot be sent until the prompt has been processed and the first decode step has run. A longer prompt is more matrix work before that token exists. Other requests prefilling at the same time share the same GPU, so TTFT moves with contention too.

TPOT moves with decode because the first-token wait has already been removed. What remains is the time to produce the rest of the tokens. Decode is usually memory-bandwidth heavy: each step reads the weights and the growing KV cache.

Raising concurrency does not raise throughput in a straight line. A quiet GPU has unused math units and unused memory bandwidth, so a second request can often share the step and add tokens. Once those units are busy, extra requests do not create extra silicon. They wait. The KV pool is finite as well. At 192 KiB of BF16 KV per token on this model, long sequences fill it. The scheduler then queues. Queue time is added on top of prefill, which is why latency climbs after throughput flattens.

## Decisions

### Serve `Qwen/Qwen2.5-14B-Instruct` in BF16

1. **What.** The baseline checkpoint is the official 14.7B instruct model, dtype chosen by vLLM `auto` from `torch_dtype: bfloat16`.
2. **Why.** The platform is a chat API. The instruct model is the checkpoint trained to follow that API. BF16 is the unmodified weight file, so the first GPU curve is about the serving stack rather than about a quantizer.
3. **Problem it solves.** It fixes the model id, the context cap, and the memory budget before anyone writes a gateway. The base model, the 1M-context variant, the AWQ file, and the Ollama GGUF are different artifacts.
4. **What happens internally.** vLLM reads `config.json`, loads 14,770,033,664 BF16 parameters (27.511 GiB of weights), and sizes a KV pool from the memory that remains. Each token stores K and V for 48 layers and 8 KV heads.
5. **Production change.** Pin the Hugging Face revision. Move files to an internal mirror. Switch to the AWQ profile, or to tensor parallel, only after a GPU run shows BF16 on one GPU misses the budget or the SLO.
6. **On this Mac.** The parameter recount, the file sizes, and the rendered command are checked. `python -m qwen_platform model-math` prints them.
7. **On NVIDIA Linux.** Downloading the weights, allocating the pool, and generating tokens.

### Keep vLLM off this Mac

1. **What.** The package renders `vllm serve` and refuses to treat this computer as a CUDA host.
2. **Why.** PyPI `vllm==0.31.0` (uploaded 2026-10-05) publishes Linux `manylinux` wheels and a source tarball. It does not publish a macOS wheel. Apple Silicon has no NVIDIA GPU and no CUDA driver.
3. **Problem it solves.** It prevents a green test from being mistaken for a GPU baseline. A community Metal plugin exists (`vllm-metal`). It runs MLX kernels, wants Python 3.12, and serves MLX checkpoints. That would be a third runtime next to Ollama and CUDA vLLM, so it is not installed in this phase.
4. **What happens internally.** `preflight` reports the OS, the machine, and whether `import vllm` succeeds. On this machine the import does not succeed. `render-serve` still prints the Linux command.
5. **Production change.** The same YAML is copied to a Linux NVIDIA host. That host installs vLLM 0.31.0, runs the command, and the Mac can stay the benchmark client if you point `--base-url` at the GPU machine.
6. **On this Mac.** Platform check, command rendering, mock-server tests, optional Ollama calls.
7. **On NVIDIA Linux.** The actual `vllm serve` process.

### Put serving settings in YAML

1. **What.** `model-serving/configs/*.yaml` holds the model id and the engine flags. A small renderer turns a profile into an argument list.
2. **Why.** A shell one-liner hides which flags are real and which were typed from memory. The renderer only emits flags checked against the vLLM 0.31 engine-args page, plus `--host` and `--port` from the Qwen deployment guide.
3. **Problem it solves.** The next phase can change context length or memory fraction in one file and review the diff. Unknown keys fail the load instead of being dropped.
4. **What happens internally.** Null fields are omitted. That matters for `max_num_seqs` and `max_num_batched_tokens`, whose CLI placeholders are not the runtime defaults. `--generation-config vllm` skips the checkpoint's temperature 0.7 sampling so the latency run is greedy.
5. **Production change.** The same profile gains a pinned revision, a local model path, and whatever scheduler caps the GPU curve justifies. Tensor parallel stays 1 until that curve says otherwise.
6. **On this Mac.** `pytest` loads both profiles and checks the argument list. `render-serve` prints the command.
7. **On NVIDIA Linux.** The process starts, profiles GPU memory, and serves.

### Time the stream in our own client

1. **What.** A small `httpx` client posts to `/v1/chat/completions` and records request start, first content chunk, and stream end.
2. **Why.** TTFT is invisible in a non-streaming body. An SDK that returns the finished message also hides the first-token instant.
3. **Problem it solves.** The harness can tell prefill-dominated wait from decode-dominated wait, which is the split capacity planning needs.
4. **What happens internally.** The client parses server-sent events. A role chunk is ignored. The first non-empty `delta.content` sets TTFT. `usage.completion_tokens` sets the denominator for TPOT. Chunk gaps are stored beside that, because a chunk is not defined to be one token.
5. **Production change.** The gateway will add its own queue. Client TTFT will then include gateway queue plus vLLM queue plus prefill. The SLO should move to vLLM's server histogram once Prometheus is scraping it. This client remains the external probe.
6. **On this Mac.** Unit tests drive a local mock server with an artificial pause before the first token. Ollama can run the same function against a real 14B GGUF.
7. **On NVIDIA Linux.** Point the same function at port 8000. Pass `--mark-nvidia-baseline` only after you know that process is vLLM on NVIDIA GPUs. Until then `vllm_gpu_baseline` stays false.

### Record a not-run instead of a guessed curve

1. **What.** The harness supports concurrency, request count, prompt length, output cap, and endpoint. The methodology starts at concurrency 1 and lists 2, 4, and 8.
2. **Why.** The SLO knee is an empirical point. Writing it down early would freeze a fiction into the capacity doc.
3. **Problem it solves.** Later GPU runs append real JSON next to a stable schema. The Mac can prove the schema with a mock server.
4. **What happens internally.** Warmup requests are discarded. A semaphore caps in-flight calls. Wall-clock throughput uses only the measured window. If `/v1/models` is down, the file says `not_run` and the latency fields stay null.
5. **Production change.** Run the same file on the GPU host, one concurrency at a time, then follow `docs/CAPACITY_PLANNING.md`. Retries and open-loop arrivals belong in the gateway, not in this sample.
6. **On this Mac.** Mock-server tests, the `not_run` record against port 8000, and the Ollama smoke config.
7. **On NVIDIA Linux.** The concurrency sweep that produces the first real operating point.

## Verification labels

**LOCAL VERIFIED** means the check ran on this Mac: tests, the rendered command, the memory arithmetic, and any Ollama call you choose to make.

**REQUIRES NVIDIA GPU** means loading BF16 or AWQ weights in vLLM, observing the KV pool, and trusting a latency curve. The harness will talk to that server from the Mac when you have one. It cannot create it here.

Details: `model-serving/README.md`, `benchmark/README.md`, `docs/CAPACITY_PLANNING.md`.
