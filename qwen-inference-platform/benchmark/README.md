# Benchmark harness

The harness sends closed-loop traffic at an OpenAI-compatible `/v1/chat/completions` endpoint and writes JSON, CSV, and a text summary.

It is ready to point at vLLM. On this Mac, vLLM is not running, so a vLLM URL produces a `not_run` record. An optional Ollama config exercises the same client against `qwen2.5:14b`.

## Metrics

Client clock, from the moment the request is sent:

```text
TTFT = time of first non-empty content chunk - request start
E2E  = time the stream finishes - request start
TPOT = (E2E - TTFT) / (output_tokens - 1)    # only when output_tokens > 1
output tokens/second = output_tokens / E2E
decode tokens/second = 1 / TPOT
ITL sample           = gap between content-bearing SSE chunks
```

TPOT matches the per-request formula in vLLM's `benchmark_serving.py`. The first token sits inside TTFT. Later tokens are decode.

Output token counts come from `usage.completion_tokens`. Streaming requests send `stream_options.include_usage`. One SSE chunk can contain more than one token, so chunk gaps are stored as inter-chunk latency and are not relabeled as token counts. vLLM's own benchmark discussion makes the same distinction: TPOT is per request, ITL is per streamed event.

Aggregate throughput is different from the average of per-request tokens/second:

```text
aggregate output tokens/second = sum of output tokens / measured wall-clock seconds
```

The wall clock starts after warmup and ends when the last measured request finishes. Overlapping requests share that window, which is what concurrency is for.

A non-streaming response has an E2E time and no TTFT. The inference client can send one with `--no-stream` to show that difference. The harness always streams.

These client times include the network and any queue. vLLM's own histograms (`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds`) are server-side and belong to the Prometheus phase. They are not collected here.

## Methodology we will reuse

The file is `configs/methodology.yaml`.

1. Start with concurrency 1, one fixed user prompt (target 256 estimated tokens), and a fixed `max_tokens` of 128. Temperature is 0.
2. Send one warmup request and drop it from the summary, so the first measured call is not paying for weight load or allocator warmup.
3. Record the measured requests.
4. Repeat at concurrency 2, then 4, then 8. The command runs one level. Higher levels stay listed in the file and are called out in the notes when they were not executed.
5. After 8 looks healthy on a GPU, continue the same way: 16, and so on.
6. Stop climbing when p95 TTFT, p95 TPOT, or p95 E2E crosses the SLO you write into the `slo` block. That block is null until a GPU run exists.
7. Read aggregate tokens/second and request throughput at the last level that stayed inside the SLO. That level is the candidate operating point for one replica. Capacity planning turns it into a replica count. This phase does not.

Each level should keep the prompt and the output cap fixed. Changing both at once mixes prefill cost with decode cost.

## How a run is shaped

Closed loop: a semaphore lets at most `concurrency` requests sit in flight, and the runner launches `num_requests` measured calls. This is the right shape for "how many simultaneous users does one replica hold." It is not an open-loop arrival rate. Arrival rate comes later, with the gateway.

There are no automatic retries. A retry would hide an error and add its latency to the sample.

If `/v1/models` cannot be reached, the runner writes `status: not_run`, leaves the request list empty, and does not invent latencies.

## Commands

vLLM methodology, expected `not_run` on this Mac:

```bash
python -m qwen_platform bench \
  --config benchmark/configs/methodology.yaml \
  --output benchmark/results/runs/vllm-baseline.not-run.json
```

One concurrency level, once a GPU server exists:

```bash
python -m qwen_platform bench \
  --config benchmark/configs/methodology.yaml \
  --concurrency 1
```

Then repeat with `--concurrency 2`, `4`, and `8`.

Ollama comparison on this Mac:

```bash
python -m qwen_platform bench \
  --config benchmark/configs/ollama-smoke.yaml
```

## Result files

See `results/README.md`. Percentiles use linear interpolation, the same default NumPy uses.

### Format specimen

The numbers below are a shape example so the fields are easy to read. They are not a measurement.

```text
Benchmark summary
status: completed
verification: REQUIRES_NVIDIA_GPU
vllm_gpu_baseline: true
engine: vllm
model: Qwen/Qwen2.5-14B-Instruct
endpoint: http://127.0.0.1:8000/v1
timestamp: 2026-10-05T00:00:00+00:00
concurrency: 1
measured_requests: 4
success_count: 4
failure_count: 0
wall_seconds: 1.0
TTFT seconds: count=4 mean=0.1 p50=0.1 p95=0.1 min=0.1 max=0.1
TPOT seconds: count=4 mean=0.02 p50=0.02 p95=0.02 min=0.02 max=0.02
E2E seconds: count=4 mean=0.3 p50=0.3 p95=0.3 min=0.3 max=0.3
aggregate output tokens/second: 100.0
request throughput per second: 4.0
mean inter-chunk latency seconds: 0.02
```

A real not-run file has `status: not_run`, `vllm_gpu_baseline: false`, null percentile means, and an empty request list.

CSV columns: `timestamp`, `model`, `endpoint`, `engine`, `concurrency`, `request_index`, `input_tokens`, `output_tokens`, `ttft_seconds`, `tpot_seconds`, `e2e_seconds`, `output_tokens_per_second`, `success`, `error_type`, `error_message`.

## Prompt length

`target_input_tokens` grows a fixed sentence until a 4-character estimate reaches the target. That estimate is not the Qwen tokenizer. The recorded `input_tokens` field is `usage.prompt_tokens` when the server sends it, and that count includes the chat template. Profiles `short`, `medium`, and `long` are fixed repeats of the same sentence for the single-request client.

## What was measured on this Mac

2026-10-05, Ollama `qwen2.5:14b`, quantization `Q4_K_M`, model already loaded after the first call. These numbers are **LOCAL VERIFIED** and `vllm_gpu_baseline: false`.

The first short streaming call, while the model was coming into memory, took 7.01 s to the first token and 7.61 s end to end for 16 output tokens. After that, short prompts (56 API prompt tokens, including the chat template) at concurrency 1 showed TTFT around 0.04–0.09 s. Treat the 7 s figure as startup, not as prefill cost.

Warm calls with `max_tokens` 8, same machine:

| Profile | API prompt tokens | TTFT | TPOT | E2E | Decode tokens/s |
| --- | --- | --- | --- | --- | --- |
| medium | 550 | 2.24 s | 0.041 s | 2.53 s | 24.4 |
| long | 2110 | 7.07 s | 0.042 s | 7.37 s | 23.7 |

TTFT grew with the prompt. TPOT stayed near 42 ms. That split is the reason the two clocks exist. It was observed on Apple Silicon GGUF, so it is not the vLLM GPU curve.

On these Ollama streams, content-chunk count matched `completion_tokens`. That alignment is an observation from these calls. The harness still stores chunk gaps separately from TPOT.

A non-streaming short call returned E2E only. TTFT and TPOT were null, which is what a finished JSON body can tell you.

The concurrency-1 smoke of 2 short requests is in `benchmark/results/runs/ollama-smoke.json` (gitignored). Both requests succeeded. Aggregate output throughput was 23.7 tokens/s over a 1.35 s wall clock.

The vLLM methodology run is `benchmark/results/runs/vllm-baseline.not-run.json`. Status is `not_run`. Port 8000 refused the connection. No latency was filled in.

## Local check versus NVIDIA

**LOCAL VERIFIED:** formula tests, a mock OpenAI server, the Ollama calls above.

**REQUIRES NVIDIA GPU:** every number you would use to pick a concurrency limit for a vLLM replica.
