# Architecture

This repository is phase 0 and phase 1 of an inference platform. Later boxes are drawn so the direction stays visible. Their code is not in the tree.

## Phase 1, what exists now

```text
benchmark client  (this repo)
        |
        |  HTTP, OpenAI chat completions, streaming
        v
inference API     (vLLM's own server, started later by `vllm serve`)
        |
        v
vLLM engine       (Linux, NVIDIA GPU)
        |
        v
Qwen2.5-14B-Instruct
```

On this Mac the same client can talk to a different process:

```text
benchmark client
        |
        v
Ollama OpenAI-compatible API   http://127.0.0.1:11434/v1
        |
        v
qwen2.5:14b   (local GGUF, Apple Silicon)
```

Those two servers are not interchangeable measurements. Ollama checks the client. vLLM on NVIDIA is the baseline the rest of the platform will scale.

The serving shape is a YAML profile rendered into a `vllm serve` command. The profile names the model, dtype, context cap, tensor-parallel size, and GPU memory fraction. The client and the harness are the only running Python in this phase.

vLLM's HTTP server is part of vLLM. It is not the FastAPI gateway this project will add later. Building that gateway now would put a second hop in front of a server we have not measured.

## Later phases, not implemented

```text
React
  |
  v
FastAPI gateway
  |
  v
model router
  |
  v
Kubernetes Service
  |
  v
vLLM replicas  ---- one Pod per replica, baseline tensor parallel size 1
  |
  v
GPU nodes
  |
  v
Qwen2.5-14B-Instruct

Prometheus  -->  inference metrics  -->  autoscaling  -->  GPU node provisioning
```

Left out on purpose until the single-replica curve exists:

- FastAPI gateway and React
- Kubernetes, Docker packaging, service discovery
- Prometheus and Grafana
- HPA, KEDA, GPU autoscaling
- node pools, node affinity, taints, tolerations
- tensor-parallel benchmarks, multi-GPU, NCCL, distributed inference
- high availability and failure drills

## Request path inside one replica

1. The client sends `POST /v1/chat/completions` with `stream: true` and a `max_tokens` cap.
2. vLLM applies the chat template and tokenizes the messages. That token count is the prefill length.
3. The scheduler admits the request if the KV pool can hold it. Otherwise the request waits. Wait time becomes part of client TTFT.
4. Prefill runs the prompt through all 48 layers and writes K and V for every prompt token.
5. The first generated token is sent on the stream. The client records TTFT.
6. Decode repeats, one new token per step, until `max_tokens` or an end-of-sequence token. The client records E2E and computes TPOT from the tokens after the first.
7. Continuous batching can fold other requests into those same steps. That is why concurrency can raise throughput, and why a crowded step slows every request in it.

## Replica, later

One replica is one `vllm serve` process plus the GPUs in its tensor-parallel group. The baseline group is one GPU. A Kubernetes Service would spread requests across replicas. The gateway would own authentication, timeouts, and retries. Those retries must stay out of the benchmark harness, or the latency sample stops describing the model server.

Autoscaling belongs after step 6 of `docs/CAPACITY_PLANNING.md`. The scaler needs a measured requests-per-second per replica. Without that number it would scale on a guess.

## Repository map

```text
qwen-inference-platform/
  README.md                 how to run this phase, and why the choices were made
  ARCHITECTURE.md           this file
  qwen_platform/            client, metrics, benchmark, vLLM command renderer
  model-serving/configs/    baseline BF16 profile and alternate AWQ profile
  benchmark/                methodology, harness docs, result files
  scripts/                  thin wrappers around python -m qwen_platform
  docs/CAPACITY_PLANNING.md procedure for the GPU run
  tests/                    formulas, config renderer, mock server
```
