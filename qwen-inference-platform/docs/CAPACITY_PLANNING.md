# Capacity planning

This document is the procedure for a later GPU run. It contains no measured concurrency, no replica count, and no cost.

The question it answers, once data exists:

> How many simultaneous requests can one vLLM replica hold before the service level objective breaks, and how many replicas does that imply?

## Signals

Per replica, from the benchmark harness in this repo:

- p95 TTFT
- p95 TPOT
- p95 end-to-end latency
- aggregate output tokens per second
- successful requests per second
- failure count and error text

From the vLLM process, in the Prometheus phase, not collected now. Names below are from vLLM's metrics design notes and must be re-checked against the `/metrics` page of the 0.31 server, because series names move between versions:

- `vllm:time_to_first_token_seconds`
- `vllm:inter_token_latency_seconds`
- `vllm:e2e_request_latency_seconds`
- `vllm:request_queue_time_seconds`
- `vllm:request_prefill_time_seconds`
- `vllm:request_decode_time_seconds`
- `vllm:kv_cache_usage_perc`
- `vllm:num_requests_running`, plus the waiting-request series exposed beside it

GPU utilization is a device metric from the NVIDIA driver or DCGM. The benchmark client cannot see it. A replica can show a full KV cache while the client only sees rising TTFT, so both views are required before a capacity number is trusted.

Client TTFT includes the network. Server TTFT starts when the API process accepts the request. Use the server histogram for the SLO once Prometheus exists. Use this harness to get a first curve and to debug the client.

## Why the curve bends

Prefill reads the whole prompt and writes the KV cache. TTFT grows with prompt length and with how many prefills share the GPU.

Decode reads that cache and writes one new token per step. TPOT is the average time of those later steps.

Concurrency lets several sequences share a model step, so throughput can rise. It does not rise forever. The GPU's math units and memory bandwidth saturate, the KV pool fills, and extra requests wait. Queue time is then added to TTFT and to end-to-end latency. Past that point, more concurrency buys waiting, not tokens.

KV bytes for this model are 192 KiB per token at BF16. A few long prompts can fill the pool while a larger number of short prompts still fit. The operating point is therefore a property of the workload, not a property of the model name alone.

## Decision process

1. **Define the workload.** Fix the model, the prompt-length mix, and the output length. The baseline in `benchmark/configs/methodology.yaml` is one 256-token user target and 128 output tokens. A production mix may be several of those buckets. Keep them as separate curves so a long-prompt failure is visible.
2. **Define the SLO.** Write p95 TTFT, p95 TPOT, and p95 end-to-end limits into the `slo` block before looking at the curve. Add an error-rate limit. Leave GPU and KV ceilings explicit too, for example a KV usage level you refuse to cross. Those ceilings are blank until you choose them.
3. **Benchmark one replica.** Serve the baseline profile with `tensor_parallel_size: 1` on one NVIDIA GPU. Warm the process, then measure concurrency 1.
4. **Increase concurrency.** Repeat at 2, 4, 8, and further steps of the same shape. Keep the prompt and the output cap fixed.
5. **Find the SLO-safe operating point.** The highest concurrency whose p95 TTFT, p95 TPOT, p95 end-to-end latency, and error rate are all inside the SLO, and whose KV usage and GPU memory still have the headroom from step 2.
6. **Read requests per second at that point.** Use the harness `request_throughput_per_second` from the safe level. That is the sustainable rate of one replica for this workload.
7. **Add headroom.** Divide that rate by a headroom factor you choose for bursts, deploys, and a replica dying. The factor is a product decision. This document does not pick one.
8. **Calculate replicas.** Required replicas are the offered request rate divided by the headroom-adjusted rate from step 7, rounded up. Offered rate is an input from the product, not a number this repo can know yet.
9. **Decide tensor parallel.** Stay at TP 1 while one GPU meets the memory budget and the SLO. Move to TP 2, 4, or 8 when the weights plus the KV pool do not fit, or when one GPU cannot reach the latency target at the concurrency you need. Re-run steps 3 through 8 after any TP change, because one TP group is one replica and it occupies that many GPUs.
10. **Validate cost and performance.** Compare the safe point on the BF16 80 GB-class path with the AWQ profile on a smaller GPU. Cost is GPU-hours per 1,000 requests at the safe operating point. Choose the cheaper path that still meets the SLO. Do that with measured runs.

## What this phase deliberately leaves blank

- maximum sustainable concurrency
- requests per second per replica
- replica count
- headroom factor
- dollar cost
- a filled-in SLO

Filling any of those in from the Ollama model, or from the memory arithmetic in `model-serving/README.md`, would mix a laptop GGUF runtime with a GPU serving plan.
