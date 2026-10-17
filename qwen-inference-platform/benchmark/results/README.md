# Benchmark results

`runs/` holds files produced on a machine. Those files are gitignored.

A run has three siblings:

- `*.json` — full record
- `*.csv` — one row per request
- `*.summary.txt` — the human-readable summary

`status` is one of:

- `completed` — every measured request returned tokens
- `completed_with_failures` — some requests failed
- `failed` — the server answered `/models`, then every measured request failed
- `not_run` — the endpoint was unreachable, so no latency sample was taken

`vllm_gpu_baseline` stays `false` until someone passes `--mark-nvidia-baseline` while pointing at a Linux NVIDIA vLLM server. An Ollama run is `LOCAL_VERIFIED` and is still not that baseline.

The vLLM methodology in `benchmark/configs/methodology.yaml` has no measured GPU numbers in this phase.
