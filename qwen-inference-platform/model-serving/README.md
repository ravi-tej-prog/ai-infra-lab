# Model serving baseline

Phase 1 prepares one vLLM process for `Qwen/Qwen2.5-14B-Instruct`. The process is not started on this Mac.

## Which checkpoint

The serving model is the instruction-tuned Hugging Face checkpoint:

| Field | Value | Source |
| --- | --- | --- |
| Model id | `Qwen/Qwen2.5-14B-Instruct` | Hugging Face model card |
| Base model | `Qwen/Qwen2.5-14B` | Model card `base_model` |
| Parameters | 14,770,033,664 | Safetensors header, recomputed in `qwen_platform.model_math` |
| Non-embedding parameters | about 13.2 billion | 48 layers plus the final norm |
| Type | Causal language model | Model card |
| Architecture | `Qwen2ForCausalLM` | `config.json` |
| Layers | 48 | `config.json` |
| Hidden size | 5120 | `config.json` |
| Attention | Grouped-query, 40 query heads, 8 KV heads, head dim 128 | `config.json` |
| MLP | SwiGLU (`silu`), intermediate size 13824 | `config.json` |
| Position encoding | RoPE, `rope_theta` 1,000,000 | `config.json` |
| Normalization | RMSNorm | Model card |
| Extra attention detail | QKV bias | Model card and the parameter recount |
| Embeddings | Untied input and output embeddings, vocab 152064 | `config.json` `tie_word_embeddings: false` |
| Native context | 32,768 tokens | `max_position_embeddings` |
| Extended context | 131,072 tokens with YaRN | Model card |
| Recommended generation length | 8,192 tokens | Model card |
| Weight dtype | bfloat16 | `config.json` `torch_dtype` |

Print the arithmetic with:

```bash
python -m qwen_platform model-math
```

The recount matches the published parameter count exactly: each layer has Q, K, V, and output projections, QKV biases, three SwiGLU matrices, and two RMSNorms. Two embedding matrices are added because the word embeddings are not tied.

### Why this variant

`Qwen/Qwen2.5-14B` is the base model. It has the same shapes and is the wrong default for a chat API, because it was not instruction-tuned.

`Qwen/Qwen2.5-14B-Instruct-1M` is a different long-context checkpoint. Its own card recommends a custom vLLM build and multi-GPU settings. That is outside this phase.

`Qwen/Qwen2.5-14B-Instruct-AWQ` and `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4` are official 4-bit checkpoints of the same instruct model. They are the smaller-GPU alternative, not the quality baseline.

The local Ollama tag `qwen2.5:14b` is a 9.0 GB GGUF. `ollama show` on this machine reports architecture `qwen2`, 14.8B parameters (Ollama's rounding of the same model), embedding length 5120, context length 32768, and quantization `Q4_K_M`. It is a useful client target on this Mac. It is a different file and a different runtime from the BF16 checkpoint vLLM will load.

## Weight memory

BF16 stores two bytes per parameter.

- Weight bytes: 14,770,033,664 × 2 = 29,540,067,328 bytes = **27.511 GiB**
- Hugging Face `usedStorage` for the BF16 repo, fetched 2026-10-05: 29,540,134,000 bytes. The extra bytes are safetensors framing.

AWQ repo storage: 9,980,170,584 bytes (9.295 GiB). GPTQ-Int4 repo storage: 9,988,759,504 bytes (9.303 GiB). Both headers still report 14,770,033,664 logical parameters. The AWQ and GPTQ configs say `quant_method` `awq` and `gptq`, 4 bits. A large FP16 slice remains, which is why the files are near 9.3 GiB rather than a pure 4-bit packing of every parameter.

These are file sizes. A running process also needs activations, CUDA graphs, and the KV pool.

## KV cache

For BF16 or FP16:

```text
bytes per token = 2 (K and V) × layers × kv_heads × head_dim × 2 bytes
                = 2 × 48 × 8 × 128 × 2
                = 196,608 bytes
                = 192 KiB
```

Grouped-query attention is why the 8 KV heads matter. Forty query heads would make this cache five times larger.

One sequence that fills the whole context costs:

| Context | BF16 KV for one full sequence |
| --- | --- |
| 8,192 tokens (this baseline) | 1.5 GiB |
| 32,768 tokens (native `max_position_embeddings`) | 6.0 GiB |
| 131,072 tokens (YaRN) | 24.0 GiB |

vLLM does not allocate that amount per request up front. It takes the memory left after weights and builds a shared block pool. The table is the cost of one sequence that actually occupies that many tokens. Several long requests add up inside the same pool.

YaRN is left off. The Qwen card says vLLM's YaRN factor is static, so enabling it changes behavior for short prompts too. The baseline is a short-prompt measurement. The engine-args page checked for vLLM 0.31.0 did not list a `--rope-scaling` flag, so the renderer does not emit one. A later long-context experiment can pass the scaling through a flag that `vllm serve --help` shows on the GPU host.

## One GPU, or tensor parallel

`tensor_parallel_size` is 1 in the baseline profile.

BF16 weights are 27.511 GiB before KV and activations. Read this as planning arithmetic, not a benchmark:

| GPU memory | BF16 single GPU | Practical path |
| --- | --- | --- |
| 24 GB | Weights do not fit | AWQ or GPTQ-Int4 profile. Files are about 9.3 GiB |
| 40 GB | Weights fit, leftover room is small once KV and CUDA graphs are added | Treat as marginal. Prefer AWQ, or two GPUs, and confirm on the machine |
| 48 GB | Weights fit with room for the 8,192-token baseline and modest concurrency | Reasonable single-GPU BF16 candidate |
| 80 GB | Weights fit with substantial KV room | Baseline target. Tensor parallel is not required to load the model |

Tensor parallel becomes necessary when one GPU cannot hold the weights plus the KV pool you need, or later when one GPU's decode rate misses the latency target. It splits each matrix and adds an all-reduce on every layer, so it is extra communication as well as extra memory.

Head counts limit the easy split sizes. Both 40 query heads and 8 KV heads divide evenly by 1, 2, 4, and 8. Confirm any other size with the vLLM error on the GPU host. This phase does not run tensor parallel.

## vLLM version and flags

PyPI `vllm` version on 2026-10-05 is **0.31.0**, uploaded that morning. `requires_python` is `>=3.10,<3.15`. The published wheels are:

- `vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl`
- `vllm-0.31.0-cp38-abi3-manylinux_2_28_aarch64.whl`
- an sdist

There is no macOS wheel on that release. vLLM is not installed in this environment. The flag list was taken from `https://docs.vllm.ai/en/latest/configuration/engine_args/` on the same day, which is the documentation URL advertised by the PyPI package. `--host` and `--port` are documented by the Qwen vLLM deployment guide as `vllm serve` frontend arguments. They do not appear on the engine-args page.

The renderer refuses unknown keys, so a typo cannot silently disappear. Null values are omitted. That is deliberate for `max_num_seqs` and `max_num_batched_tokens`: the engine-args page says the CLI default is a testing placeholder and the real default is chosen in `EngineArgs.create_engine_config`. Inventing 256 would pretend we measured a scheduler cap.

`gpu_memory_utilization` is set to 0.90. The same docs page says the built-in default is 0.92. Older Qwen FAQ text said 0.9. The profile pins 0.90 so the command does not change underneath us if the default moves again.

`generation_config.json` on the checkpoint uses temperature 0.7, top-p 0.8, top-k 20, and repetition penalty 1.05. `--generation-config vllm` skips that file, and the override sets temperature 0. The benchmark client also sends temperature 0. Greedy decoding keeps output length and latency comparable across runs.

## What the command does internally

`vllm serve` loads the checkpoint, profiles GPU memory, reserves a KV block pool, and opens an HTTP server. That server speaks the OpenAI chat completions API. Continuous batching puts several requests into one model step. Prefill processes the prompt. Decode then generates one new token at a time for each active sequence, reading the KV cache. PagedAttention stores that cache in fixed-size blocks so sequences can grow without reserving their maximum length as one contiguous slab.

On a single GPU the tensor-parallel size is 1, so there is no all-reduce between GPUs.

## What changes in production

The listen address moves behind the future gateway. The model files come from a local cache or an internal mirror, with the revision pinned. `max_model_len`, `max_num_seqs`, and `gpu_memory_utilization` change after a real GPU profile, not before. Quantization and tensor parallel stay at these defaults until a GPU run shows the BF16 replica misses the memory budget or the latency target.

## Local check versus NVIDIA

**LOCAL VERIFIED:** the YAML loads, unknown flags fail, and `render-serve` prints a command. This Mac has 48 GiB of unified memory. `ollama show qwen2.5:14b` reports `Q4_K_M`, and that 9.0 GB model answered chat requests. See `benchmark/README.md` for the timings.

**REQUIRES NVIDIA GPU:** installing the vLLM wheel, downloading the BF16 weights, allocating the KV pool, and answering a request. CUDA graphs, PagedAttention residency, and GPU utilization are part of that run. Unified memory does not remove that requirement. vLLM 0.31.0 has no macOS CUDA build, and this machine has no NVIDIA device. The 27.511 GiB BF16 weight file would also leave little room next to the operating system inside 48 GiB, which is a second reason the AWQ profile exists for smaller GPUs. It is not a reason to pretend the BF16 file ran here.
