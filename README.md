# keys-Nvidia — Two-Tower (AR) on a Single DGX Spark

Running **NVIDIA `Nemotron-Labs-TwoTower-30B-A3B-Base`** in **context-tower autoregressive
(AR) mode on ONE DGX Spark (GB10 / sm_121a / 128 GB unified memory)** — coherent output,
benchmarked, with the full reproduction recipe.

TwoTower is a block-diffusion two-tower model (frozen AR **context tower** + trainable
**denoiser tower**). Full two-tower diffusion inference needs **two** GPUs (~59 GB BF16 each).
This repo serves just the **context tower in AR mode on a single GB10** — the officially
supported way to run it on one Spark.

## Purpose — what each mode is for

**Single Spark / single GPU (this repo): context-tower AR.** NVIDIA ships three inference
modes; `--mode ar` is the only one that fits on one GPU, and it is *purposeful*, not a
consolation prize:

- **It is the ST-AR baseline.** The context tower is the frozen `Nemotron-3-Nano-30B-A3B`
  backbone, so AR mode is the exact single-tower autoregressive baseline that the diffusion
  mode's quality (98.7% retention) and speedup (2.42×) are measured *against*. Anyone
  reproducing NVIDIA's numbers needs this serve first.
- **It proves the checkpoint + kernels on sm_121a.** Coherent AR output validates the
  weights, the key remap, and the mamba2/attention kernel path on GB10 before any
  two-tower work — which is precisely how we used it.
- **It is a production-grade serve on its own.** 26 tok/s single / 164 tok/s at C8 with
  256K-context passkey retrieval, from one 128 GB Spark.

**The TRUE Two-Tower requires 2 GPUs — on DGX Spark, that means 2 Sparks.** NVIDIA's
reference (`place_towers_on_devices("cuda:0", "cuda:1")`) assumes two ~80 GB cards in one
box: the **context tower** (AR, prefills the prompt and commits blocks, owns the KV/Mamba
cache) on one device and the **denoiser tower** (mask-diffusion, iteratively unmasks each
16-token block against the frozen context cache) on the other. A GB10 is one GPU, so the
diffusion base as NVIDIA envisioned it maps onto **two DGX Sparks — one tower per Spark**,
with the cross-tower cache traffic (Mamba conv/ssm states + 6 attention layers of KV)
carried over the 200G fabric instead of NVLink/PCIe.

That true 2-GPU / 2-Spark autoregressive-diffusion implementation is documented in its own
repo: **[Keys-NVIDIA-Two-Tower-Diffusion--dual-dgx-spark](https://github.com/drowzeys/Keys-NVIDIA-Two-Tower-Diffusion--dual-dgx-spark)**.

## The key insight

**The context tower *is* the frozen `Nemotron-3-Nano-30B-A3B` backbone.** So instead of the
model's HF remote-code path (which hits a `causal_conv1d` numerical bug on sm_121a and emits
degenerate garbage — `" and, and, and…"`), we:

1. **Remap the checkpoint keys** (`context_tower.* → backbone.*`, `context_lm_head.weight →
   lm_head.weight`) and set `architectures: [NemotronHForCausalLM]`, producing a checkpoint
   that **stock vLLM `nemotron_h.py` loads directly**.
2. **Serve it on a DGX-Spark vLLM image** using vLLM's own Triton mamba2/conv kernels — the
   buggy HF `causal_conv1d` code is never on the path.

Result: **coherent, fluent output** where every prior HF-Transformers attempt was garbage.

## Results (single GB10, BF16, `--enforce-eager`, temp 0)

**Coherence — PASS.** Fluent, accurate prose (geography, physics, working Fibonacci code,
correct French). No degeneration.

**Throughput / concurrency** (vLLM streaming, `include_usage`, 128-token outputs):

| Concurrency | Aggregate tok/s | Per-request tok/s | Avg TTFT (s) |
|---|---|---|---|
| 1 | 26.3 | 26.3 | 0.09 |
| 2 | 43.7 | 21.9 | 1.00 |
| 4 | 56.1 | 14.0 | 0.65 |
| 8 | **163.7** | 20.5 | 0.60 |

Single-stream 26 tok/s is eager-mode (no CUDA graphs); the 3B-active MoE decodes fast
relative to its 59 GB, escaping the GB10 dense-bandwidth ceiling. KV pool 1.03M tokens at
8K/0.6 (125× concurrency).

**Context sweep — passkey retrieval at 50% depth** (secret code `739140`):

| Context | Prompt tokens | Passkey | Latency (s) |
|---|---|---|---|
| 6K | 6,042 | ✅ | 3.4 |
| 32K | 32,040 | ✅ | 8.7 |
| 96K | 96,048 | ✅ | 37.0 |
| 128K | 128,038 | ✅ | 58.2 |
| 256K | 256,040 | ✅ | 186.3 |

**Retrieved correctly at every length through 256K — 2× the card's documented 128K max.**
Latency is prefill-dominated in eager mode.

**Eval (fluency/quality — AR base model, judged on coherence):** all pass — capital-of-Japan,
train-distance math (correct LaTeX), primary/secondary colors, French geography, boiling-point
physics. Raw numbers in [`data/`](data/).

## Prerequisites

- **Hardware:** one NVIDIA DGX Spark (GB10, sm_121a, 128 GB unified). Driver ≥ 580.x, CUDA
  13 host runtime, Docker ≥ 25 with `nvidia-container-toolkit`.
- **Model weights:** `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16` (the context-tower
  shards; ~59 GB). Requires HF auth + accepting the NVIDIA Open Model License.
- **A DGX-Spark vLLM image** with sm_121a kernels — this recipe was validated on
  `vllm-dspark-runtime:mia-raf-pr1-nvfp4-b` (vLLM 0.21.1rc1, "mia-raf" DGX-Spark build). Any
  vLLM built for sm_121a that registers `NemotronHForCausalLM` and has Triton mamba2 +
  `TRITON_ATTN` should work. **This image is a prerequisite you supply** (see
  [Base image](#base-image)).

## Install (single DGX Spark)

```bash
# 1. Get the TwoTower context-tower weights (context tower only is enough).
hf download nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16 \
  --local-dir ~/models/tt-context

# 2. Remap the context tower -> stock NemotronHForCausalLM checkpoint.
#    Edit SRC/DST paths at the top of scripts/remap_tt.py, then:
python3 scripts/remap_tt.py
#    -> writes ~/models/tt-context-vllm (header-only rewrite; fast, self-validating)

# 3. Serve on the DGX-Spark vLLM image (Triton MoE + Triton attention).
#    Edit IMG / MODEL_DIR at the top of scripts/twotower-ar-serve.sh, then:
bash scripts/twotower-ar-serve.sh
#    knobs (env): MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.7 MAX_NUM_SEQS=8 EAGER=1

# 4. Smoke test.
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-twotower-30b-bf16-context-ar",
       "prompt":"France is a country","max_tokens":64}' | python3 -m json.tool
```

### The three GB10 walls this recipe clears (and how)

| Wall | Symptom | Fix (baked into the launcher) |
|---|---|---|
| HF `causal_conv1d` sm_121a bug | coherent-load but garbage output | serve as stock vLLM (Triton kernels), not HF remote code |
| Prebuilt FlashAttention-2 | `cudaErrorUnsupportedPtxVersion` | `--attention-backend TRITON_ATTN` (CLI flag; the env var doesn't exist in this build) |
| CUTLASS FP8-MoE JIT | `ninja exit 127` / `cicc` not found | `VLLM_USE_FLASHINFER_MOE_FP16=0` → Triton MoE (no nvcc JIT) |

## Base image

Validated on `vllm-dspark-runtime:mia-raf-pr1-nvfp4-b` (~22.7 GB, vLLM 0.21.1rc1, a DGX-Spark
"mia-raf" build). It is **not redistributed here** (size + upstream licensing). Provide your
own sm_121a vLLM image that:
- registers `NemotronHForCausalLM` (stock vLLM ≥ ~0.21),
- has Triton mamba2 kernels and a working `--attention-backend TRITON_ATTN`,
- honors `VLLM_USE_FLASHINFER_MOE_FP16=0` (Triton MoE fallback).

If you have the mia-raf image locally, the launcher uses it as-is; no build step is required.

## Files

```
scripts/
  remap_tt.py            # context_tower.* -> backbone.* header rewrite (the key step)
  twotower-ar-serve.sh   # gpu-clear + fastkill watchdog + docker run (the launcher)
  fastkill-tt.sh         # OOM watchdog (kills container at <3 GB MemAvailable)
  bench.py  eval.py  coh.py  ctx_sweep.py   # benchmark / eval / coherence / context-sweep clients
data/
  results.json           # all measured numbers (throughput, ctx sweep, eval)
```

## Scope & credits

- This is **context-only single-tower AR** (not the full 2-node two-tower diffusion). The
  forward is proven numerically correct on GB10 via the vLLM path. The **true two-tower
  diffusion mode — context tower on one Spark, denoiser tower on a second Spark, over the
  200G fabric — lives in
  [Keys-NVIDIA-Two-Tower-Diffusion--dual-dgx-spark](https://github.com/drowzeys/Keys-NVIDIA-Two-Tower-Diffusion--dual-dgx-spark)**.
- Model: `nvidia/Nemotron-Labs-TwoTower-30B` (NVIDIA Open Model License).
- Recipe, scripts, and measurements: MIT (see [`LICENSE`](LICENSE)). Validated on a
  DGX-Spark "mia-raf" vLLM image (not included).
