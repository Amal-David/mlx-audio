# Optimized ZONOS2 inference for MLX (Apple Silicon)

This repo is an **optimized inference path for Zyphra's [ZONOS2](https://huggingface.co/Zyphra/ZONOS2)
text-to-speech model on Apple Silicon**, built on [`mlx-audio`](https://github.com/Blaizzy/mlx-audio).

It exists because, as of this writing, ZONOS2 support in mlx-audio lives only in an **experimental,
unmerged branch** by [Lucas Newman](https://github.com/lucasnewman) — not in a release. We use that
work as a **reference** and build a faster, smaller, better-documented version on top of it:

- a **mixed-precision quantized model** (4.68 GB vs 15.34 GB BF16) that the reference ships nothing for,
- the **recipe discovery** that makes 4-bit actually work on this top-1 MoE model,
- **batched throughput tooling** and a reproducible **benchmark suite**,
- a full **performance analysis** on Apple M4 Pro with a concrete optimization roadmap.

> **Attribution.** mlx-audio is © Prince Canuma (MIT). The ZONOS2 MLX port we build on is by Lucas
> Newman. The model is by Zyphra. This repo keeps the MIT license and credits all three. Our additions
> are the quantization recipe, tooling, benchmarks, and analysis below.

---

## What we add over the reference

### 1. A working mixed-precision 4-bit model
Uniform 4-bit **breaks** ZONOS2: it's a top-1 mixture-of-experts model, so quantization noise on the
single active expert doesn't average out. We bisected the failure to **4-bit attention** (the model
emits end-of-audio immediately → empty clips). The recipe that stays healthy across seeds:

| Component | Precision | Why |
|---|---|---|
| MoE experts (`SwitchGLU`) — 94.5% of weights | **4-bit** gs64 | The bandwidth lever; tolerates 4-bit |
| Attention `wq`/`wo` | **8-bit** gs64 | **4-bit attention → spurious early EOS** |
| Token embeddings | **8-bit** gs64 | Phonetic stability |
| Output head, MoE router, gater, temp, RMSNorm, `ChunkedLinear` | **bf16** | Quality-critical / tiny |

→ Published model: **[`amal-david/Zyphra-ZONOS2-4bit`](https://huggingface.co/amal-david/Zyphra-ZONOS2-4bit)**.

### 2. Performance (measured, Apple M4 Pro, 20-core GPU, 273 GB/s, MLX 0.31)

| Metric | BF16 (reference) | **This 4-bit** |
|---|---|---|
| Weights | 15.34 GB | **4.68 GB** |
| Single-stream RTF | ~1.40 | **~0.85** (faster than real-time) |
| Forward-only compute | 1.0× | **~1.6×** |
| Batched throughput (B=32, real EOS workload) | — | **~3.5× real-time**, ~300 frames/s |
| Peak memory @ B=32 | ~19 GB | **~12 GB** |

### 3. Findings that matter if you optimize this further
- The decode bottleneck is split between **weight bandwidth** (forward) and **lost pipelining** from a
  per-frame `.tolist()` GPU→CPU sync in the sampler — not raw compute.
- **Top-1 MoE is sparse at batch 1** (~1/16 experts read), so batching is **sublinear** and saturates
  around **B=32–64** (peak ~4.7× aggregate; it *regresses* past that as expert diversity forces a
  near-dense read into the 273 GB/s wall).
- Quantization raises the batch ceiling by cutting per-expert bandwidth.

---

## Install

```bash
pip install git+https://github.com/Amal-David/mlx-audio.git@zonos2-optimized
```

## Usage

```python
from mlx_audio.tts import load
from mlx_audio.audio_io import write as audio_write

model = load("amal-david/Zyphra-ZONOS2-4bit", lazy=False)
r = next(model.generate(text="Hello from optimized ZONOS two on Apple Silicon.", max_tokens=1024))
audio_write("out.wav", r.audio, r.sample_rate)
```

Runnable scripts in [`zonos2_tools/`](./zonos2_tools): `generate.py` (single / clone / long-form),
`batch_generate.py` (throughput runner), `quantize.py` (reproduce the 4-bit build from BF16), and
`benchmark.py` (bf16 vs 8-bit vs 4-bit RTF + batch sweep).

---

## Implemented decode-loop optimizations
- **On-device batch sampling (single sync/step).** The reference samples each sequence in a
  Python loop, each doing its own `.tolist()` GPU→CPU sync — **B syncs per step**, which serialized
  the GPU at large batch (throughput plateaued ~B=64). We sample every sequence on-device and sync
  **once** per step (`sample_frame_device` → stack → one `.tolist()`). Math and RNG keys are
  unchanged, so output is **byte-identical** (verified MAE = 0). Measured on M4 Pro (4-bit, ignore-eos):

  | Batch | before | after | gain |
  |---|---|---|---|
  | 16 | 2.68× | **3.36×** | +25% |
  | 32 | 3.37× | **4.04×** | +20% |
  | 64 | 3.79× | **4.27×** | +13% |
  | 96 | 3.79× (plateau) | **4.35×** | +15% |

- **Vectorized repetition penalty.** Replaced the per-codebook Python set loop with one on-device
  scatter — byte-identical, removes per-step host work (helps most at large batch).

- **Ragged / continuous batching (drop finished rows).** On mixed-length batches the dense loop runs
  every row until the *longest* finishes — 35–55% of forwards are wasted on already-completed rows.
  We compact the batch (slice the KV-cache batch axis + states) the moment a row hits EOS, keying RNG
  by original index. Measured fork-vs-clone (4-bit, real-EOS, M4 Pro): **80.0 s → 56.6 s = 1.41× at
  B=32**, up to **~1.78× at B=48** (workload-dependent — more length skew reclaims more waste). At
  B=8 the gain is negligible (the forward is dispatch-bound, not compute-bound, at small batch).
  **Not bit-exact** (important caveat): for most sequences output is identical, but the *longest*
  surviving sequence finishes alone (B=1) after the others are compacted away, and a B=1 matmul
  rounds slightly differently than a B=8 one — the AR loop amplifies that into a *different but
  equally-valid* sample. So this is a lossy-but-quality-preserving optimization, not a bit-exact one.
  (Verified: 7/8 sequences identical, only the longest diverged, same length, perceptually equivalent.)

> **Honest ceiling note.** A further *5×* on **single-stream** is not reachable without retraining —
> RTF 0.16 sits below the ~0.26 weight-bandwidth floor (273 GB/s, ~810 MB/token). Single-stream
> pure-inference headroom is ~2.3× (forward-bound). The reachable wins are on **batch throughput**
> (this section) toward a ~10–14× aggregate ceiling.

## Roadmap (further speedups — not yet done)
1. **Single-stream decode pipelining** — `mx.async_eval` + `mx.compile` of the per-step forward+sample
   to recover serialization on the single-stream path (forward-bound at RTF ~0.82; ~1.3–1.6×).
2. **Full batch-sampler vectorization** — fold the per-sequence penalty/filtering into one
   `[B, n_cb, vocab]` op (only the categorical stays per-key) to cut the remaining per-step Python.
3. **Quantize `ChunkedLinear` (wkv/w_in)** — currently bf16; adding a `to_quantized` ≈ +17% forward.
4. **Speculative / multi-token decode** — needs lightweight trained draft heads (backbone frozen);
   ~2–3× single-stream on this hardware, not the CUDA-paper 4–5× (no tree-attention kernel in MLX).

## Credits & license
- **Model:** [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2)
- **MLX framework:** [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) — Prince Canuma (MIT)
- **Reference ZONOS2 MLX port:** [Lucas Newman](https://github.com/lucasnewman) (experimental branch)
- **This optimized fork:** quantization recipe, tooling, benchmarks, and analysis by Amal David

Licensed MIT (see [`LICENSE`](./LICENSE)), inheriting mlx-audio's MIT license.
