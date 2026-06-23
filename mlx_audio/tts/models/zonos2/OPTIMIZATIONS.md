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

## Roadmap (further speedups — not yet done)
1. **Decode-loop de-serialization** — on-device sampling, remove the per-frame `.tolist()` sync,
   `mx.async_eval` pipeline, `mx.compile`. Expected ~1.5–1.9× single-stream.
2. **Quantize `ChunkedLinear` (wkv/w_in)** — currently bf16; adding a `to_quantized` ≈ +17% forward.
3. **Ragged-batch compaction** — drop finished rows so a batch isn't paced by its slowest sequence.
4. **Speculative / multi-token decode** on the 9-codebook delay axis — the path to 4–5× single-stream.

## Credits & license
- **Model:** [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2)
- **MLX framework:** [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) — Prince Canuma (MIT)
- **Reference ZONOS2 MLX port:** [Lucas Newman](https://github.com/lucasnewman) (experimental branch)
- **This optimized fork:** quantization recipe, tooling, benchmarks, and analysis by Amal David

Licensed MIT (see [`LICENSE`](./LICENSE)), inheriting mlx-audio's MIT license.
