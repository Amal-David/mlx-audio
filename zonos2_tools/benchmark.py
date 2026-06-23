"""Benchmark ZONOS2 in MLX: bf16 vs 8-bit vs 4-bit, single-stream RTF + batch throughput.

Reproduces the numbers in ZONOS2_OPTIMIZED.md on your own machine.

Run:
    python benchmark.py                       # bf16 baseline + 4-bit, B in {1,8,32}
    python benchmark.py --bits 8 --batches 1,16,32,64
"""
from __future__ import annotations
import argparse, time
import mlx.core as mx
import mlx.nn as nn
from mlx_audio.tts import load

BASE = "mlx-community/Zyphra-ZONOS2"          # bf16 source
FPS = 44100 / 512
TEXT = ("This is a fixed benchmark sentence used to compare decode speed and batch "
        "throughput of the ZONOS two model on Apple Silicon.")
SKIP = ("router", "multi_output", "gater", "norm", "temp")

def peak_gb():
    try: return mx.get_peak_memory() / 1e9
    except Exception: return float("nan")
def reset_peak():
    try: mx.reset_peak_memory()
    except Exception: pass

def quantize(model, bits):
    """Mixed-precision: experts at `bits`, attention+embeddings at max(8,bits)."""
    attn_emb_bits = max(8, bits)
    def cp(p, m):
        if any(s in p for s in SKIP): return False
        if not hasattr(m, "to_quantized"): return False
        if "experts" in p:   return {"group_size": 64, "bits": bits}
        if "attention" in p: return {"group_size": 64, "bits": attn_emb_bits}
        if "embedders" in p: return {"group_size": 64, "bits": attn_emb_bits}
        return False
    nn.quantize(model, group_size=64, bits=bits, mode="affine", class_predicate=cp)

def bench(label, bits, batches):
    m = load(BASE, lazy=False)
    if bits:
        quantize(m, bits)
    next(m.generate(text=TEXT, max_tokens=48, ignore_eos=True, seed=1))  # warmup
    print(f"\n=== {label} ===", flush=True)
    for B in batches:
        reset_peak()
        t0 = time.perf_counter()
        results = list(m.batch_generate([TEXT] * B, max_tokens=200, ignore_eos=True, seed=1))
        dt = time.perf_counter() - t0
        frames = sum(int(r.token_count) for r in results)
        audio_s = (200 / FPS)
        print(f"  B={B:3d}: per-stream RTF={dt/audio_s:5.2f}  "
              f"aggregate_realtime_x={(B*audio_s)/dt:5.2f}  {frames/dt:6.0f} frames/s  "
              f"peak={peak_gb():.1f}GB", flush=True)
    del m; mx.clear_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4, help="experts quant bits (4 or 8); 0 = skip quant")
    ap.add_argument("--batches", default="1,8,32", help="comma-separated batch sizes")
    args = ap.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    bench("bf16 (baseline)", None, batches)
    if args.bits:
        bench(f"{args.bits}-bit experts (mixed)", args.bits, batches)

if __name__ == "__main__":
    main()
