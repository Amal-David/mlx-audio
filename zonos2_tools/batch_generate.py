"""ZONOS2-4bit (MLX) — batched throughput runner (the lever for running many generations).

Loads the model once and runs many prompts through batch_generate in batches of B.
On an M4 Pro, B = 32-64 is the sweet spot (~3.5x real-time aggregate on real workloads);
throughput saturates past that as top-1 MoE routing hits more distinct experts.

Run:
    python batch_generate.py --texts prompts.txt --batch_size 32
    python batch_generate.py                       # built-in demo corpus
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import mlx.core as mx
from mlx_audio.tts import load
from mlx_audio.audio_io import write as audio_write

MODEL = "amal-david/Zyphra-ZONOS2-4bit"
FPS = 44100 / 512  # audio frames per second


def peak_gb():
    try:
        return mx.get_peak_memory() / 1e9
    except Exception:
        return float("nan")


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--texts", help="file with one prompt per line")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--ref_audio", default=None, help="shared reference voice for all rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/batch")
    args = ap.parse_args()

    texts = ([ln.strip() for ln in open(args.texts) if ln.strip()] if args.texts
             else [f"This is benchmark sentence number {i+1}." for i in range(64)])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model = load(args.model, lazy=False)
    spk = model.extract_speaker_embedding(args.ref_audio) if args.ref_audio else None

    t0, total_frames = time.perf_counter(), 0
    for base, batch in chunks(texts, args.batch_size):
        t1 = time.perf_counter()
        results = sorted(
            model.batch_generate(batch, speaker_embedding=spk,
                                 max_tokens=args.max_tokens, seed=args.seed),
            key=lambda r: r.sequence_idx)
        dt = time.perf_counter() - t1
        bf = sum(int(r.token_count) for r in results)
        total_frames += bf
        for r in results:
            audio_write(str(out / f"sample_{base + r.sequence_idx:04d}.wav"), r.audio, r.sample_rate)
        print(f"  batch[{base:4d}:{base+len(batch):4d}] wall={dt:6.2f}s "
              f"aggregate_realtime_x={(bf/FPS)/dt:5.2f} peak={peak_gb():.1f}GB", flush=True)

    wall = time.perf_counter() - t0
    audio_s = total_frames / FPS
    print(f"\n{len(texts)} clips, {audio_s:.1f}s audio in {wall:.1f}s "
          f"=> {audio_s/wall:.2f}x real-time aggregate, peak={peak_gb():.1f}GB -> {out}")


if __name__ == "__main__":
    main()
