"""ZONOS2-4bit (MLX) — single-stream, voice cloning, and long-form examples.

Requires mlx-audio with ZONOS2 support:
    pip install git+https://github.com/lucasnewman/mlx-audio.git@zonos2-stream-and-batching

Run:
    python generate.py
    python generate.py --text "Custom line." --ref_audio speaker.wav
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import mlx.core as mx
from mlx_audio.tts import load
from mlx_audio.audio_io import write as audio_write

MODEL = "amal-david/Zyphra-ZONOS2-4bit"


def speak_long(model, text, ref_audio=None, max_chars=350, gap_s=0.12, seed=42):
    """Long-form: split on sentence boundaries, batch, concatenate with small gaps."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) > max_chars and cur:
            chunks.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    spk = model.extract_speaker_embedding(ref_audio) if ref_audio else None
    results = sorted(
        model.batch_generate(chunks, speaker_embedding=spk, max_tokens=1024, seed=seed),
        key=lambda r: r.sequence_idx,
    )
    gap = mx.zeros((int(gap_s * model.sample_rate),))
    pieces = []
    for i, r in enumerate(results):
        if i:
            pieces.append(gap)
        pieces.append(r.audio)
    return mx.concatenate(pieces, axis=0), model.sample_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--text", default="Hello, this is the four bit ZONOS two model on Apple Silicon.")
    ap.add_argument("--ref_audio", default=None, help="reference clip for voice cloning")
    ap.add_argument("--long_file", default=None, help="a .txt/.md file to read as long-form")
    ap.add_argument("--out", default="outputs");
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = load(args.model, lazy=False)

    if args.long_file:
        audio, sr = speak_long(model, Path(args.long_file).read_text(), ref_audio=args.ref_audio)
        audio_write(f"{args.out}/long.wav", audio, sr)
        print(f"wrote {args.out}/long.wav ({audio.shape[0]/sr:.1f}s)")
        return

    spk = model.extract_speaker_embedding(args.ref_audio) if args.ref_audio else None
    r = next(model.generate(text=args.text, speaker_embedding=spk, max_tokens=1024, seed=42))
    audio_write(f"{args.out}/zonos2.wav", r.audio, r.sample_rate)
    print(f"wrote {args.out}/zonos2.wav  dur={r.audio_duration}  RTF={r.real_time_factor:.3f}")


if __name__ == "__main__":
    main()
