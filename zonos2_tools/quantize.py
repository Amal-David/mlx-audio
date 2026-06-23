"""Reproduce this mixed-precision quantization from the BF16 base.

Recipe (verified to keep generation healthy — uniform 4-bit breaks EOS because
4-bit *attention* corrupts the model; the experts tolerate 4-bit fine):
  - 4-bit gs64 : MoE experts (SwitchGLU)         <- 94.5% of weights, the bandwidth lever
  - 8-bit gs64 : attention wq/wo, token embeddings
  - bf16       : output head, MoE router, gater, per-head temp, RMSNorm, ChunkedLinear

Run:
    pip install git+https://github.com/lucasnewman/mlx-audio.git@zonos2-stream-and-batching
    python quantize.py --out Zyphra-ZONOS2-4bit
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from mlx_audio.tts.utils import convert

SRC = "mlx-community/Zyphra-ZONOS2"
SKIP = ("router", "multi_output", "gater", "norm", "temp")


def quant_predicate(path, module):
    if any(s in path for s in SKIP):
        return False
    if "experts" in path:
        return {"group_size": 64, "bits": 4}
    if "attention" in path:                 # wq/wo (gater already skipped); 4-bit here breaks EOS
        return {"group_size": 64, "bits": 8}
    if "embedders" in path:
        return {"group_size": 64, "bits": 8}
    return False                            # everything else stays bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="Zyphra-ZONOS2-4bit")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    convert(args.src, str(out), quantize=True, q_group_size=64, q_bits=4, q_mode="affine",
            quant_predicate=quant_predicate)

    # convert() copies the source bf16 shards in too; the loader globs *.safetensors,
    # so prune any shard not referenced by the quantized index.
    idx = json.load(open(out / "model.safetensors.index.json"))
    keep = set(idx["weight_map"].values())
    for f in out.glob("*.safetensors"):
        if f.name not in keep:
            f.unlink()

    # ensure the speaker encoder (voice cloning) is present
    if not (out / "speaker_encoder").exists():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(args.src, allow_patterns=["speaker_encoder/*"]))
        shutil.copytree(src / "speaker_encoder", out / "speaker_encoder", dirs_exist_ok=True)

    sz = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    print(f"done -> {out}  ({sz:.2f} GB)")


if __name__ == "__main__":
    main()
