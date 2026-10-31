#!/usr/bin/env python3
"""On your analysis computer: export a single 27B layer to legacy-shaped NPZ.

Example:
  python -m mac.export_layer 3.8 tatum 32 last_token

Unlike Qwen3-Embedding, this causal LM is NOT MRL-trained; no 1024D
truncation is offered. Reads 4D memory-mapped arrays lazily.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

try:
    from mac.data import ROOT
    from mac.embed_layers import MODEL_SLUGS, POOLINGS
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mac.data import ROOT
    from mac.embed_layers import MODEL_SLUGS, POOLINGS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=MODEL_SLUGS)
    p.add_argument("dataset", choices=["score", "tatum", "natural", "external"])
    p.add_argument("layer", type=int)
    p.add_argument("pooling", choices=POOLINGS)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not (0 <= args.layer <= 64):
        p.error("layer must be in [0,64]")
    source = ROOT / "embedding_results/mlx_27b" / MODEL_SLUGS[args.model] / args.dataset
    if not (source / "manifest.json").exists():
        p.error(f"No completed dataset found: {source}")
    meta = json.loads((source / "manifest.json").read_text())
    if meta.get("status") != "complete":
        p.error(f"Dataset incomplete: {source}")
    array = np.load(source / "embeddings.npy", mmap_mode="r")
    if list(array.shape) != meta["shape"]:
        p.error("Bad embedding matrix shape")
    output = args.output or (source / f"layer_{args.layer:02d}_{args.pooling}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = np.asarray(array[:, args.layer, POOLINGS.index(args.pooling), :]).copy()
    np.savez_compressed(output, embeddings_native=selected)
    print(f"Exported {selected.shape} to {output}")
    print(f"Input metadata: {source / 'metadata.csv'}")


if __name__ == "__main__":
    main()
