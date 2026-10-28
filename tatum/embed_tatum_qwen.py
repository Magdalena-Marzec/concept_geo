#!/usr/bin/env python3
"""Embed Tatum fragments/paragraphs/captions using EXACT existing Qwen pipeline.

Run from the concept_geo root:
    python -u tatum/embed_tatum_qwen.py --models 0.6B 4B 8B
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from embed_three_qwen import MODEL_SPECS, COMMON_DIM, MAX_LENGTH, encode_texts, load_model, save_npz

INPUT = REPO / "tatum/data/tatum_embedding_texts.csv"
OUTPUT = REPO / "embedding_results/tatum"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["0.6B", "4B", "8B"], choices=list(MODEL_SPECS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT}: run python tatum/prepare_inputs.py first")

    df = pd.read_csv(INPUT, dtype={"text_id": str, "type": str, "source_id": str, "text": str})
    required = {"row_id", "text_id", "type", "source_id", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Input has null required fields")
    if df["text_id"].duplicated().any():
        raise ValueError("Duplicate text_id")
    if df["row_id"].tolist() != list(range(len(df))):
        raise ValueError("Row IDs must be zero-based and consecutive")
    if not df["type"].isin(["fragment", "paragraph", "caption"]).all():
        raise ValueError("Unexpected text type")
    if df["text"].str.strip().eq("").any():
        raise ValueError("Empty input text")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Exclude verbatim textbook excerpts from the public repository.
    ids_only = df[["row_id", "text_id", "type", "source_id"]].copy()
    ids_only.to_csv(OUTPUT / "tatum_embeddings_metadata.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(INPUT),
        "total_texts": len(df),
        "type_counts": {k: int(v) for k,v in df["type"].value_counts().items()},
        "pooling": "last non-padding token",
        "normalization": "L2",
        "instruction": None,
        "max_length": MAX_LENGTH,
        "inference_dtype": "float16",
        "saved_dtype": "float32",
        "common_mrl_dimension": COMMON_DIM,
        "models": {},
    }
    texts = df["text"].tolist()
    device = torch.device("cuda")
    for size in args.models:
        spec = MODEL_SPECS[size]
        slug = size.lower().replace(".", "_")
        print(f"\n=== {spec['repo_id']} ===", flush=True)
        tokenizer, model = load_model(spec, device)
        native, common = encode_texts(model, tokenizer, texts, spec["batch_size"], spec["native_dim"], device)
        expected = (len(df), spec["native_dim"])
        if native.shape != expected or common.shape != (len(df), COMMON_DIM):
            raise RuntimeError(f"Unexpected dimensions: {native.shape}, {common.shape}")
        if not np.isfinite(native).all() or not np.isfinite(common).all():
            raise RuntimeError("Non-finite embeddings")
        for array in (native, common):
            if not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-5):
                raise RuntimeError("Embeddings are not L2-normalized")
        out_path = OUTPUT / f"tatum_qwen3_{slug}.npz"
        save_npz(out_path, native, common)
        manifest["models"][size] = {
            "repo_id": spec["repo_id"],
            "native_dim": spec["native_dim"],
            "file": str(out_path.relative_to(REPO)),
            "shape_native": list(native.shape),
            "shape_1024": list(common.shape),
        }
        (OUTPUT / "tatum_embedding_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print("Saved", out_path, flush=True)
        del model, tokenizer, native, common
        gc.collect()
        torch.cuda.empty_cache()
    print("TATUM EMBEDDINGS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
