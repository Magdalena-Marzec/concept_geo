#!/usr/bin/env python3
"""MLX-VLM Qwen3.8/3.5 27B: collect all 64 *text* decoder hidden states.

One full text prefill per input. No generate(), no LM head, no vision forward.
B=1, no padding; mean pooling over all tokenizer output tokens.
Raw post-block states are captured *before* final model RMSNorm. Layer 0
is the token embedding lookup. Both pooled vectors are L2-normalized.

The per-layer capture instrument wraps the actual upstream decoder layer
forward (no reimplementation of Gated DeltaNet / masks / MRoPE). This
relies on mlx-vlm's current Qwen3.5 model class; fail fast if it changes.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from mac.data import ROOT, load_dataset, sha256, write_metadata
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mac.data import ROOT, load_dataset, sha256, write_metadata

MODEL_IDS = {
    "3.8": "mlx-community/Qwen3.8-27B-8bit",
    "3.5": "mlx-community/Qwen3.5-27B-8bit",
}
MODEL_SLUGS = {"3.8": "qwen3_8_27b_8bit", "3.5": "qwen3_5_27b_8bit"}
DATASETS = ("score", "tatum", "natural", "external")
LAYERS = 64
DIM = 5120
POOLINGS = ("last_token", "mean")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def resolve_backbone(model):
    """Use MLX-VLM's exact text backbone, never logits-producing __call__."""
    language = getattr(model, "language_model", None)
    backbone = getattr(language, "model", None)
    layers = getattr(backbone, "layers", None)
    if backbone is None or layers is None or len(layers) != LAYERS:
        raise RuntimeError("Unknown mlx-vlm Qwen3.5 model layout; expected language_model.model.layers[64].")
    if not hasattr(backbone, "embed_tokens"):
        raise RuntimeError("No text embedding layer; inspect installed mlx-vlm version.")
    return backbone, list(layers)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(x))
    if not np.isfinite(x).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Invalid hidden-state pooling vector (NaN/Inf/zero norm).")
    return x / norm


def pooled_arrays(hidden, mx):
    """Collect B=1 last-token and mean in float32, forcing evaluation per layer."""
    if len(hidden.shape) != 3 or hidden.shape[0] != 1 or hidden.shape[2] != DIM:
        raise RuntimeError(f"Unexpected token hidden-state shape: {hidden.shape}")
    last = hidden[0, -1, :].astype(mx.float32)
    mean = mx.mean(hidden[0, :, :].astype(mx.float32), axis=0)
    mx.eval(last, mean)
    return np.stack([normalize(np.array(last)), normalize(np.array(mean))], axis=0)


@contextmanager
def capture_layer_outputs(layers, mx, result):
    """Temporary class-level wrappers preserve each layer's native MLX forward.

    Hooks evaluate and transfer only two vectors, not entire token activations.
    """
    index = {id(layer): i + 1 for i, layer in enumerate(layers)}
    saved = {}
    if len(index) != LAYERS:
        raise RuntimeError("Duplicate layer object: capture would be ambiguous.")
    for cls in {type(layer) for layer in layers}:
        original = cls.__call__
        saved[cls] = original
        def tap(self, *args, _original=original, **kwargs):
            state = _original(self, *args, **kwargs)
            number = index.get(id(self))
            if number is not None:
                if result[number] is not None:
                    raise RuntimeError(f"Decoder layer {number} called twice")
                result[number] = pooled_arrays(state, mx)
            return state
        cls.__call__ = tap
    try:
        yield
    finally:
        for cls, original in saved.items():
            cls.__call__ = original


def embed_one(model, tokenizer, text, max_length: int, mx):
    """Single text-only prefill. Explicitly no chat template and no generated tokens."""
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids:
        raise ValueError("Tokenizer returned empty input for nonempty text")
    if len(ids) > max_length:
        raise ValueError(f"Input token length {len(ids)} exceeds --max-length={max_length}; refusing silent truncation")
    tokens = mx.array([ids], dtype=mx.int32)
    backbone, layers = resolve_backbone(model)
    collected = [None] * (LAYERS + 1)
    collected[0] = pooled_arrays(backbone.embed_tokens(tokens), mx)

    # Qwen3.5 VLM's official input path prepares text-specific 3D MRoPE IDs.
    features = model.get_input_embeddings(input_ids=tokens)
    with capture_layer_outputs(layers, mx, collected):
        # These kwargs mirror the library's hidden_state_at_layer() method.
        # Unlike calling it 64 times, capture all intermediate states in one pass.
        output = backbone(
            tokens,
            inputs_embeds=features.inputs_embeds,
            position_ids=features.position_ids,
            stop_after_layer=LAYERS,
            apply_final_norm=False,
        )
        mx.eval(output)
    if any(x is None for x in collected):
        missing = [i for i, x in enumerate(collected) if x is None]
        raise RuntimeError(f"Some hidden states were not captured: {missing}")
    vectors = np.stack(collected, axis=0)  # [65,2,5120]
    if vectors.shape != (LAYERS + 1, 2, DIM):
        raise RuntimeError(f"Bad embedding shape: {vectors.shape}")
    if not np.allclose(np.linalg.norm(vectors, axis=-1), 1.0, atol=3e-5):
        raise RuntimeError("L2 normalization failed")
    return vectors, len(ids)


def get_model(model_version, offline):
    try:
        from huggingface_hub import snapshot_download
        from mlx_vlm import load
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("Missing macOS dependencies. Run bash setup_mac.sh") from exc
    if not mx.metal.is_available():
        raise RuntimeError("Metal GPU is not available; this runner needs Apple Silicon.")
    model_id = MODEL_IDS[model_version]
    local = ROOT / "models" / MODEL_SLUGS[model_version]
    if offline:
        if not (local / "config.json").exists():
            raise RuntimeError(f"Model not downloaded: {local}. First run without --offline.")
    else:
        local.mkdir(parents=True, exist_ok=True)
        print(f"Checking/downloading {model_id} (~29.5 GB) to {local}", flush=True)
        snapshot_download(repo_id=model_id, local_dir=str(local))
    print("Loading 8-bit checkpoint (may take substantial unified memory)...", flush=True)
    model, processor = load(str(local))
    if hasattr(model, "eval"):
        model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)
    if not callable(tokenizer):
        raise RuntimeError("mlx-vlm processor has no callable tokenizer")
    _, layers = resolve_backbone(model)
    # Validate type/ordering of full-attention blocks, do not infer merely from config.
    attention = [i for i, l in enumerate(layers, 1) if not getattr(l, "is_linear", i % 4 != 0)]
    if attention and attention != list(range(4, 65, 4)):
        raise RuntimeError(f"Unexpected model full-attention positions: {attention}")
    revision = "unknown"
    # The local snapshot may not expose a .git revision; record provenance via HF metadata.
    try:
        from huggingface_hub import HfApi
        if not offline:
            revision = HfApi().model_info(model_id).sha
    except Exception as exc:
        print(f"WARNING: cannot resolve remote model SHA ({exc}); inspect local snapshot", flush=True)
    return mx, model, tokenizer, model_id, revision, str(local)


def run_dataset(name, args, mx, model, tokenizer, model_id, revision, local_dir):
    src, all_records, fields = load_dataset(name)
    records = all_records[:args.limit] if args.limit else all_records
    n = len(records)
    out_dir = ROOT / "embedding_results" / "mlx_27b" / MODEL_SLUGS[args.model]
    if args.limit:
        out_dir = out_dir / "smoke"
    out_dir = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    array_path = out_dir / "embeddings.npy"
    manifest_path = out_dir / "manifest.json"
    checkpoint_path = out_dir / "progress.json"
    metadata_path = out_dir / "metadata.csv"

    expected = {
        "model_id": model_id,
        "local_config_sha256": sha256(Path(local_dir) / "config.json"),
        "local_weights_index_sha256": (
            sha256(Path(local_dir) / "model.safetensors.index.json")
            if (Path(local_dir) / "model.safetensors.index.json").exists() else None
        ),
        "local_tokenizer_sha256": (
            sha256(Path(local_dir) / "tokenizer.json")
            if (Path(local_dir) / "tokenizer.json").exists() else None
        ),
        "source_sha256": sha256(src),
        "source_path": str(src.relative_to(ROOT)),
        "rows": n,
        "shape": [n, LAYERS + 1, len(POOLINGS), DIM],
        "poolings": list(POOLINGS),
        "output_dtype": "float32",
        "layer0": "token_embedding_before_transformer",
        "layers1_to_64": "after_decoder_block_before_final_RMSNorm",
        "model_backend": "mlx-vlm",
        "quantization": "8-bit (mlx-community)",
        "tokenizer_add_special_tokens": True,
        "chat_template": None,
        "max_length": args.max_length,
        "truncation": "error",
        "L2_normalized": True,
        "input_language": "en",
    }
    if args.force:
        for old in (array_path, manifest_path, checkpoint_path, metadata_path):
            if old.exists():
                old.unlink()
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state["settings"] != expected:
            raise RuntimeError(f"Resume settings do not match for {name}; run with --force only if you intend to discard old result.")
        if not array_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Checkpoint exists but embedding/metadata missing: {out_dir}")
        matrix = np.load(array_path, mmap_mode="r+")
        if list(matrix.shape) != expected["shape"]:
            raise RuntimeError("Existing array has wrong shape")
        start = state["next_row"]
        print(f"RESUME {name}: {start}/{n}", flush=True)
    else:
        if any(p.exists() for p in (array_path, manifest_path, metadata_path)):
            raise RuntimeError(f"Existing files without checkpoint: {out_dir}. Use --force to replace.")
        matrix = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32, shape=tuple(expected["shape"]))
        write_metadata(metadata_path, records, fields)
        start = 0
        write_json_atomic(checkpoint_path, {"settings": expected, "next_row": 0, "created_at_utc": now()})

    token_lengths_path = out_dir / "token_lengths.csv"
    # On resumption rewrite length log from first interrupted row onward.
    if start == 0 or not token_lengths_path.exists():
        with token_lengths_path.open("w", encoding="utf-8") as f:
            f.write("embedding_row,token_count\n")
    else:
        with token_lengths_path.open(encoding="utf-8") as f:
            prior = list(csv.reader(f))
        with token_lengths_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["embedding_row", "token_count"])
            w.writerows([row for row in prior[1:] if int(row[0]) < start])

    print(f"DATASET {name}: {n} texts, output {array_path}", flush=True)
    for row_idx in range(start, n):
        vector, token_len = embed_one(model, tokenizer, records[row_idx]["text"], args.max_length, mx)
        matrix[row_idx] = vector
        matrix.flush()  # complete data on disk before advancing progress
        with token_lengths_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row_idx, token_len])
        write_json_atomic(checkpoint_path, {"settings": expected, "next_row": row_idx + 1, "created_at_utc": now()})
        if (row_idx + 1) % 10 == 0 or row_idx + 1 == n:
            print(f"{name}: {row_idx + 1}/{n} ({token_len} tokens last)", flush=True)

    stats = {
        **expected,
        "completed_at_utc": now(),
        "status": "complete",
        "model_local_path": local_dir,
        "model_revision_resolved_online": revision,
        "numpy_version": np.__version__,
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_vlm_version": importlib.metadata.version("mlx-vlm"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "output_file": "embeddings.npy",
        "metadata_file": "metadata.csv",
        "token_lengths_file": "token_lengths.csv",
        "attention_layers_1based": list(range(4, 65, 4)),
    }
    write_json_atomic(manifest_path, stats)
    del matrix
    print(f"COMPLETE {name}: {manifest_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, default="3.8")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, help="Smoke run only: first N rows to separate /smoke folder")
    parser.add_argument("--offline", action="store_true", help="Use already downloaded model")
    parser.add_argument("--force", action="store_true", help="Delete and re-create selected output dataset(s)")
    parser.add_argument("--check-inputs", action="store_true", help="Only validate and count CSVs (no model imports/download)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    for ds in args.datasets:
        src, rows, _ = load_dataset(ds)
        print(f"{ds}: {len(rows)} texts from {src.relative_to(ROOT)} sha256={sha256(src)[:16]}", flush=True)
    if args.check_inputs:
        return
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("MLX GPU runner requires macOS Apple Silicon (arm64); --check-inputs works elsewhere")
    mx, model, tokenizer, model_id, revision, local = get_model(args.model, args.offline)
    print(f"Model {model_id} revision {revision}", flush=True)
    for ds in args.datasets:
        run_dataset(ds, args, mx, model, tokenizer, model_id, revision, local)


if __name__ == "__main__":
    main()
