#!/usr/bin/env python3
"""Download frozen snapshots outside Git; requires network on download node.

Usage:
  python -m v100_next_models.download --models medembed pubmedbert sapbert
  python -m v100_next_models.download --models nvembed nemotron_medical
  python -m v100_next_models.download --models medemb2b medemb8b
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from .config import MODELS, model_dir, select_models

# Exclude duplicated non-PyTorch weights (bin/tf/flax, ONNX etc), preserve
# custom Python code, tokenizer files and safetensors shards/indexes.
ALLOW = ["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "*.tiktoken", "*.sentencepiece", "vocab*", "merges*", "*.md"]


def download(name: str):
    spec = MODELS[name]
    path = model_dir(name)
    path.mkdir(parents=True, exist_ok=True)
    meta_file = path / "mag_download_manifest.json"
    api = HfApi()
    try:
        info = api.model_info(spec.repo_id)
    except Exception as e:
        raise RuntimeError(f"Cannot access {spec.repo_id}; check HF login/terms: {e}") from e
    sha = info.sha
    print(f"DOWNLOAD {name}: {spec.repo_id} @ {sha} -> {path}", flush=True)
    snapshot_download(
        repo_id=spec.repo_id, revision=sha, local_dir=str(path),
        allow_patterns=ALLOW, max_workers=4,
    )
    if not (path / "config.json").exists() and not (path / "modules.json").exists():
        raise RuntimeError(f"No model config in {path}: inspect snapshot before inference")
    manifest = {
        "name": name, "repo_id": spec.repo_id, "revision": sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": spec.backend, "dtype": spec.dtype,
        "experimental": spec.experimental,
        "excluded_weight_formats": ["*.bin", "*.h5", "*.msgpack", "*.onnx"],
    }
    meta_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print("DONE", name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS), help="Default: 5 core models")
    args = parser.parse_args()
    errors = []
    for name in select_models(args.models):
        try:
            download(name)
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)
            errors.append(name)
    if errors:
        raise SystemExit(f"Download errors: {errors}")


if __name__ == "__main__":
    main()
