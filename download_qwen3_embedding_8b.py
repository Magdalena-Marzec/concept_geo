#!/usr/bin/env python3
from pathlib import Path
import json
from datetime import datetime, timezone

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Qwen/Qwen3-Embedding-8B"
LOCAL_DIR = Path("models/Qwen3-Embedding-8B")
MANIFEST = LOCAL_DIR / "download_manifest.json"

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()
info = api.model_info(REPO_ID)
revision = info.sha

print(f"Model: {REPO_ID}")
print(f"Revision: {revision}")
print(f"Destination: {LOCAL_DIR}")

snapshot_download(
    repo_id=REPO_ID,
    revision=revision,
    local_dir=str(LOCAL_DIR),
)

manifest = {
    "repo_id": REPO_ID,
    "revision": revision,
    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
}

MANIFEST.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8"
)

print("Download complete.")
print(f"Manifest: {MANIFEST}")
