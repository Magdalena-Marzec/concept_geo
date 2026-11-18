"""Harrier INT4 extraction: strict inputs, causal last-token embeddings, resume."""

import argparse
import csv
import fcntl
import json
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.gemma3_text import ModelArgs
from mlx_lm.utils import load_model
from transformers import AutoTokenizer

from mac.data import load_dataset, sha256
from mac.harrier.model import HarrierEmbeddingModel
from mac.harrier.preflight import run as preflight

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "mac/harrier/config.json").read_text())
MODEL_DIR = (
    Path.home()
    / "models/harrier_oss_v1_27b_mlx_int4_ff69d92"
)
OUT = ROOT / "embedding_results" / CFG["experiment"]


def save_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_inputs(tokenizer):
    report = preflight(True)
    assert report["status"] == "INPUTS_CHECKED_ONLY"

    result = {}
    for dataset in ("score", "tatum"):
        source, records, _ = load_dataset(dataset)
        assert len(records) == CFG["expected_rows"][dataset]

        ids = [
            tokenizer.encode(
                record["text"],
                add_special_tokens=True,
                truncation=False,
            )
            for record in records
        ]
        lengths = [len(item) for item in ids]

        if not all(0 < n <= CFG["max_tokens_initial"] for n in lengths):
            failures = [
                (i, n)
                for i, n in enumerate(lengths)
                if not 0 < n <= CFG["max_tokens_initial"]
            ]
            raise RuntimeError(
                f"{dataset}: token limit violation; "
                f"first failures={failures[:10]}"
            )

        result[dataset] = (source, records, ids, lengths)
        print(
            f"INPUT OK: {dataset}: {len(records)} texts; "
            f"maximum {max(lengths)} tokens",
            flush=True,
        )

    return result


def write_csvs(directory, dataset, records, lengths):
    if dataset == "score":
        fields = (
            "embedding_index", "row_id", "concept",
            "type", "paraphrase_id",
        )
    else:
        fields = (
            "embedding_index", "row_id", "text_id",
            "type", "source_id",
        )

    with (directory / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(records):
            writer.writerow({
                field: i if field == "embedding_index"
                else record[field]
                for field in fields
            })

    with (directory / "token_lengths.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("embedding_index", "token_count"))
        writer.writerows(enumerate(lengths))


def vector_for(model, ids):
    tokens = mx.array([ids], dtype=mx.int32)
    result = model(tokens)
    mx.eval(result)

    vector = np.asarray(result[0].tolist(), dtype=np.float32)

    if vector.shape != (CFG["dimension"],):
        raise RuntimeError(f"Incorrect embedding shape: {vector.shape}")

    if not np.all(np.isfinite(vector)):
        raise RuntimeError("Non-finite embedding")

    norm = float(np.linalg.norm(vector))
    if not abs(norm - 1.0) < 1e-3:
        raise RuntimeError(f"Incorrect L2 norm: {norm}")

    return vector


def extract_dataset(model, dataset, prepared, tokenizer_sha):
    source, records, ids, lengths = prepared
    directory = OUT / dataset
    directory.mkdir(parents=True, exist_ok=True)

    matrix_path = directory / "embeddings.npy"
    state_path = directory / "progress.json"

    identity = {
        "dataset": dataset,
        "input_sha256": sha256(source),
        "tokenizer_sha256": tokenizer_sha,
        "model_repo": CFG["mlx_repo"],
        "model_revision": CFG["mlx_revision"],
        "rows": len(records),
        "dimension": CFG["dimension"],
        "max_tokens": CFG["max_tokens_initial"],
    }

    if state_path.exists():
        if not matrix_path.exists():
            raise RuntimeError("Progress exists but embedding matrix is missing")

        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise RuntimeError("Resume identity mismatch; refusing to overwrite")

        start = state["next_index"]
    else:
        if matrix_path.exists():
            raise RuntimeError("Embedding matrix exists without resume state")

        matrix = np.lib.format.open_memmap(
            matrix_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(records), CFG["dimension"]),
        )
        matrix.flush()
        del matrix

        start = 0
        save_json(
            state_path,
            {"identity": identity, "next_index": 0},
        )

    if not 0 <= start <= len(records):
        raise RuntimeError("Invalid resume position")

    matrix = np.load(matrix_path, mmap_mode="r+")
    if matrix.shape != (len(records), CFG["dimension"]):
        raise RuntimeError("Existing matrix has wrong shape")
    if matrix.dtype != np.float32:
        raise RuntimeError("Existing matrix has wrong dtype")

    write_csvs(directory, dataset, records, lengths)

    print(
        f"START {dataset}: {start}/{len(records)} completed",
        flush=True,
    )
    start_time = time.monotonic()

    for i in range(start, len(records)):
        matrix[i] = vector_for(model, ids[i])
        matrix.flush()

        save_json(
            state_path,
            {"identity": identity, "next_index": i + 1},
        )

        completed_this_run = i + 1 - start
        if completed_this_run == 1 or (i + 1) % 10 == 0:
            seconds_per_text = (
                (time.monotonic() - start_time) / completed_this_run
            )
            remaining_hours = (
                (len(records) - i - 1) * seconds_per_text / 3600
            )
            print(
                f"{dataset}: {i+1}/{len(records)}, "
                f"{seconds_per_text:.2f} s/text, "
                f"ETA ~{remaining_hours:.2f} h",
                flush=True,
            )

    save_json(directory / "manifest.json", {
        **identity,
        "status": "complete_unvalidated_against_bf16",
        "pooling": CFG["pooling"],
        "attention": CFG["attention"],
        "normalization": CFG["normalization"],
        "input_mode": CFG["input_mode"],
        "dtype": CFG["output_dtype"],
        "mlx_version": version("mlx"),
        "mlx_lm_version": version("mlx-lm"),
        "quantization": CFG["quantization"],
        "bf16_reference_checked": False,
    })

    print(f"COMPLETE {dataset}: {len(records)} embeddings", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.check == args.run:
        parser.error("Specify exactly one of --check or --run")

    assert MODEL_DIR.is_dir(), MODEL_DIR

    shard_names = [
        f"model-{i:05d}-of-00003.safetensors"
        for i in range(1, 4)
    ]
    assert all((MODEL_DIR / name).is_file() for name in shard_names)
    assert not (MODEL_DIR / "model.safetensors.index.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        use_fast=True,
        local_files_only=True,
    )
    prepared = prepare_inputs(tokenizer)

    if args.check:
        print("HARRIER CSV PREFLIGHT OK", flush=True)
        return

    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU unavailable")

    mx.set_default_device(mx.gpu)
    OUT.mkdir(parents=True, exist_ok=True)

    with (OUT / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Harrier extraction is running")

        print("LOADING REAL HARRIER INT4 MODEL", flush=True)
        started = time.monotonic()

        model, config = load_model(
            MODEL_DIR,
            lazy=True,
            strict=True,
            get_model_classes=lambda config: (
                HarrierEmbeddingModel,
                ModelArgs,
            ),
        )

        assert config["quantization"] == CFG["quantization"]
        print(
            f"MODEL STRUCTURE READY in "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )

        first_ids = prepared["score"][2][0]
        probe = vector_for(model, first_ids)

        print(
            f"REAL FORWARD OK: dimension={probe.size}, "
            f"L2={np.linalg.norm(probe):.6f}",
            flush=True,
        )
        print(
            "WARNING: BF16 numerical reference not yet validated",
            flush=True,
        )

        tokenizer_sha = sha256(MODEL_DIR / "tokenizer.json")

        for dataset in ("score", "tatum"):
            extract_dataset(
                model,
                dataset,
                prepared[dataset],
                tokenizer_sha,
            )

        print("HARRIER ALL DATASETS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
