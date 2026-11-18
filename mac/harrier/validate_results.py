"""Validate Harrier embedding alignment, metadata and geometry without loading 27B."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from mac.data import load_dataset, sha256
from mac.harrier.preflight import run as preflight

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "embedding_results/harrier_oss_v1_27b_mlx_int4"
REPORTS = OUT / "validation"
MODEL = Path.home() / "models/harrier_oss_v1_27b_mlx_int4_ff69d92"

EXPECTED_ROWS = {"score": 1827, "tatum": 597}
DIMENSION = 5376

FIELDS = {
    "score": [
        "embedding_index", "row_id", "concept",
        "type", "paraphrase_id",
    ],
    "tatum": [
        "embedding_index", "row_id", "text_id",
        "type", "source_id",
    ],
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    return fields, rows


def save_json(name, data):
    path = REPORTS / name
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("SAVED:", path.relative_to(ROOT))


def cosine_pairs(x, pairs, batch=512):
    result = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        a = x[chunk[:, 0]]
        b = x[chunk[:, 1]]
        result.extend(np.sum(a * b, axis=1).tolist())
    return np.asarray(result, dtype=np.float32)


def describe(values):
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def validate_metadata(dataset, tokenizer):
    directory = OUT / dataset
    source, records, _ = load_dataset(dataset)
    n = EXPECTED_ROWS[dataset]

    assert len(records) == n, f"{dataset}: source row mismatch"

    matrix = np.load(
        directory / "embeddings.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    assert matrix.shape == (n, DIMENSION)
    assert matrix.dtype == np.float32

    manifest = json.loads(
        (directory / "manifest.json").read_text()
    )
    progress = json.loads(
        (directory / "progress.json").read_text()
    )

    assert manifest["rows"] == n
    assert manifest["dimension"] == DIMENSION
    assert manifest["input_sha256"] == sha256(source)
    assert manifest["model_revision"] == (
        "ff69d92ce0aff47e8b8f8dfb48715888046d970a"
    )
    assert manifest["status"] == "complete_unvalidated_against_bf16"
    assert progress["next_index"] == n
    assert progress["identity"]["input_sha256"] == sha256(source)
    assert progress["identity"]["rows"] == n

    tokenizer_hash = sha256(MODEL / "tokenizer.json")
    assert manifest["tokenizer_sha256"] == tokenizer_hash

    fields, metadata = read_csv(directory / "metadata.csv")
    assert fields == FIELDS[dataset]
    assert len(metadata) == n

    for i, (saved, original) in enumerate(zip(metadata, records)):
        for field in fields:
            expected = str(i) if field == "embedding_index" else str(original[field])
            assert saved[field] == expected, (
                f"{dataset}, row {i}, {field}: "
                f"{saved[field]!r} != {expected!r}"
            )

    length_fields, saved_lengths = read_csv(
        directory / "token_lengths.csv"
    )
    assert length_fields == ["embedding_index", "token_count"]
    assert len(saved_lengths) == n

    lengths = []

    for i, (saved, original) in enumerate(
        zip(saved_lengths, records)
    ):
        ids = tokenizer.encode(
            original["text"],
            add_special_tokens=True,
            truncation=False,
        )
        count = len(ids)

        assert 0 < count <= 2048, (dataset, i, count)
        assert int(saved["embedding_index"]) == i
        assert int(saved["token_count"]) == count, (
            dataset, i, count
        )
        lengths.append(count)

    result = {
        "dataset": dataset,
        "status": "PASS",
        "rows": n,
        "dimension": DIMENSION,
        "dtype": "float32",
        "source_sha256": sha256(source),
        "tokenizer_sha256": tokenizer_hash,
        "metadata_order_matches_source": True,
        "token_lengths_match_tokenizer": True,
        "max_tokens": int(max(lengths)),
        "mean_tokens": float(np.mean(lengths)),
        "bf16_reference_checked": False,
    }

    print(
        f"METADATA PASS: {dataset} | "
        f"{n} rows | max_tokens={max(lengths)}",
        flush=True,
    )

    return np.asarray(matrix), metadata, result


def validate_geometry(dataset, x, metadata):
    n, dim = x.shape

    assert (n, dim) == (EXPECTED_ROWS[dataset], DIMENSION)
    assert np.all(np.isfinite(x)), f"{dataset}: NaN or Inf"

    norms = np.linalg.norm(x, axis=1)
    assert np.all(norms > 0), f"{dataset}: zero vector"
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"{dataset}: vectors not L2-normalized"
    )

    # SHA-256 on exact float32 bytes; no destructive deduplication.
    fingerprints = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in x
    ]
    counts = Counter(fingerprints)
    duplicate_groups = sum(v > 1 for v in counts.values())
    duplicate_excess = sum(v - 1 for v in counts.values())

    normalized = x / norms[:, None]

    rng = np.random.default_rng(20261118)
    candidate_pairs = rng.integers(
        0, n, size=(12000, 2), dtype=np.int64
    )
    candidate_pairs = candidate_pairs[
        candidate_pairs[:, 0] != candidate_pairs[:, 1]
    ]

    pair_cosines = cosine_pairs(normalized, candidate_pairs)

    # Resultant length: useful preliminary indicator of anisotropy.
    mean_vector = np.mean(normalized, axis=0)
    mean_vector_norm = float(np.linalg.norm(mean_vector))

    result = {
        "dataset": dataset,
        "status": "PASS",
        "vectors": n,
        "dimension": dim,
        "norms": describe(norms),
        "exact_duplicate_groups": int(duplicate_groups),
        "exact_duplicate_excess": int(duplicate_excess),
        "random_pair_cosine": describe(pair_cosines),
        "mean_vector_norm": mean_vector_norm,
        "mean_vector_norm_squared": mean_vector_norm ** 2,
        "anisotropy_note": (
            "Preliminary indicator only; not a complete "
            "spectral anisotropy analysis."
        ),
    }

    if dataset == "score":
        groups = defaultdict(list)
        for i, row in enumerate(metadata):
            groups[row["concept"]].append(i)

        within_pairs = []

        for indices in groups.values():
            if len(indices) < 2:
                continue

            for _ in range(20):
                a, b = rng.choice(indices, size=2, replace=False)
                within_pairs.append((int(a), int(b)))

        within_pairs = np.asarray(within_pairs, dtype=np.int64)

        if len(within_pairs):
            within_cosines = cosine_pairs(
                normalized, within_pairs
            )
            result["within_concept_cosine"] = describe(
                within_cosines
            )

        result["concept_count"] = len(groups)
        result["concept_sizes"] = dict(
            Counter(len(v) for v in groups.values())
        )

    print(
        f"GEOMETRY PASS: {dataset} | "
        f"duplicates={duplicate_excess} | "
        f"mean_cos={np.mean(pair_cosines):.4f} | "
        f"mean_vector_norm={mean_vector_norm:.4f}",
        flush=True,
    )

    return result, fingerprints


def main():
    assert MODEL.is_dir(), "Local Harrier model directory missing"

    report = preflight(True)
    assert report["status"] == "INPUTS_CHECKED_ONLY"

    REPORTS.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL),
        use_fast=True,
        local_files_only=True,
    )

    data = {}
    metadata_reports = {}
    geometry_reports = {}
    all_hashes = {}

    for dataset in ("score", "tatum"):
        matrix, metadata, meta_report = validate_metadata(
            dataset, tokenizer
        )
        geometry_report, hashes = validate_geometry(
            dataset, matrix, metadata
        )

        data[dataset] = matrix
        metadata_reports[dataset] = meta_report
        geometry_reports[dataset] = geometry_report
        all_hashes[dataset] = set(hashes)

    cross_duplicates = len(
        all_hashes["score"] & all_hashes["tatum"]
    )

    geometry_reports["cross_dataset"] = {
        "exact_shared_embedding_vectors": cross_duplicates
    }

    # Dataset-level comparisons, not a substitute for clinical validation.
    geometry_reports["interpretation"] = (
        "Random-pair cosine statistics and mean-vector norm "
        "are descriptive diagnostics, not scientific validation "
        "of semantic quality or equivalence to BF16."
    )

    save_json("metadata_validation.json", metadata_reports)
    save_json("geometry_validation.json", geometry_reports)

    summary = {
        "status": "PASS",
        "model": "harrier-oss-v1-27b-MLX-4bit",
        "total_vectors": sum(EXPECTED_ROWS.values()),
        "dimension": DIMENSION,
        "metadata_validation": "PASS",
        "geometric_sanity": "PASS",
        "bf16_reference_checked": False,
        "reports": [
            "metadata_validation.json",
            "geometry_validation.json",
        ],
    }

    save_json("validation_summary.json", summary)

    print("\nHARRIER FINAL METADATA AND GEOMETRY VALIDATION PASSED")


if __name__ == "__main__":
    main()
