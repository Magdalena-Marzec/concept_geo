"""Standard-library-only input adapters; preserve ordering from existing CUDA scripts."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE_COLUMNS = ["concept", "definition", *(f"paraphrase_{i}" for i in range(1, 8))]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run git pull / check tracked Tatum CSVs.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _nonempty(row, column):
    if column not in row:
        raise ValueError(f"Missing column {column!r}")
    result = (row[column] or "").strip()
    if not result:
        raise ValueError(f"Empty {column!r} cell: {row}")
    return result


def load_dataset(name: str):
    """Return (source CSV, records, id keys) in exact legacy embedding order."""
    if name == "score":
        path = ROOT / "score_concepts_completed.csv"
        rows = _csv(path)
        records = []
        for idx, row in enumerate(rows):
            for col in SCORE_COLUMNS:
                value = _nonempty(row, col)
                entry = {
                    "row_id": idx,
                    "concept": row["concept"].strip(),
                    "type": "paraphrase" if col.startswith("paraphrase_") else col,
                    "paraphrase_id": f"{int(col.split('_')[-1])}.0" if col.startswith("paraphrase_") else "",
                    "text": value,
                }
                records.append(entry)
        return path, records, ["row_id", "concept", "type", "paraphrase_id", "text"]

    if name == "tatum":
        path = ROOT / "tatum/data/tatum_embedding_texts.csv"
        rows = _csv(path)
        expected = {"row_id", "text_id", "type", "source_id", "text"}
        for i, row in enumerate(rows):
            if not expected.issubset(row):
                raise ValueError(f"Tatum missing columns: {expected - set(row)}")
            if int(row["row_id"]) != i:
                raise ValueError(f"Unexpected Tatum row_id at position {i}")
            for col in ("text_id", "source_id", "text"):
                _nonempty(row, col)
            if row["type"] not in {"fragment", "paragraph", "caption"}:
                raise ValueError(f"Invalid Tatum type: {row['type']}")
        if len({row["text_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate Tatum text_id")
        return path, rows, ["row_id", "text_id", "type", "source_id", "text"]

    if name == "natural":
        path = ROOT / "natural_language_words.csv"
        rows = _csv(path)
        for row in rows:
            row["text"] = _nonempty(row, "word")
        return path, rows, list(rows[0])

    if name == "external":
        path = ROOT / "external_domain_definitions.csv"
        rows = _csv(path)
        records = []
        for idx, row in enumerate(rows):
            for col in ("domain", "concept", "definition", "source_name", "source_url"):
                _nonempty(row, col)
            for col in ("concept", "definition"):
                records.append({
                    "external_row_id": idx,
                    "domain": row["domain"],
                    "concept": row["concept"],
                    "source_name": row["source_name"],
                    "source_url": row["source_url"],
                    "type": col,
                    "text": row[col].strip(),
                })
        return path, records, list(records[0])

    raise ValueError(f"Unknown dataset: {name}")


def write_metadata(path: Path, records, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
