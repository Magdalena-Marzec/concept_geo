#!/usr/bin/env python3
"""Build Tatum embedding inputs and joins from the approved final_v2 CSV tables.

Run in the root of concept_geo: python tatum/prepare_inputs.py
No model/GPU dependencies. Only standard-library Python is required.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
DATA = ROOT / "data"
REQUIRED = [
    "concept_fragments_all.csv",
    "fragment_metadata_all.csv",
    "paragraphs_all.csv",
    "figure_captions_all.csv",
]


def read_csv(name):
    path = SOURCE / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing input {path}. Copy from final_v2/ of the mapping ZIP.")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def register(values, key, text, label):
    if not key or not text or not text.strip():
        raise ValueError(f"Empty {label} ID/text: {key!r}")
    old = values.get(key)
    if old is not None and old != text:
        raise ValueError(f"Conflicting text for {label}: {key}")
    values[key] = text


def main():
    wide = read_csv(REQUIRED[0])
    meta = read_csv(REQUIRED[1])
    pars = read_csv(REQUIRED[2])
    captions = read_csv(REQUIRED[3])
    if len(wide) != 203:
        raise ValueError(f"Expected 203 SCORE concepts, got {len(wide)}")

    # The wide table has fragment_NN_id and fragment_NN_text pairs.
    all_fragments = OrderedDict()
    for row in wide:
        for field in row:
            if field.startswith("fragment_") and field.endswith("_id"):
                stem = field[:-3]
                fragment_id = row[field].strip()
                fragment_text = row.get(stem + "_text", "")
                if fragment_id:
                    register(all_fragments, fragment_id, fragment_text, "fragment")
                elif fragment_text.strip():
                    raise ValueError(f"Text without ID in {row['concept_id']}/{stem}")

    # Keep source paragraphs with accepted fragment references only.
    paragraph_lookup = OrderedDict()
    for p in pars:
        register(paragraph_lookup, p["paragraph_id"], p["paragraph_text"], "paragraph")

    # The order of the metadata table is authoritative for embedded fragments.
    fragment_ids = list(OrderedDict.fromkeys(m["fragment_id"] for m in meta))
    paragraph_ids = list(OrderedDict.fromkeys(m["paragraph_id"] for m in meta))
    if set(fragment_ids) != set(all_fragments):
        raise ValueError("Wide fragment table does not match fragment_metadata_all.csv")
    if not set(paragraph_ids).issubset(paragraph_lookup):
        raise ValueError("Missing original source paragraphs")
    if len({(m["concept_id"], m["fragment_id"]) for m in meta}) != len(meta):
        raise ValueError("Duplicate concept-fragment relation")

    caption_lookup = OrderedDict()
    for row in captions:
        register(caption_lookup, row["caption_id"], row["caption_text"], "caption")
    if len({(r["concept_id"], r["caption_id"]) for r in captions}) != len(captions):
        raise ValueError("Duplicate concept-caption relation")

    texts = []
    for kind, keys, lookup in [
        ("fragment", fragment_ids, all_fragments),
        ("paragraph", paragraph_ids, paragraph_lookup),
        ("caption", list(caption_lookup), caption_lookup),
    ]:
        for key in keys:
            text = lookup[key]
            texts.append({
                "row_id": len(texts),
                "text_id": f"{kind}:{key}",
                "type": kind,
                "source_id": key,
                "text": text,
            })

    # A paragraph can contribute multiple fragments to the same concept.
    # Its concept link is kept once, with all corresponding fragment IDs.
    par_concept = OrderedDict()
    for r in meta:
        key = (r["concept_id"], r["paragraph_id"])
        if key not in par_concept:
            par_concept[key] = {
                "concept_id": r["concept_id"], "concept": r["concept"],
                "paragraph_id": r["paragraph_id"], "fragment_ids": [],
                "match_types": [], "semantic_relations": [],
                "information_foci": [], "aspect_tags": [],
            }
        o = par_concept[key]
        o["fragment_ids"].append(r["fragment_id"])
        for src, dst in [("match_type", "match_types"),
                         ("semantic_relation", "semantic_relations"),
                         ("information_focus", "information_foci")]:
            if r[src] and r[src] not in o[dst]:
                o[dst].append(r[src])
        for t in (r.get("aspect_tags") or "").split(";"):
            t = t.strip()
            if t and t not in o["aspect_tags"]:
                o["aspect_tags"].append(t)

    par_relations = []
    for record in par_concept.values():
        par_relations.append({
            k: (";".join(v) if isinstance(v, list) else v)
            for k, v in record.items()
        })

    DATA.mkdir(parents=True, exist_ok=True)
    write_csv(DATA / "tatum_embedding_texts.csv", texts,
              ["row_id", "text_id", "type", "source_id", "text"])
    write_csv(DATA / "tatum_fragment_relations.csv", meta, list(meta[0]))
    write_csv(DATA / "tatum_paragraph_relations.csv", par_relations,
              list(par_relations[0]))
    write_csv(DATA / "tatum_caption_relations.csv", captions, list(captions[0]))

    counts = {t: sum(row["type"] == t for row in texts)
              for t in ("fragment", "paragraph", "caption")}
    print("Prepared texts:", counts, "total:", len(texts))
    print("Concept relations: fragments", len(meta), "paragraphs", len(par_relations),
          "captions", len(captions))
    print("Output (private):", DATA)


if __name__ == "__main__":
    main()
