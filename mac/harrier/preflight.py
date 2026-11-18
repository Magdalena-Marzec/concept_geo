"""Standard-library-only Harrier input/contract audit. No MLX or weights needed.

Run: python3 -m mac.harrier.preflight [--require-tatum]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mac.data import ROOT, load_dataset, sha256

CONFIG_PATH = Path(__file__).with_name("config.json")


def check_config(cfg: dict) -> None:
    """Refuse accidentally using generative or mean-pooled model settings."""
    assert cfg["kind"] == "dedicated_text_embedding"
    assert cfg["architecture"] == "gemma3_text"
    assert cfg["attention"] == "causal", "Harrier requires causal attention"
    assert cfg["pooling"] == "last_nonpadding_token", "Harrier is NOT mean-pooled"
    assert cfg["normalization"] == "l2"
    assert cfg["dimension"] == 5376
    assert cfg["quantization"] == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert len(cfg["mlx_revision"]) == 40 and all(c in "0123456789abcdef" for c in cfg["mlx_revision"])
    assert cfg["truncation"] == "error"


def check_score(cfg: dict) -> dict:
    path, rows, _ = load_dataset("score")  # existing loader, existing row order
    assert len(rows) == cfg["expected_rows"]["score"], "Unexpected SCORE row count"
    assert len(rows) % 9 == 0
    assert len(rows) // 9 == cfg["expected_score_concepts"]
    assert sha256(path) == cfg["score_sha256"], "SCORE version differs; STOP and version explicitly"
    for i in range(0, len(rows), 9):
        block = rows[i:i + 9]
        assert [int(r["row_id"]) for r in block] == [i // 9] * 9
        assert len({r["concept"] for r in block}) == 1
        assert [r["type"] for r in block] == ["concept", "definition"] + ["paraphrase"] * 7
        assert [r["paraphrase_id"] for r in block] == ["", ""] + [f"{j}.0" for j in range(1, 8)]
    return {"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(rows), "concepts": len(rows) // 9}


def check_tatum(cfg: dict) -> dict:
    path, rows, _ = load_dataset("tatum")
    assert len(rows) == cfg["expected_rows"]["tatum"], "Unexpected Tatum row count"
    counts = dict(Counter(r["type"] for r in rows))
    assert counts == cfg["expected_tatum_types"], "Unexpected Tatum type distribution"
    # Accept LF/CRLF serialization of the same verified CSV.
    # Never rewrite the source file.
    import hashlib
    original = path.read_bytes()
    lf = original.replace(b"\r\n", b"\n")
    assert b"\r" not in lf, "Unexpected standalone CR in Tatum CSV"
    canonical_crlf = lf.replace(b"\n", b"\r\n")
    assert hashlib.sha256(canonical_crlf).hexdigest() == cfg["tatum_sha256"], (
        "Tatum content differs beyond newline encoding; STOP and version explicitly"
    )
    return {"file": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(rows), "types": counts}


def run(require_tatum: bool = False) -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    check_config(cfg)
    result = {"experiment": cfg["experiment"], "status": "INPUTS_CHECKED_ONLY", "score": check_score(cfg)}
    try:
        result["tatum"] = check_tatum(cfg)
    except FileNotFoundError:
        if require_tatum:
            raise
        result["tatum"] = {"status": "MISSING_LOCALLY"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tatum", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.require_tatum), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
