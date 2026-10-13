#!/usr/bin/env python3

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


SCORE_CSV = Path("score_concepts_completed.csv")
NATURAL_CSV = Path("natural_language_words.csv")
EXTERNAL_CSV = Path("external_domain_definitions.csv")
OUTPUT_DIR = Path("embedding_results")

MAX_LENGTH = 8192
COMMON_DIM = 1024

TEXT_COLUMNS = [
    "concept",
    "definition",
    "paraphrase_1",
    "paraphrase_2",
    "paraphrase_3",
    "paraphrase_4",
    "paraphrase_5",
    "paraphrase_6",
    "paraphrase_7",
]

MODEL_SPECS = {
    "0.6B": {
        "repo_id": "Qwen/Qwen3-Embedding-0.6B",
        "local_dir": Path("models/Qwen3-Embedding-0.6B"),
        "native_dim": 1024,
        "batch_size": 16,
    },
    "4B": {
        "repo_id": "Qwen/Qwen3-Embedding-4B",
        "local_dir": Path("models/Qwen3-Embedding-4B"),
        "native_dim": 2560,
        "batch_size": 8,
    },
    "8B": {
        "repo_id": "Qwen/Qwen3-Embedding-8B",
        "local_dir": Path("models/Qwen3-Embedding-8B"),
        "native_dim": 4096,
        "batch_size": 4,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (
        attention_mask[:, -1].sum()
        == attention_mask.shape[0]
    )
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def build_score_metadata(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [c for c in TEXT_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing SCORE columns: {missing_columns}")

    missing = int(df[TEXT_COLUMNS].isna().sum().sum())
    if missing:
        raise ValueError(
            f"SCORE CSV contains {missing} missing cells in embedding columns."
        )

    records = []

    for row_idx, row in df.iterrows():
        concept = str(row["concept"]).strip()

        records.append({
            "row_id": row_idx,
            "concept": concept,
            "type": "concept",
            "paraphrase_id": np.nan,
            "text": concept,
        })

        records.append({
            "row_id": row_idx,
            "concept": concept,
            "type": "definition",
            "paraphrase_id": np.nan,
            "text": str(row["definition"]).strip(),
        })

        for i in range(1, 8):
            records.append({
                "row_id": row_idx,
                "concept": concept,
                "type": "paraphrase",
                "paraphrase_id": i,
                "text": str(row[f"paraphrase_{i}"]).strip(),
            })

    return pd.DataFrame(records)


def build_external_metadata(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "domain",
        "concept",
        "definition",
        "source_name",
        "source_url",
    ]

    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing external-definition columns: {missing_columns}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "external_domain_definitions.csv contains missing values."
        )

    records = []

    for row_idx, row in df.iterrows():
        common = {
            "external_row_id": row_idx,
            "domain": str(row["domain"]).strip(),
            "concept": str(row["concept"]).strip(),
            "source_name": str(row["source_name"]).strip(),
            "source_url": str(row["source_url"]).strip(),
        }

        records.append({
            **common,
            "type": "concept",
            "text": str(row["concept"]).strip(),
        })

        records.append({
            **common,
            "type": "definition",
            "text": str(row["definition"]).strip(),
        })

    return pd.DataFrame(records)


def encode_texts(
    model,
    tokenizer,
    texts,
    batch_size: int,
    expected_native_dim: int,
    device: torch.device,
):
    native_chunks = []
    common_chunks = []
    n = len(texts)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        encoded = tokenizer(
            texts[start:end],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded, use_cache=False)

            pooled = last_token_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            ).float()

            if pooled.shape[1] != expected_native_dim:
                raise RuntimeError(
                    f"Unexpected embedding dimension {pooled.shape[1]} "
                    f"(expected {expected_native_dim})."
                )

            native = F.normalize(pooled, p=2, dim=1)

            common = F.normalize(
                pooled[:, :COMMON_DIM],
                p=2,
                dim=1,
            )

        native_chunks.append(native.cpu().numpy().astype(np.float32))
        common_chunks.append(common.cpu().numpy().astype(np.float32))

        print(f"  embedded {end}/{n}", flush=True)

    return np.vstack(native_chunks), np.vstack(common_chunks)


def load_model(spec, device):
    model_dir = spec["local_dir"]

    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        padding_side="left",
        fix_mistral_regex=True,
    )

    # Tesla V100 supports FP16 but not native BF16.
    # Use the same inference dtype for all three model sizes.
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return tokenizer, model


def save_npz(path, native, common):
    np.savez_compressed(
        path,
        embeddings_native=native,
        embeddings_1024=common,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["0.6B", "4B", "8B"],
        choices=list(MODEL_SPECS),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script.")

    device = torch.device("cuda")

    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Inference dtype: float16")
    print("Saved embedding dtype: float32")
    print("No query/passage instruction is added.")
    print()

    if not SCORE_CSV.exists():
        raise FileNotFoundError(SCORE_CSV)
    if not NATURAL_CSV.exists():
        raise FileNotFoundError(NATURAL_CSV)
    if not EXTERNAL_CSV.exists():
        raise FileNotFoundError(EXTERNAL_CSV)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    score_df = pd.read_csv(SCORE_CSV)
    score_meta = build_score_metadata(score_df)

    natural_meta = pd.read_csv(NATURAL_CSV)
    external_df = pd.read_csv(EXTERNAL_CSV)
    external_meta = build_external_metadata(external_df)

    if "word" not in natural_meta.columns:
        raise ValueError("natural_language_words.csv needs a 'word' column.")
    if natural_meta["word"].isna().any():
        raise ValueError("Missing word in natural_language_words.csv.")

    score_texts = score_meta["text"].tolist()
    natural_texts = natural_meta["word"].astype(str).tolist()
    external_texts = external_meta["text"].tolist()

    print("SCORE concepts:", len(score_df))
    print("SCORE texts:", len(score_texts))
    print("Natural-language words:", len(natural_texts))
    print("External-domain entries:", len(external_df))
    print("External-domain texts:", len(external_texts))
    print(
        "External domains:",
        external_df["domain"].value_counts().to_dict()
    )

    duplicated = score_df["concept"].duplicated(keep=False)
    if duplicated.any():
        print(
            "WARNING: duplicate concept labels:",
            score_df.loc[duplicated, "concept"].tolist(),
        )

    score_meta.to_csv(
        OUTPUT_DIR / "score_embeddings_metadata.csv",
        index=False,
    )
    natural_meta.to_csv(
        OUTPUT_DIR / "natural_language_embeddings_metadata.csv",
        index=False,
    )
    external_meta.to_csv(
        OUTPUT_DIR / "external_domain_embeddings_metadata.csv",
        index=False,
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_csv": str(SCORE_CSV),
        "score_csv_sha256": sha256(SCORE_CSV),
        "natural_csv": str(NATURAL_CSV),
        "natural_csv_sha256": sha256(NATURAL_CSV),
        "external_csv": str(EXTERNAL_CSV),
        "external_csv_sha256": sha256(EXTERNAL_CSV),
        "score_concepts": int(len(score_df)),
        "score_texts": int(len(score_texts)),
        "natural_words": int(len(natural_texts)),
        "external_entries": int(len(external_df)),
        "external_texts": int(len(external_texts)),
        "external_domains": external_df["domain"].value_counts().to_dict(),
        "text_columns": TEXT_COLUMNS,
        "pooling": "last non-padding token",
        "normalization": "L2",
        "instruction": None,
        "max_length": MAX_LENGTH,
        "inference_dtype": "float16",
        "saved_dtype": "float32",
        "common_mrl_dimension": COMMON_DIM,
        "models": {},
    }

    for model_size in args.models:
        spec = MODEL_SPECS[model_size]
        slug = model_size.lower().replace(".", "_")

        print()
        print("=" * 72)
        print(spec["repo_id"])
        print("=" * 72)

        tokenizer, model = load_model(spec, device)

        score_native, score_common = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=score_texts,
            batch_size=spec["batch_size"],
            expected_native_dim=spec["native_dim"],
            device=device,
        )

        natural_native, natural_common = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=natural_texts,
            batch_size=spec["batch_size"],
            expected_native_dim=spec["native_dim"],
            device=device,
        )

        external_native, external_common = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=external_texts,
            batch_size=spec["batch_size"],
            expected_native_dim=spec["native_dim"],
            device=device,
        )

        score_out = OUTPUT_DIR / f"score_qwen3_{slug}.npz"
        natural_out = OUTPUT_DIR / f"natural_qwen3_{slug}.npz"
        external_out = OUTPUT_DIR / f"external_qwen3_{slug}.npz"

        save_npz(score_out, score_native, score_common)
        save_npz(natural_out, natural_native, natural_common)
        save_npz(external_out, external_native, external_common)

        for name, arr in [
            ("score_native", score_native),
            ("score_1024", score_common),
            ("natural_native", natural_native),
            ("natural_1024", natural_common),
            ("external_native", external_native),
            ("external_1024", external_common),
        ]:
            norms = np.linalg.norm(arr, axis=1)

            if not np.isfinite(arr).all():
                raise RuntimeError(f"{name}: NaN/Inf detected.")

            if not np.allclose(norms, 1.0, atol=1e-5):
                raise RuntimeError(
                    f"{name}: embeddings are not L2-normalized."
                )

        manifest["models"][model_size] = {
            "repo_id": spec["repo_id"],
            "local_dir": str(spec["local_dir"]),
            "native_dim": spec["native_dim"],
            "batch_size": spec["batch_size"],
            "score_file": str(score_out),
            "natural_file": str(natural_out),
            "external_file": str(external_out),
            "score_shape_native": list(score_native.shape),
            "score_shape_1024": list(score_common.shape),
            "natural_shape_native": list(natural_native.shape),
            "natural_shape_1024": list(natural_common.shape),
            "external_shape_native": list(external_native.shape),
            "external_shape_1024": list(external_common.shape),
        }

        print("Saved:", score_out)
        print("Saved:", natural_out)
        print("Saved:", external_out)
        print("SCORE native:", score_native.shape)
        print("SCORE common 1024:", score_common.shape)
        print("Natural native:", natural_native.shape)
        print("Natural common 1024:", natural_common.shape)
        print("External native:", external_native.shape)
        print("External common 1024:", external_common.shape)

        del model, tokenizer
        del score_native, score_common, natural_native, natural_common
        del external_native, external_common
        gc.collect()
        torch.cuda.empty_cache()

    manifest_path = OUTPUT_DIR / "embedding_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("ALL REQUESTED MODELS COMPLETED")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
