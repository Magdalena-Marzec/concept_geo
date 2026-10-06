import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "./models/Qwen3-Embedding-0.6B"
INPUT_CSV = "score_concepts.csv"

OUTPUT_NPZ = "score_embeddings.npz"
OUTPUT_METADATA = "score_embeddings_metadata.csv"

BATCH_SIZE = 16
MAX_LENGTH = 8192

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


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# POOLING
# ============================================================

def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    """
    Qwen3-Embedding pooling:
    take the hidden state of the last non-padding token.
    """

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
        sequence_lengths
    ]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

print("Model loaded.")


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

missing_columns = [
    col for col in TEXT_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )


# ============================================================
# CONVERT WIDE CSV TO LONG FORMAT
# ============================================================

records = []

for row_idx, row in df.iterrows():

    concept_name = row["concept"]

    # concept name
    if pd.notna(row["concept"]) and str(row["concept"]).strip():
        records.append({
            "row_id": row_idx,
            "concept": concept_name,
            "type": "concept",
            "paraphrase_id": None,
            "text": str(row["concept"]).strip()
        })

    # definition
    if pd.notna(row["definition"]) and str(row["definition"]).strip():
        records.append({
            "row_id": row_idx,
            "concept": concept_name,
            "type": "definition",
            "paraphrase_id": None,
            "text": str(row["definition"]).strip()
        })

    # paraphrases
    for i in range(1, 8):
        col = f"paraphrase_{i}"

        if pd.notna(row[col]) and str(row[col]).strip():
            records.append({
                "row_id": row_idx,
                "concept": concept_name,
                "type": "paraphrase",
                "paraphrase_id": i,
                "text": str(row[col]).strip()
            })


long_df = pd.DataFrame(records)

print(f"Number of texts to embed: {len(long_df)}")


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def encode_texts(texts):
    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.inference_mode():

            outputs = model(**encoded)

            embeddings = last_token_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"]
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        embeddings = (
            embeddings
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        all_embeddings.append(embeddings)

        end = min(
            start + BATCH_SIZE,
            len(texts)
        )

        print(
            f"Embedded {end}/{len(texts)}"
        )

    return np.vstack(all_embeddings)


# ============================================================
# RUN
# ============================================================

texts = long_df["text"].tolist()

embeddings = encode_texts(texts)

print(f"Embedding matrix shape: {embeddings.shape}")


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.savez_compressed(
    OUTPUT_NPZ,
    embeddings=embeddings
)

long_df.to_csv(
    OUTPUT_METADATA,
    index=False
)

print()
print("Saved:")
print(f"  embeddings: {OUTPUT_NPZ}")
print(f"  metadata:   {OUTPUT_METADATA}")
print()
print(f"Shape: {embeddings.shape}")