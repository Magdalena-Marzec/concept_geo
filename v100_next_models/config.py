"""Model registry for the MAG SCORE/Tatum follow-up.

Avoid using generic last-token pooling for non-Qwen models. Each backend below
calls the author's embedding implementation or the model's native pooling.
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    backend: str
    dtype: str
    batch_size: int
    max_length: int
    experimental: bool = False
    trust_remote_code: bool = False
    note: str = ""

MODELS = {
    "medembed": ModelSpec("abhinand/MedEmbed-large-v0.1", "sentence_transformer", "float32", 16, 512),
    "pubmedbert": ModelSpec("NeuML/pubmedbert-base-embeddings", "sentence_transformer", "float32", 24, 512),
    "sapbert": ModelSpec("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "sapbert_cls", "float32", 24, 512),
    "nvembed": ModelSpec(
        "nvidia/NV-Embed-v2", "nv_embed_custom", "float16", 1, 768,
        trust_remote_code=True,
        note="custom latent-attention pooling; no generic LLM pooling; HF access may require authentication",
    ),
    "nemotron_medical": ModelSpec(
        "minetta/nemotron-3-embed-8b-medical", "sentence_transformer", "float16", 1, 768,
        note="medical clustering head is NOT a full fine-tune; raw retrieval embeddings may equal base Nemotron",
    ),
    "medemb2b": ModelSpec(
        "MEDHEB/MedEmb-2B", "qwen3_vl_official", "float16", 1, 512,
        experimental=True, note="Qwen3-VL based multimodal medical encoder; dedicated smoke test required",
    ),
    "medemb8b": ModelSpec(
        "MEDHEB/MedEmb-8B", "qwen3_vl_official", "float16", 1, 384,
        experimental=True, note="~8.8B multimodal weights; memory and adapter compatibility need smoke test",
    ),
}
CORE = ("medembed", "pubmedbert", "sapbert", "nvembed", "nemotron_medical")
EXPERIMENTAL = ("medemb2b", "medemb8b")


def model_dir(name: str) -> Path:
    return Path("models") / "next_v100" / name


def select_models(names):
    requested = list(names) if names else list(CORE)
    unknown = sorted(set(requested) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Allowed: {list(MODELS)}")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate model names are not allowed")
    return requested
