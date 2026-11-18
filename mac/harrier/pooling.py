"""Last-token pooling and L2 normalization for Harrier embeddings."""

import mlx.core as mx


def last_token_l2(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    if hidden_states.ndim != 3:
        raise ValueError("Expected hidden states [batch, tokens, dimension]")

    batch, length, _ = hidden_states.shape
    if attention_mask.shape != (batch, length):
        raise ValueError("Attention mask shape does not match hidden states")

    if bool(mx.any((attention_mask != 0) & (attention_mask != 1)).item()):
        raise ValueError("Attention mask must contain only 0 and 1")

    positions = mx.arange(length)[None, :]
    last = mx.max(
        mx.where(attention_mask == 1, positions, -1),
        axis=1,
    )

    if bool(mx.any(last < 0).item()):
        raise ValueError("Cannot embed an empty token sequence")

    pooled = hidden_states[mx.arange(batch), last].astype(mx.float32)
    squared_norm = mx.sum(pooled * pooled, axis=-1, keepdims=True)

    if bool(mx.any(squared_norm <= 0).item()):
        raise ValueError("Cannot normalize a zero-length embedding")

    return pooled / mx.sqrt(squared_norm)
