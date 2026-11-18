"""Causal Gemma3 embedding backbone for Harrier. No LM/Dense head."""

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gemma3_text import Gemma3Model

from mac.harrier.pooling import last_token_l2


class HarrierEmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Gemma3Model(config)

    def __call__(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Harrier initial backend requires batch_size=1")

        if input_ids.shape[1] == 0:
            raise ValueError("Cannot embed empty token sequence")

        # No padding: the Gemma3 backbone constructs its own causal
        # global/sliding-window masks. Do not pass a bidirectional mask.
        hidden = self.model(input_ids)

        attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        return last_token_l2(hidden, attention_mask)
