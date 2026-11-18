import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.gemma3_text import ModelArgs

from mac.harrier.model import HarrierEmbeddingModel
from mac.harrier.weights import load_strict_shards


def make_model():
    config = ModelArgs(
        model_type="gemma3_text",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=128,
        sliding_window=8,
        sliding_window_pattern=2,
    )

    model = HarrierEmbeddingModel(config)
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        mode="affine",
    )
    return model


def save_three_shards(model, directory):
    weights = dict(tree_flatten(model.parameters()))
    names = sorted(weights)

    for i in range(3):
        shard = {
            name: weights[name]
            for j, name in enumerate(names)
            if j % 3 == i
        }
        mx.save_safetensors(
            str(directory / f"model-{i+1:05d}-of-00003.safetensors"),
            shard,
        )

    return len(weights)


class TestHarrierWeights(unittest.TestCase):

    def test_strict_roundtrip_ignores_stale_index(self):
        mx.random.seed(7)

        original = make_model()
        tokens = mx.array([[3, 4, 5]], dtype=mx.int32)

        expected_embedding = original(tokens)
        mx.eval(expected_embedding)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            expected_count = save_three_shards(original, directory)

            stale_index = {
                "weight_map": {
                    "invalid.weight": "model-00001-of-00011.safetensors"
                }
            }
            (directory / "model.safetensors.index.json").write_text(
                json.dumps(stale_index)
            )

            restored = make_model()
            loaded_count = load_strict_shards(restored, directory)

            actual_embedding = restored(tokens)
            mx.eval(actual_embedding)

            self.assertEqual(loaded_count, expected_count)
            self.assertTrue(
                bool(mx.allclose(
                    expected_embedding,
                    actual_embedding,
                    atol=1e-5,
                    rtol=1e-5,
                ).item())
            )

    def test_missing_shard_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            original = make_model()
            save_three_shards(original, directory)

            (directory / "model-00002-of-00003.safetensors").unlink()

            with self.assertRaises(ValueError):
                load_strict_shards(make_model(), directory)
