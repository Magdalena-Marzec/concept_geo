import unittest

import mlx.core as mx
from mlx_lm.models.gemma3_text import ModelArgs

from mac.harrier.model import HarrierEmbeddingModel


class TestHarrierModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mx.random.seed(7)
        config = ModelArgs(
            model_type="gemma3_text",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            sliding_window=8,
            sliding_window_pattern=2,
        )
        cls.model = HarrierEmbeddingModel(config)

    def test_embedding_shape_and_l2(self):
        ids = mx.array([[3, 4, 5]], dtype=mx.int32)
        result = self.model(ids)
        mx.eval(result)

        self.assertEqual(result.shape, (1, 16))
        norm_sq = mx.sum(result * result).item()
        self.assertAlmostEqual(norm_sq, 1.0, places=4)

    def test_causal_no_future_leakage(self):
        prefix = mx.array([[3, 4, 5]], dtype=mx.int32)
        longer = mx.array([[3, 4, 5, 6]], dtype=mx.int32)

        short_h = self.model.model(prefix)
        long_h = self.model.model(longer)
        difference = mx.max(mx.abs(short_h[0, 2] - long_h[0, 2]))

        self.assertLess(difference.item(), 1e-3)

    def test_rejects_batch_larger_than_one(self):
        ids = mx.array([[3, 4], [5, 6]], dtype=mx.int32)
        with self.assertRaises(ValueError):
            self.model(ids)

    def test_rejects_empty_input(self):
        ids = mx.zeros((1, 0), dtype=mx.int32)
        with self.assertRaises(ValueError):
            self.model(ids)
