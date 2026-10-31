"""Linux/CPU-friendly tests for data alignment and capture adapter logic."""
import csv
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mac.data import ROOT, load_dataset
from mac.embed_layers import capture_layer_outputs, embed_one, DIM


class InputTests(unittest.TestCase):
    def test_input_counts_and_columns(self):
        counts = {"score": 1827, "tatum": 597, "natural": 27, "external": 160}
        for name, target in counts.items():
            path, records, fields = load_dataset(name)
            self.assertTrue(path.exists())
            self.assertEqual(len(records), target)
            self.assertIn("text", fields)
            self.assertTrue(all(rec["text"].strip() for rec in records))

    def test_score_order_matches_cuda_metadata(self):
        _, rows, _ = load_dataset("score")
        path = ROOT / "embedding_results/score_embeddings_metadata.csv"
        if not path.exists():
            self.skipTest("Existing reference metadata not present")
        with path.open(newline="", encoding="utf-8") as f:
            expected = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(expected))
        for new, old in zip(rows, expected):
            for key in ("concept", "text", "type", "paraphrase_id", "row_id"):
                self.assertEqual(str(new[key]), str(old[key]), f"Mismatch at {new['concept']} / {key}")

    def test_tatum_order_matches_cuda_metadata(self):
        _, rows, _ = load_dataset("tatum")
        path = ROOT / "embedding_results/tatum/tatum_embeddings_metadata.csv"
        if not path.exists():
            self.skipTest("Existing reference metadata not present")
        with path.open(newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(old))
        for new, previous in zip(rows, old):
            for key in ("row_id", "text_id", "source_id", "type"):
                self.assertEqual(new[key], previous[key])


class DummyMLX:
    float32 = np.float32
    int32 = np.int32

    @staticmethod
    def array(x, dtype=None):
        return np.array(x, dtype=dtype)

    @staticmethod
    def mean(x, axis):
        return np.mean(x, axis=axis)

    @staticmethod
    def eval(*_):
        pass


class Layer:
    def __init__(self, number):
        self.number = number
        self.is_linear = number % 4 != 0

    def __call__(self, x, **_kwargs):
        # Add token-dependent perturbation to distinguish last from mean pooling.
        return x + (self.number / 100) * np.arange(x.shape[1], dtype=np.float32)[None, :, None]


class Backbone:
    def __init__(self):
        self.layers = [Layer(i) for i in range(1, 65)]

    def embed_tokens(self, tokens):
        basis = np.arange(DIM, dtype=np.float32)[None, None, :]
        return basis / (DIM + 2) + tokens[:, :, None] / 200

    def __call__(self, inputs, inputs_embeds, position_ids, stop_after_layer, apply_final_norm):
        assert position_ids is None
        assert stop_after_layer == 64
        assert apply_final_norm is False
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return x


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [3, 5, 8]}


class DummyModel:
    def __init__(self):
        self.language_model = SimpleNamespace(model=Backbone())

    def get_input_embeddings(self, input_ids):
        return SimpleNamespace(inputs_embeds=self.language_model.model.embed_tokens(input_ids), position_ids=None)


class CaptureTests(unittest.TestCase):
    def test_single_forward_captures_all_65_states_both_poolings(self):
        original_forward = Layer.__call__
        vec, length = embed_one(DummyModel(), DummyTokenizer(), "example", 20, DummyMLX)
        self.assertIs(Layer.__call__, original_forward, "Temporary hook was not restored")
        self.assertEqual(length, 3)
        self.assertEqual(vec.shape, (65, 2, DIM))
        self.assertTrue(np.allclose(np.linalg.norm(vec, axis=-1), 1.0, atol=1e-5))
        self.assertGreater(float(np.linalg.norm(vec[64, 0] - vec[64, 1])), 0.001)

    def test_too_long_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "refusing silent truncation"):
            embed_one(DummyModel(), DummyTokenizer(), "test", 2, DummyMLX)


if __name__ == "__main__":
    unittest.main()
