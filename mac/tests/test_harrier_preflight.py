"""Run without MLX: python3 -m unittest mac.tests.test_harrier_preflight"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from mac.harrier.preflight import CONFIG_PATH, check_config, check_score


class TestHarrierPreflight(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_model_contract(self):
        check_config(self.cfg)

    def test_wrong_attention_rejected(self):
        bad = {**self.cfg, "attention": "bidirectional"}
        with self.assertRaises(AssertionError):
            check_config(bad)

    def test_wrong_pooling_rejected(self):
        bad = {**self.cfg, "pooling": "mean"}
        with self.assertRaises(AssertionError):
            check_config(bad)

    def test_score_identity(self):
        self.assertEqual(check_score(self.cfg)["rows"], 1827)


if __name__ == "__main__":
    unittest.main()
