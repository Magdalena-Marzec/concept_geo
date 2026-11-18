import unittest

import mlx.core as mx

from mac.harrier.pooling import last_token_l2


class TestHarrierPooling(unittest.TestCase):

    def test_last_token_without_padding(self):
        hidden = mx.array([[[3, 4], [5, 12]]], dtype=mx.float32)
        mask = mx.array([[1, 1]])

        result = last_token_l2(hidden, mask)

        self.assertEqual(result.shape, (1, 2))
        self.assertAlmostEqual(result[0, 0].item(), 5 / 13, places=6)
        self.assertAlmostEqual(result[0, 1].item(), 12 / 13, places=6)

    def test_ignores_right_padding(self):
        hidden = mx.array([[[3, 4], [5, 12], [999, 999]]])
        mask = mx.array([[1, 1, 0]])

        result = last_token_l2(hidden, mask)

        self.assertAlmostEqual(result[0, 0].item(), 5 / 13, places=6)

    def test_rejects_empty_sequence(self):
        hidden = mx.zeros((1, 3, 2))
        mask = mx.array([[0, 0, 0]])

        with self.assertRaises(ValueError):
            last_token_l2(hidden, mask)

    def test_rejects_wrong_mask_shape(self):
        hidden = mx.zeros((1, 3, 2))
        mask = mx.array([[1, 1]])

        with self.assertRaises(ValueError):
            last_token_l2(hidden, mask)
