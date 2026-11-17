import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np

from v100_next_models import io
from v100_next_models.config import CORE, EXPERIMENTAL, MODELS, select_models


class IOTest(unittest.TestCase):
    def test_shapes_and_scores_are_exact(self):
        frames = io.load_inputs()
        score, tatum = frames['score'], frames['tatum']
        self.assertEqual(len(score), 203 * 9)
        self.assertEqual(score.iloc[0]['type'], 'concept')
        self.assertEqual(score.iloc[1]['type'], 'definition')
        self.assertEqual(score.iloc[2]['type'], 'paraphrase')
        self.assertEqual(score.iloc[9]['type'], 'concept')
        self.assertEqual(len(tatum), 597)
        self.assertEqual(tatum['type'].value_counts().to_dict(), {
            'fragment': 252, 'paragraph': 197, 'caption': 148,
        })
        # Compare directly with the existing Qwen3 implementation.
        # Parse the function directly so this CPU-only test does not need
        # transformers or CUDA (both imported at old module top-level).
        import ast
        import pandas as pd
        src = Path('embed_three_qwen.py').read_text()
        parsed = ast.parse(src)
        fn = next(x for x in parsed.body if isinstance(x, ast.FunctionDef)
                  and x.name == 'build_score_metadata')
        scope = {'np': np, 'pd': pd, 'TEXT_COLUMNS': io.COLS}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'embed_three_qwen.py', 'exec'), scope)
        old = scope['build_score_metadata'](pd.read_csv(io.SCORE))
        self.assertTrue(old.reset_index(drop=True).equals(score.reset_index(drop=True)))

    def test_normalize_and_atomic_npz(self):
        a = np.array([[1.0, 0.0] + [0.0]*30, [3.0, 4.0] + [0.0]*30])
        x = io.check_embeddings(a, 2)
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(x, axis=1), 1))
        with TemporaryDirectory() as d:
            p = Path(d) / 'test.npz'
            io.save_npz_atomic(p, x)
            with np.load(p) as z:
                self.assertTrue(np.array_equal(z['embeddings_native'], x))

    def test_metadata_excludes_copyrighted_tatum_text(self):
        t = io.load_inputs()['tatum']
        with TemporaryDirectory() as d, patch.object(io, 'OUT', Path(d)):
            p = io.write_public_metadata('tatum', t)
            content = p.read_text()
            self.assertNotIn('text,', content)
            self.assertNotIn(t.iloc[0]['text'][:70], content)

    def test_model_selection(self):
        self.assertEqual(select_models(None), list(CORE))
        self.assertTrue(set(EXPERIMENTAL).issubset(MODELS))
        with self.assertRaises(ValueError):
            select_models(['pubmedbert', 'pubmedbert'])


if __name__ == '__main__':
    unittest.main()
