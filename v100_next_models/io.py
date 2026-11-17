"""Input/output validation and atomic result saving, independent of GPU code."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

SCORE = Path('score_concepts_completed.csv')
TATUM = Path('tatum/data/tatum_embedding_texts.csv')
OUT = Path('embedding_results/next_v100')
COLS = ['concept', 'definition', *[f'paraphrase_{i}' for i in range(1, 8)]]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_inputs():
    if not SCORE.is_file() or not TATUM.is_file():
        raise FileNotFoundError(f'Required files: {SCORE}, {TATUM}')
    score = pd.read_csv(SCORE)  # original repository: comma-separated 203 rows
    if any(c not in score for c in COLS):
        raise ValueError(f'SCORE missing {sorted(set(COLS) - set(score.columns))}')
    if score[COLS].isna().any().any():
        raise ValueError('SCORE missing definition/name/paraphrase')
    if score['concept'].duplicated().any():
        raise ValueError('SCORE duplicate concept labels; verify the input version')
    if len(score) != 203:
        raise ValueError(f'Expected 203 SCORE concepts, got {len(score)}; do not mix ontologies')

    # Preserve EXACT order produced by embed_three_qwen.build_score_metadata.
    rows = []
    for i, row in score.iterrows():
        concept = str(row['concept']).strip()
        rows.append(dict(row_id=i, concept=concept, type='concept', paraphrase_id=np.nan, text=concept))
        rows.append(dict(row_id=i, concept=concept, type='definition', paraphrase_id=np.nan, text=str(row['definition']).strip()))
        for j in range(1, 8):
            rows.append(dict(row_id=i, concept=concept, type='paraphrase', paraphrase_id=j, text=str(row[f'paraphrase_{j}']).strip()))
    sm = pd.DataFrame(rows)
    if len(sm) != 1827 or sm['text'].eq('').any():
        raise ValueError('Unexpected SCORE texts or empty texts')

    tatum = pd.read_csv(TATUM, dtype={'text_id': str, 'type': str, 'source_id': str, 'text': str})
    required = {'row_id', 'text_id', 'type', 'source_id', 'text'}
    if not required.issubset(tatum):
        raise ValueError(f'Tatum missing: {required-set(tatum.columns)}')
    if tatum[list(required)].isna().any().any() or tatum['text'].str.strip().eq('').any():
        raise ValueError('Tatum has missing/empty fields')
    if tatum['text_id'].duplicated().any() or tatum['row_id'].tolist() != list(range(len(tatum))):
        raise ValueError('Tatum IDs not unique/consecutive')
    if len(tatum) != 597 or not tatum['type'].isin({'fragment', 'paragraph', 'caption'}).all():
        raise ValueError('Unexpected Tatum version/types; expected exactly 597')
    for kind, expected in [('fragment', 252), ('paragraph', 197), ('caption', 148)]:
        if int((tatum['type'] == kind).sum()) != expected:
            raise ValueError(f'Unexpected Tatum {kind} count')
    return {'score': sm, 'tatum': tatum}


def write_public_metadata(dataset: str, df: pd.DataFrame):
    meta_dir = OUT / 'metadata'
    meta_dir.mkdir(parents=True, exist_ok=True)
    if dataset == 'score':
        columns = ['row_id', 'concept', 'type', 'paraphrase_id', 'text']
    else:
        # Never copy copyright-protected Tatum text into public output.
        columns = ['row_id', 'text_id', 'type', 'source_id']
    dest = meta_dir / f'{dataset}_embeddings_metadata.csv'
    data = df[columns]
    if dest.exists():
        old = pd.read_csv(dest, dtype=str).fillna('')
        new = data.astype(str).replace({'nan': ''}).fillna('')
        if not old.equals(new):
            raise RuntimeError(f'Metadata mismatch at {dest}; investigate before running')
    else:
        data.to_csv(dest, index=False)
    return dest


def check_embeddings(a, n):
    x = np.asarray(a, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] < 32:
        raise ValueError(f'Bad embedding matrix {x.shape}; expected ({n},D>=32)')
    if not np.isfinite(x).all():
        raise ValueError('Non-finite embedding found')
    lengths = np.linalg.norm(x, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths < 1e-12):
        raise ValueError('Zero embedding found')
    return (x / lengths[:, None]).astype(np.float32)


def save_npz_atomic(dest: Path, x: np.ndarray):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.writing')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, embeddings_native=x)
        with np.load(tmp) as z:
            y = z['embeddings_native']
            if y.shape != x.shape or not np.isfinite(y).all():
                raise ValueError('Corrupt temporary NPZ')
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json_atomic(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)
