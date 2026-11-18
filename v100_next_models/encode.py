#!/usr/bin/env python3
"""MAG V100 embedding, preserving SCORE/Tatum row order and model-specific pooling.

From concept_geo root:
  python -m v100_next_models.encode --dry-run
  python -m v100_next_models.encode --model medembed --smoke
  python -m v100_next_models.encode --model medembed --full --require-smoke

No models are fine-tuned. Saved arrays are float32, L2-normalized, native D.
Never publishes textbook text, and never modifies existing Qwen result files.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from .config import MODELS, model_dir
from .io import OUT, SCORE, TATUM, sha256, load_inputs, write_public_metadata, check_embeddings, save_npz_atomic, read_json, write_json_atomic


class Encoder:
    def __init__(self, name, batch_size=None, max_length=None):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('NVIDIA GPU required; dry-run works on CPU')
        self.name = name
        self.spec = spec = MODELS[name]
        self.batch_size = batch_size or spec.batch_size
        self.max_length = max_length or spec.max_length
        self.pooling = spec.backend
        self.path = model_dir(name)
        self.dtype = torch.float32 if spec.dtype == 'float32' else torch.float16
        if not self.path.is_dir():
            raise FileNotFoundError(f'Missing local weights: {self.path}. Run download.py')
        manifest = self.path / 'mag_download_manifest.json'
        if not manifest.exists():
            raise FileNotFoundError(f'Missing pinned download manifest: {manifest}')
        self.download_manifest = read_json(manifest)
        if self.download_manifest['repo_id'] != spec.repo_id:
            raise RuntimeError('Downloaded repo differs from selected model')
        print(f'GPU: {torch.cuda.get_device_name(0)}; {name}; {spec.dtype}; batch={self.batch_size}; max_len={self.max_length}', flush=True)
        self._init_model()

    def _init_model(self):
        import torch
        spec = self.spec
        if spec.backend == 'sapbert_cls':
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            self.model = AutoModel.from_pretrained(
                str(self.path), local_files_only=True, torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            self.model.to('cuda').eval()
            self.pooling = 'CLS (first last_hidden_state token, SapBERT standard)'
        elif spec.backend == 'nv_embed_custom':
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            config = AutoConfig.from_pretrained(
                str(self.path), trust_remote_code=True, local_files_only=True
            )
            config.text_config._name_or_path = str(self.path.resolve())
            # The author's encode() implements learned latent-attention pooling.
            # Trust remote code only for this explicitly selected/pinned repo.
            self.model = AutoModel.from_pretrained(
                str(self.path), config=config, trust_remote_code=True, local_files_only=True,
                torch_dtype=self.dtype, low_cpu_mem_usage=True,
                attn_implementation='eager',
            )
            self.model.to('cuda').eval()
            self.pooling = 'NV-Embed-v2 official model.encode(), latent-attention; instruction=""'
        elif spec.backend == 'sentence_transformer':
            from sentence_transformers import SentenceTransformer
            # dtype at load time avoids loading 8B FP32 onto a 32GB V100.
            self.model = SentenceTransformer(
                str(self.path), device='cuda',
                trust_remote_code=spec.trust_remote_code,
                model_kwargs={'torch_dtype': self.dtype, 'attn_implementation': 'eager'},
                local_files_only=True,
            )
            self.model.max_seq_length = min(self.max_length, self.model.max_seq_length or self.max_length)
            self.pooling = 'checkpoint SentenceTransformer pooling; explicit prompt=""'
            self.tokenizer = getattr(self.model, 'tokenizer', None)
        elif spec.backend == 'qwen3_vl_official':
            official = Path('third_party/Qwen3-VL-Embedding')
            if not (official / 'src/models/qwen3_vl_embedding.py').is_file():
                raise RuntimeError(
                    'For MedEmb install/review official Qwen3-VL-Embedding implementation at '
                    'third_party/Qwen3-VL-Embedding (see README). Do not substitute generic last-token pooling.'
                )
            sys.path.insert(0, str(official.resolve()))
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
            self.model = Qwen3VLEmbedder(
                model_name_or_path=str(self.path.resolve()),
                max_length=self.max_length,
                torch_dtype=self.dtype,
                attn_implementation='eager',  # V100 cannot use FA2
            )
            self.pooling = 'official Qwen3-VL-Embedding Qwen3VLEmbedder.process; text-only; default instruction'
            self.tokenizer = getattr(self.model.processor, 'tokenizer', None)
        else:
            raise ValueError(spec.backend)

    def encode(self, texts):
        import torch
        results = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = list(texts[i:i+self.batch_size])
            with torch.inference_mode():
                if self.spec.backend == 'sapbert_cls':
                    tokens = self.tokenizer(
                        batch, padding=True, truncation=True,
                        max_length=self.max_length, return_tensors='pt'
                    ).to('cuda')
                    y = self.model(**tokens).last_hidden_state[:, 0]
                elif self.spec.backend == 'nv_embed_custom':
                    y = self.model.encode(batch, instruction='', max_length=self.max_length)
                elif self.spec.backend == 'sentence_transformer':
                    # Suppress any model default query prompt: original Qwen experiment
                    # is symmetric and instruction-free. Record this deliberate choice.
                    payload = batch
                    if self.name == 'nvembed':
                        raise AssertionError('NV uses its own model.encode')
                    y = self.model.encode(
                        payload, batch_size=self.batch_size, prompt='',
                        normalize_embeddings=False, convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                else:
                    y = self.model.process([{'text': t} for t in batch])
            if isinstance(y, torch.Tensor):
                y = y.detach().float().cpu().numpy()
            y = np.asarray(y, dtype=np.float32)
            if y.ndim == 1:
                y = y[None, :]
            if y.ndim != 2 or len(y) != len(batch):
                raise ValueError(f'Model returned shape {y.shape} for {len(batch)} texts')
            results.append(check_embeddings(y, len(batch)))
            print(f'  {self.name}: {min(i+self.batch_size, total)}/{total}', flush=True)
        return check_embeddings(np.concatenate(results, axis=0), total)

    def audit_token_lengths(self, texts, output_dir, dataset):
        """Best-effort truncation audit (tokenizer only, never saves source text).

        For custom multimodal templates, these are estimates only: extra EOS and
        instruction tokens may be inserted inside the author's wrapper.
        """
        tok = getattr(self, 'tokenizer', None)
        if tok is None:
            print(f'  TOKEN AUDIT unavailable for backend={self.spec.backend}', flush=True)
            return {'available': False}
        lengths = []
        try:
            for start in range(0, len(texts), 64):
                subset = texts[start:start+64]
                encoded = tok(subset, truncation=False, add_special_tokens=True,
                              return_attention_mask=False)
                lengths.extend(len(x) for x in encoded['input_ids'])
        except Exception as exc:
            # Non-standard tokenizer/template must not prevent actual model
            # inference; mark token counts unavailable instead of guessing.
            print(f'  TOKEN AUDIT skipped ({type(exc).__name__}: {exc})', flush=True)
            return {'available': False, 'error_type': type(exc).__name__}
        import pandas as pd
        maxlen = int(getattr(self.model, 'max_seq_length', 0) or self.max_length)
        maxlen = min(maxlen, self.max_length)
        frame = pd.DataFrame({
            'row_index': list(range(len(texts))), 'token_length_estimate': lengths,
            'may_be_truncated': [int(n > maxlen) for n in lengths],
        })
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / f'token_lengths_{dataset}.csv', index=False)
        nclip = int(frame['may_be_truncated'].sum())
        print(f'  TOKEN AUDIT: {nclip}/{len(texts)} exceed {maxlen} tokens', flush=True)
        return {'available': True, 'max_length_effective': maxlen,
                'truncated_estimate': nclip, 'max_tokens_observed': int(max(lengths)),
                'note': 'estimated from tokenizer, custom templates may add tokens'}

    def cleanup(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def dependency_versions():
    import importlib.metadata
    result = {}
    for pkg in ('torch', 'transformers', 'sentence-transformers', 'huggingface_hub', 'numpy', 'pandas'):
        try:
            result[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            result[pkg] = None
    return result


def smoke_indices():
    # SCORE: concept, definition, paraphrase from multiple records.
    # Tatum: first/last fragment; first/last paragraph; first/last caption.
    return {'score': [0, 1, 2, 9, 10, 11], 'tatum': [0, 251, 252, 448, 449, 596]}


def common_info(enc, mode):
    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'mode': mode, 'model_name': enc.name, 'repo_id': enc.spec.repo_id,
        'revision': enc.download_manifest['revision'],
        'backend': enc.spec.backend, 'pooling': enc.pooling,
        'dtype_inference': enc.spec.dtype, 'dtype_saved': 'float32',
        'l2_normalized': True, 'batch_size': enc.batch_size,
        'max_length_requested': enc.max_length,
        'max_length_effective': getattr(enc.model, 'max_seq_length', enc.max_length),
        "instruction": "Represent the user's input." if enc.spec.backend == "qwen3_vl_official" else None, 'sentence_transformer_prompt_override': '' if enc.spec.backend == 'sentence_transformer' else None,
        'score_sha256': sha256(SCORE), 'tatum_sha256': sha256(TATUM),
        'versions': dependency_versions(),
        'experiment_note': ('raw, no centering, symmetric text geometry; MedEmb uses official default instruction' if enc.spec.backend == 'qwen3_vl_official' else 'raw, no centering, symmetric text geometry; default query prompts intentionally disabled'),
    }


def run_one(name, mode, batch_size, max_length, require_smoke, force, allow_experimental):
    spec = MODELS[name]
    if spec.experimental and not allow_experimental:
        raise RuntimeError(f'{name}: experimental MedEmb needs --allow-experimental (after smoke validation)')
    tables = load_inputs()
    if mode == 'dry-run':
        print('Validated:', len(tables['score']), 'SCORE texts;', len(tables['tatum']), 'Tatum texts')
        print('Inputs SHA256:', sha256(SCORE), sha256(TATUM))
        print('Model:', name, spec)
        return
    if mode == 'full' and require_smoke:
        p = OUT / 'smoke' / name / 'smoke_manifest.json'
        if not p.exists():
            raise RuntimeError(f'No passing smoke test for {name}: {p}')
        smoke = read_json(p)
        if smoke['score_sha256'] != sha256(SCORE) or smoke['tatum_sha256'] != sha256(TATUM):
            raise RuntimeError('CSV files changed after smoke test')
        if smoke.get('revision') != read_json(model_dir(name)/'mag_download_manifest.json')['revision']:
            raise RuntimeError('Model checkpoint changed since smoke test')

    base = OUT / ('smoke' if mode == 'smoke' else 'models') / name
    dest_manifest = base / ('smoke_manifest.json' if mode == 'smoke' else 'manifest.json')
    if dest_manifest.exists() and not force:
        old = read_json(dest_manifest)
        if (old.get('score_sha256') == sha256(SCORE)
                and old.get('tatum_sha256') == sha256(TATUM)
                and old.get('revision') == read_json(model_dir(name)/'mag_download_manifest.json')['revision']):
            expected = [base / f'{ds}_{name}.npz' for ds in ('score','tatum')]
            if all(p.is_file() for p in expected):
                for ds, p in zip(('score','tatum'), expected):
                    n = old['outputs'][ds]['shape'][0]
                    with np.load(p) as z:
                        assert z['embeddings_native'].shape[0] == n
                        assert np.isfinite(z['embeddings_native']).all()
                print('EXISTING VALID RESULT; not overwriting:', base)
                return
        raise RuntimeError(f'Existing results cannot be overwritten without --force: {base}')

    enc = Encoder(name, batch_size, max_length)
    manifest = common_info(enc, mode)
    manifest['outputs'] = {}
    chosen = smoke_indices() if mode == 'smoke' else None
    try:
        for ds in ('score','tatum'):
            frame = tables[ds]
            rows = chosen[ds] if chosen is not None else list(range(len(frame)))
            texts = frame.iloc[rows]['text'].astype(str).tolist()
            manifest.setdefault('token_audit', {})[ds] = enc.audit_token_lengths(texts, base, ds)
            embeddings = enc.encode(texts)
            dest = base / f'{ds}_{name}.npz'
            if dest.exists() and not force:
                raise RuntimeError(f'Output exists without --force: {dest}')
            save_npz_atomic(dest, embeddings)
            if mode == 'full':
                write_public_metadata(ds, frame)
            manifest['outputs'][ds] = {'file': str(dest), 'shape': list(embeddings.shape), 'row_indices': rows if chosen is not None else 'all'}
            print('SAVED', dest, embeddings.shape, flush=True)
        # Extra numerical smoke sanity: identical input reproducibility.
        if mode == 'smoke':
            identical = enc.encode(['The EEG contains temporal sharp waves.'] * 2)
            if not np.allclose(identical[0], identical[1], atol=1e-4):
                raise RuntimeError('Identical text produced inconsistent embedding')
        write_json_atomic(dest_manifest, manifest)
        print(f'{mode.upper()} VALIDATION OK: {name}', flush=True)
    finally:
        enc.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), default='medembed')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--full', action='store_true')
    parser.add_argument('--require-smoke', action='store_true')
    parser.add_argument('--allow-experimental', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-length', type=int, default=None)
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be >=1')
    if args.max_length is not None and args.max_length < 32:
        parser.error('--max-length must be >=32')
    m = 'dry-run' if args.dry_run else ('smoke' if args.smoke else 'full')
    run_one(args.model, m, args.batch_size, args.max_length, args.require_smoke, args.force, args.allow_experimental)


if __name__ == '__main__':
    main()
