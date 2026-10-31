# Tatum text embeddings (add-on to `concept_geo`)

This extension uses the **existing** `embed_three_qwen.py` model specifications,
FP16 inference, last-token pooling, L2 normalization, native dimensions and
MRL-truncated 1024-dimensional embeddings. The existing SCORE embedding script
and its outputs are not modified.

## Source data (tracked with the Mac extension)

The Tatum source tables are now included in Git for convenient `git clone` on a
new computer. Both `tatum/source/` and `tatum/data/` can be committed. The
previous ignore rules were removed from `tatum/.gitignore`.

Expected files in `tatum/source/`:
- `concept_fragments_all.csv`
- `fragment_metadata_all.csv`
- `paragraphs_all.csv`
- `figure_captions_all.csv`
- `review_candidates_all.csv`
- `rejected_candidates_all.csv`

The first four are required to regenerate the derived embedding inputs; the
last two are useful for reviewing candidate matches and controls. The source
text is extracted from a copyrighted handbook; Git access to the repository
therefore also grants access to these extracts. Check access rights before
sharing the repository with others.

## Workflow

From the repo root:

```bash
python tatum/prepare_inputs.py
```

It writes private `tatum/data/` inputs:

- `tatum_embedding_texts.csv`: unique texts, numbered in embedding order;
- `tatum_fragment_relations.csv`: per-concept accepted fragment metadata;
- `tatum_paragraph_relations.csv`: per-concept unique source paragraphs;
- `tatum_caption_relations.csv`: per-concept captions and metadata.

After committing and pushing `tatum/source/*.csv` and `tatum/data/*.csv`, a
fresh `git clone` includes the Tatum tables. Running the existing CUDA job on
the cluster still works as before:

```bash
git pull
JOBID=$(sbatch --parsable run_tatum_qwen_embeddings.sbatch)
echo "$JOBID"
squeue -j "$JOBID"
```

This new job writes under `embedding_results/tatum/`, leaving all previous
SCORE, natural-language, and external-domain embedding files untouched.
Each `tatum_qwen3_*.npz` contains `embeddings_native` and `embeddings_1024`,
with rows matching `tatum_embeddings_metadata.csv`. That metadata contains **IDs
only, not quoted textbook text**. The manifest records the input SHA256
and sizes for reproducibility.

Keep generated embedding arrays out of Git unless intentionally sharing large artifacts.
The source and derived Tatum CSV files are committed for Mac/cluster input consistency.

## Comparison notebook

For a given text row:
- `type=fragment`, `source_id=F-...` joins to `fragment_id`;
- `type=paragraph`, `source_id=P...` joins to `paragraph_id`;
- `type=caption`, `source_id=FIG-...` joins to `caption_id`.

Join to `tatum_*_relations.csv` using those IDs. Keep a separate embedding row
for each unique fragment, source paragraph, and caption, even when a fragment
happens to equal the entire paragraph; filter equal-text pairs out of the
paragraph-versus-excerpt change analysis.
