# Tatum text embeddings (add-on to `concept_geo`)

This extension uses the **existing** `embed_three_qwen.py` model specifications,
FP16 inference, last-token pooling, L2 normalization, native dimensions and
MRL-truncated 1024-dimensional embeddings. The existing SCORE embedding script
and its outputs are not modified.

## Source data (private, NOT committed to public Git)

Create `tatum/source/` in your local `code/` directory and copy **exactly four**
CSV files from the `final_v2/` folder of `score_handbook_mapping_final_v2.zip`:

- `concept_fragments_all.csv`
- `fragment_metadata_all.csv`
- `paragraphs_all.csv`
- `figure_captions_all.csv`

The first three are required for fragment/paragraph embedding analysis; captions
are a separately tagged analysis category, and this runner requires the fourth file
to include them. The
`source/` and derived `data/` directories are ignored by `tatum/.gitignore`.
Do not `git add -f` their contents to a public repository.

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

Transfer the four source CSVs to the cluster via `scp` or `rsync` rather than
Git (the repository is public). A typical manual transfer of the whole private
source folder to your existing cluster checkout is:

```bash
rsync -av tatum/source/ <USER>@<HOST>:~/concept_geo/tatum/source/
```

Then on the cluster (from `~/concept_geo`):

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
only, not quoted textbook text**. The manifest records the private input SHA256
and sizes for reproducibility.

If you push embedding arrays back to Git, review repository size/LFS limits and
license/privacy implications. To reproduce analysis locally, keep your private
`tatum/source/` or `tatum/data/` alongside the ID-only embedding metadata.

## Comparison notebook

For a given text row:
- `type=fragment`, `source_id=F-...` joins to `fragment_id`;
- `type=paragraph`, `source_id=P...` joins to `paragraph_id`;
- `type=caption`, `source_id=FIG-...` joins to `caption_id`.

Join to `tatum_*_relations.csv` using those IDs. Keep a separate embedding row
for each unique fragment, source paragraph, and caption, even when a fragment
happens to equal the entire paragraph; filter equal-text pairs out of the
paragraph-versus-excerpt change analysis.
