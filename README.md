# Three-model Qwen3 embedding experiment

Designed for the existing `concept_geo` repository and ICM Rysy.

Models:
- Qwen/Qwen3-Embedding-0.6B
- Qwen/Qwen3-Embedding-4B
- Qwen/Qwen3-Embedding-8B

Inputs:
- `score_concepts_completed.csv`
- `natural_language_words.csv`
- `external_domain_definitions.csv`

`external_domain_definitions.csv` contains 80 source-based control definitions:
- 10 cardiology,
- 10 cardiac surgery,
- 10 orthopedics,
- 10 dermatology,
- 10 competitive rowing,
- 10 volleyball,
- 10 contract bridge,
- 10 association football.

The definitions are concise paraphrases of the cited authoritative web pages; the source name and URL are stored in the CSV.

No retrieval instruction and no `query:` / `passage:` prefix are added. The goal is symmetric semantic geometry.

## Saved spaces

Each NPZ contains:

- `embeddings_native`
- `embeddings_1024`

Native dimensions:
- 0.6B: 1024
- 4B: 2560
- 8B: 4096

The three models support MRL. The first 1024 dimensions are therefore also saved and re-normalized, giving a dimension-matched comparison.

## FP16

Tesla V100 supports FP16 but not native BF16. All three model sizes are loaded in FP16. Pooled vectors are converted to float32 before L2 normalization and saving.

## Files

- `download_qwen3_embedding_4b.py`
- `download_qwen3_embedding_8b.py`
- `download_qwen3_large_models.sbatch`
- `embed_three_qwen.py`
- `run_three_qwen_embeddings.sbatch`
- `natural_language_words.csv`
- `external_domain_definitions.csv`
- `compare_qwen3_embeddings.ipynb`

The embedding script does not hard-code the number of SCORE concepts, so final typo corrections do not require editing the code.

## Cluster workflow

After the final CSV is committed locally as:

```text
score_concepts_completed.csv
```

push it together with these scripts, then on Rysy:

```bash
cd ~/concept_geo
git pull

module use /apps/modulefiles/common/common
module load anaconda/23.3.1
source .venv/bin/activate
```

### Download 4B and 8B

```bash
JOBID=$(sbatch --parsable download_qwen3_large_models.sbatch)
echo "JOBID=$JOBID"
squeue -j "$JOBID"
```

After completion:

```bash
cat "qwen3_download-${JOBID}.out"
cat "qwen3_download-${JOBID}.err"
```

### Embed SCORE + ordinary English controls with all three models

```bash
JOBID=$(sbatch --parsable run_three_qwen_embeddings.sbatch)
echo "JOBID=$JOBID"
squeue -j "$JOBID"
```

After completion:

```bash
cat "logs/qwen3_scale_${JOBID}.log"
cat "qwen3_scale-${JOBID}.err"
```

Successful ending:

```text
FINAL VALIDATION OK
THREE-MODEL EMBEDDING JOB COMPLETED SUCCESSFULLY
```

## Outputs

```text
embedding_results/
├── embedding_manifest.json
├── score_embeddings_metadata.csv
├── natural_language_embeddings_metadata.csv
├── external_domain_embeddings_metadata.csv
├── score_qwen3_0_6b.npz
├── score_qwen3_4b.npz
├── score_qwen3_8b.npz
├── natural_qwen3_0_6b.npz
├── natural_qwen3_4b.npz
└── natural_qwen3_8b.npz
```

Keep `models/` ignored by Git.

The `embedding_results/` files can be committed so the analysis can be run locally without GPU.

## If 8B runs out of GPU memory

In `embed_three_qwen.py`, change:

```python
"batch_size": 4,
```

for the 8B model to:

```python
"batch_size": 2,
```

or `1`.


## External-domain controls

The embedding run also processes `external_domain_definitions.csv`.

For every external entry it embeds:
1. the concept label,
2. the definition.

The notebook compares SCORE with all eight external domains using:
- within-domain and between-domain mean cosine similarity,
- normalized domain-centroid similarities,
- nearest SCORE definition for every external definition,
- concept-label → definition cosine distributions,
- PCA of all definition embeddings colored by domain.

These controls are exploratory. They help distinguish EEG-specific effects from generic properties of technical definitions and embedding spaces.
