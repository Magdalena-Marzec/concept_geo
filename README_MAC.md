# Qwen 27B hidden layers on Mac Studio (MLX)

These scripts are independent of the existing CUDA/SLURM pipeline. They extract **token embeddings (layer 0) and post-decoder-block states (layers 1–64, before final RMSNorm)** for the two 27B checkpoints. Both **last-token** and **mean** pooling are evaluated from **one full text forward pass**. L2-normalized vectors are written in float32. No `generate()`, chat templates, LM head or image encoder are used.

- Checkpoints: `mlx-community/Qwen3.8-27B-8bit` (first), `mlx-community/Qwen3.5-27B-8bit` (second).
- On Mac Studio M5 Max, 48 GB unified memory: single-text batches (no padding), 2048-token safety limit; stop with an error rather than silently truncating.
- These are language-model hidden states, **not** purpose-trained sentence embeddings.
- Dataset order is compatible with `embed_three_qwen.py` and `tatum/embed_tatum_qwen.py`.
- The MLX-VLM implementation details were checked against the Qwen3.5 `hidden_state_at_layer`/text backbone interface. **A real 27B end-to-end forward has not been tested on this Linux authoring environment; run smoke before full extraction.**

## One-time on your working machine (to make Tatum CSVs cloneable)

The `tatum/.gitignore` rule ignoring `source/` and `data/` is removed. **Git will not auto-track files merely because they are no longer ignored.** On your source workstation, copy patch files into the real repository, then run:

```bash
git add .gitignore tatum/.gitignore tatum/source/*.csv tatum/data/*.csv \
  setup_mac.sh run_mac.sh requirements-mac.txt README_MAC.md mac/
git status --short
git commit -m "Add Mac MLX 27B layer embeddings and Tatum inputs"
git push
```

The Tatum CSVs include extracts of the textbook; this commit makes them available to anyone who can access the repository while it remains public. Check repository visibility and access accordingly. Existing notebooks, CUDA scripts, outputs and weights are not modified or committed by these commands.

## First time on Mac Studio

```bash
git clone https://github.com/Magdalena-Marzec/concept_geo.git
cd concept_geo
bash setup_mac.sh
bash run_mac.sh 3.8 check
bash run_mac.sh 3.8 smoke
```

**Smoke downloads approximately 29.5 GB of 8-bit weights on first use**, then extracts first 2 SCORE and 2 Tatum texts into separate `smoke/` folders. Inspect the output and the manifest before running the full data.

## Run full extraction

```bash
bash run_mac.sh 3.8 all     # SCORE, Tatum, natural and external controls
bash run_mac.sh 3.5 all     # second model, separate checkpoint and outputs
```

Or individually:

```bash
bash run_mac.sh 3.8 score
bash run_mac.sh 3.8 tatum
bash run_mac.sh 3.8 controls
```

Rerun the *same* command to resume after interruption. To avoid a remote model-info check and reuse already downloaded weights, append `--offline` (first complete setup/download online). If you deliberately want to redo an output, append `--force` (deletes it).

## Output layout and analysis computer

For each dataset:

```
embedding_results/mlx_27b/qwen3_8_27b_8bit/score/
  embeddings.npy      # [N, 65, 2, 5120] float32, L2 normalized
  metadata.csv        # same text order as old SCORE script
  token_lengths.csv   # monitor possible long texts
  progress.json       # resumable checkpoint; do not edit
  manifest.json       # written only after finishing
```

Axis 1: layer 0 = lexical input embedding; 1–64 = post-block hidden state (pre-final RMSNorm). Axis 2: `[last_token, mean]`. A full SCORE file occupies about 4.9 GB; Tatum about 1.6 GB. All four datasets for both models are on the order of 14 GB, excluding model files. **Do not commit generated `.npy`**; the path is ignored.

Transfer the directory to the analysis computer using an external SSD or `rsync`; existing 8B `.npz` files remain unchanged. The array can be loaded with `np.load(path, mmap_mode="r")`, with `array[:, 32, 0, :]` representing 32nd-layer last-token embeddings.

To export a single 27B layer for the existing `.npz`-based notebooks:

```bash
python -m mac.export_layer 3.8 score 32 last_token
python -m mac.export_layer 3.8 tatum 32 last_token
```

The helper writes `embeddings_native`, preserving the leading key used by older notebooks, but **not** `embeddings_1024` (Qwen 27B is not trained for Matryoshka truncation). The notebook's filename and file-loading cell must point to the new NPZ; make no assumptions about shared vector coordinate systems with 8B. The protocol-specific ROC/AUC and histogram analyses stay local.

## Memory notes

Using only the text backbone avoids running the vision encoder, but the available MLX-VLM checkpoint may still load other weights. GPU memory is shared with macOS; ensure other memory-heavy applications are closed. Per-layer captures are pooled and evaluated immediately, not stored as per-token tensors. No speed or memory bound can be guaranteed without the Mac smoke test. If any layer fails, the error stops the run and the completed rows are retained for resumption.
