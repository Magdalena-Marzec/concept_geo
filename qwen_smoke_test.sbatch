#!/bin/bash -l
#SBATCH --job-name=qwen_test
#SBATCH --partition=gpu
#SBATCH --account=g104-2594
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=qwen_test-%j.out
#SBATCH --error=qwen_test-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module use /apps/modulefiles/common/common
module load anaconda/23.3.1

source .venv/bin/activate

python - <<'PY'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = "./models/Qwen3-Embedding-0.6B"

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])

    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]

    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths
    ]

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    fix_mistral_regex=True
)

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = model.to("cuda")
model.eval()

texts = [
    "Spike",
    "A transient, clearly distinguishable from background activity, with pointed peak at conventional paper speeds and duration from 20 to under 70 ms.",
    "A sharply contoured transient lasting between 20 and less than 70 milliseconds."
]

print("\nTexts:", len(texts))

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=8192,
    return_tensors="pt"
)

batch = {k: v.to("cuda") for k, v in batch.items()}

with torch.inference_mode():
    outputs = model(**batch)

embeddings = last_token_pool(
    outputs.last_hidden_state,
    batch["attention_mask"]
)

embeddings = F.normalize(embeddings, p=2, dim=1)

print("Embedding shape:", tuple(embeddings.shape))
print("dtype:", embeddings.dtype)
print("device:", embeddings.device)

norms = embeddings.norm(dim=1)
print("Norms:", norms.cpu().tolist())

similarities = embeddings @ embeddings.T
print("\nCosine similarity matrix:")
print(similarities.cpu().numpy())

assert embeddings.shape == (3, 1024)
assert torch.isfinite(embeddings).all()
assert torch.allclose(
    norms,
    torch.ones_like(norms),
    atol=1e-5
)

print("\nQWEN SMOKE TEST OK")
PY
