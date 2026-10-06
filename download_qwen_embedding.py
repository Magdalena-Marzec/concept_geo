from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
LOCAL_DIR = "./models/Qwen3-Embedding-0.6B"

print(f"Downloading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(LOCAL_DIR)
model.save_pretrained(LOCAL_DIR)

print(f"Model saved to: {LOCAL_DIR}")