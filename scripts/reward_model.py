# scripts/reward.py
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

# -----------------------------
# 1. Paths and setup
# -----------------------------
DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models/reward"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -----------------------------
# 2. Load processed datasets
# -----------------------------
accepted_file = os.path.join(DATA_PATH, "accepted_data.jsonl")
rejected_file = os.path.join(DATA_PATH, "rejected_data.jsonl")

if not os.path.exists(accepted_file) or not os.path.exists(rejected_file):
    raise FileNotFoundError("❌ Missing accepted_data.jsonl or rejected_data.jsonl — please run policy.py first.")

with open(accepted_file, "r", encoding="utf-8") as fa:
    accepted = [json.loads(line) for line in fa.readlines()]

with open(rejected_file, "r", encoding="utf-8") as fr:
    rejected = [json.loads(line) for line in fr.readlines()]

print(f"✅ Loaded {len(accepted)} accepted and {len(rejected)} rejected samples.")

# -----------------------------
# 3. Dataset preparation
# -----------------------------
class RewardDataset(Dataset):
    def __init__(self, accepted, rejected, tokenizer, max_length=256):
        self.data = list(zip(accepted, rejected))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        acc, rej = self.data[idx]
        prompt = acc["prompt"]

        acc_text = f"{prompt}\n{acc['response']}"
        rej_text = f"{prompt}\n{rej['response']}"

        acc_tokens = self.tokenizer(
            acc_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        rej_tokens = self.tokenizer(
            rej_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # remove batch dimension
        acc_tokens = {k: v.squeeze(0) for k, v in acc_tokens.items()}
        rej_tokens = {k: v.squeeze(0) for k, v in rej_tokens.items()}
        return acc_tokens, rej_tokens

# -----------------------------
# 4. Model setup (DistilBERT)
# -----------------------------
base_model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModel.from_pretrained(base_model_name).to(device)

# Print sizes (no hard assert to avoid crashes if HF internals vary)
print(f"🔤 Using base: {base_model_name}")
print(f"   Tokenizer vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}")
print(f"   Model vocab size: {getattr(model.config, 'vocab_size', 'N/A')}")

# Linear head to map CLS embedding → scalar reward
reward_head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
optimizer = AdamW(list(model.parameters()) + list(reward_head.parameters()), lr=1e-5)

# -----------------------------
# 5. Dataset + DataLoader
# -----------------------------
dataset = RewardDataset(accepted, rejected, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# 6. Training loop
# -----------------------------
print("🚀 Training reward model...")

for epoch in range(2):  # You can increase epochs later
    model.train()
    reward_head.train()
    total_loss = 0.0

    for acc_tokens, rej_tokens in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        acc_tokens = {k: v.to(device) for k, v in acc_tokens.items()}
        rej_tokens = {k: v.to(device) for k, v in rej_tokens.items()}

        acc_out = model(**acc_tokens).last_hidden_state[:, 0, :]
        rej_out = model(**rej_tokens).last_hidden_state[:, 0, :]

        acc_reward = reward_head(acc_out)
        rej_reward = reward_head(rej_out)

        # Ranking loss
        loss = -torch.log(torch.sigmoid(acc_reward - rej_reward)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"📘 Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

# -----------------------------
# 7. Save model checkpoint
# -----------------------------
save_path = os.path.join(MODEL_SAVE_PATH, "reward_model.pt")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "reward_head_state_dict": reward_head.state_dict(),
        "tokenizer": base_model_name
    },
    save_path
)

print(f"✅ Saved reward model at: {save_path}")

# -----------------------------
# 8. Sanity check
# -----------------------------
print("\n🧪 Running sanity check...")
text = "The product quality is very good and delivery was fast."
inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    emb = model(**inputs).last_hidden_state[:, 0, :]
    val = reward_head(emb).item()

print(f"🏁 Sanity check reward output: {val:.4f}")
