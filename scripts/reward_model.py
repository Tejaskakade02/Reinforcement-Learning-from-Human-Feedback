# import os
# import json
# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
# from torch.optim import AdamW

# # -----------------------------
# # 1. Paths and setup
# # -----------------------------
# POLICY_MODEL_PATH = "models/policy"
# REWARD_MODEL_PATH = "models/reward/reward_model.pt"
# DATA_PATH = "data/processed"
# OUTPUT_MODEL_PATH = "models/ppo"
# os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🔥 Using device: {device}")

# # -----------------------------
# # 2. Load models and tokenizers
# # -----------------------------
# # Load policy model (GPT-2) + tokenizer
# policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
# policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH).to(device)

# if policy_tokenizer.pad_token is None:
#     policy_tokenizer.pad_token = policy_tokenizer.eos_token

# # Load reward model (DistilBERT) + tokenizer
# reward_checkpoint = torch.load(REWARD_MODEL_PATH, map_location=device)

# # Always reload same base used in reward_model.py
# base_model_name = "distilbert-base-uncased"
# reward_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# reward_model = AutoModel.from_pretrained(base_model_name).to(device)

# # Add the small linear head
# reward_head = torch.nn.Linear(reward_model.config.hidden_size, 1).to(device)
# reward_model.load_state_dict(reward_checkpoint["model_state_dict"])
# reward_head.load_state_dict(reward_checkpoint["reward_head_state_dict"])
# reward_model.eval()
# reward_head.eval()

# optimizer = AdamW(policy_model.parameters(), lr=1e-6)

# # -----------------------------
# # 3. Load prompts
# # -----------------------------
# prompts = []
# with open(os.path.join(DATA_PATH, "accepted_data.jsonl"), "r") as f:
#     for line in f:
#         data = json.loads(line)
#         prompts.append(data["prompt"])
# print(f"✅ Loaded {len(prompts)} prompts for PPO training")

# # -----------------------------
# # 4. Helper functions
# # -----------------------------
# def get_reward(text):
#     """Compute scalar reward using reward model + tokenizer"""
#     inputs = reward_tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding="max_length",
#         max_length=256
#     )
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     inputs["attention_mask"] = inputs["attention_mask"].to(device)

#     # Debug once: show token range and vocab size
#     if not hasattr(get_reward, "_checked"):
#         max_id = inputs["input_ids"].max().item()
#         print(f"[DEBUG] Max token id for reward model: {max_id}, vocab size: {reward_model.config.vocab_size}")
#         get_reward._checked = True

#     with torch.no_grad():
#         outputs = reward_model(**inputs)
#         cls_embeds = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
#         reward = reward_head(cls_embeds).squeeze().item()

#     return reward


# def generate_response(prompt):
#     """Generate a response using the policy model"""
#     input_ids = policy_tokenizer.encode(prompt, return_tensors="pt").to(device)
#     outputs = policy_model.generate(
#         input_ids,
#         max_new_tokens=80,
#         do_sample=True,
#         temperature=0.8,
#         top_p=0.9,
#         pad_token_id=policy_tokenizer.eos_token_id
#     )
#     return policy_tokenizer.decode(outputs[0], skip_special_tokens=True)


# # -----------------------------
# # 5. PPO training loop (simplified)
# # -----------------------------
# EPOCHS = 1
# BATCH_SIZE = 2
# KL_COEFF = 0.1

# print("🚀 Starting PPO fine-tuning...")

# # Quick test to confirm reward model works
# test_text = "The sky is blue and clear today."
# print("Reward test:", get_reward(test_text))

# policy_model.train()
# for epoch in range(EPOCHS):
#     total_loss = 0
#     for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
#         batch_prompts = prompts[i:i+BATCH_SIZE]
#         batch_loss = 0

#         for prompt in batch_prompts:
#             # Generate response
#             response = generate_response(prompt)
#             full_text = f"{prompt}\n{response}"

#             # Get scalar reward
#             reward = get_reward(full_text)

#             # Policy model forward pass
#             inputs = policy_tokenizer(
#                 full_text,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=256
#             ).to(device)

#             outputs = policy_model(**inputs, labels=inputs["input_ids"])
#             log_prob = -outputs.loss  # negative NLL loss ≈ log-probability
#             kl_penalty = KL_COEFF * outputs.loss
#             ppo_loss = -(reward - kl_penalty)  # PPO simplified objective

#             batch_loss += ppo_loss

#         optimizer.zero_grad()
#         batch_loss.backward()
#         optimizer.step()

#         total_loss += batch_loss.item()

#     avg_loss = total_loss / (len(prompts) / BATCH_SIZE)
#     print(f"📘 Epoch {epoch+1}/{EPOCHS} | Avg PPO Loss: {avg_loss:.4f}")

# # -----------------------------
# # 6. Save final PPO fine-tuned model
# # -----------------------------
# policy_model.save_pretrained(OUTPUT_MODEL_PATH)
# policy_tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
# print(f"🏁 PPO fine-tuned model saved at: {OUTPUT_MODEL_PATH}")


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
    raise FileNotFoundError("❌ Missing accepted_data.jsonl or rejected_data.jsonl — please run policy_model.py first.")

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

# Check vocab alignment
print(f"🔤 Using base: {base_model_name}")
print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"   Model vocab size: {model.config.vocab_size}")

assert tokenizer.vocab_size == model.config.vocab_size, "❌ Tokenizer and model vocab mismatch!"

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
