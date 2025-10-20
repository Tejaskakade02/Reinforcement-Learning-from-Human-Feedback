import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.optim import AdamW

# -----------------------------
# 1. Paths and setup
# -----------------------------
POLICY_MODEL_PATH = "models/policy"
REWARD_MODEL_PATH = "models/reward/reward_model.pt"
DATA_PATH = "data/processed"
OUTPUT_MODEL_PATH = "models/ppo"
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -----------------------------
# 2. Load models
# -----------------------------
# Load tokenizer and policy model (GPT-2)
policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH).to(device)

# Ensure pad token exists
if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

# Load reward model checkpoint
reward_checkpoint = torch.load(REWARD_MODEL_PATH, map_location=device)

# Load DistilBERT base model & tokenizer (same as used in reward_model.py)
reward_base_name = reward_checkpoint["tokenizer"]
reward_tokenizer = AutoTokenizer.from_pretrained(reward_base_name)
reward_model = AutoModel.from_pretrained(reward_base_name).to(device)

# Add reward head
reward_head = torch.nn.Linear(reward_model.config.hidden_size, 1).to(device)
reward_model.load_state_dict(reward_checkpoint["model_state_dict"])
reward_head.load_state_dict(reward_checkpoint["reward_head_state_dict"])
reward_model.eval()
reward_head.eval()

# PPO optimizer
optimizer = AdamW(policy_model.parameters(), lr=1e-6)

# -----------------------------
# 3. Load prompts
# -----------------------------
prompts = []
accepted_file = os.path.join(DATA_PATH, "accepted_data.jsonl")

with open(accepted_file, "r") as f:
    for line in f.readlines():
        data = json.loads(line)
        prompts.append(data["prompt"])

print(f"✅ Loaded {len(prompts)} prompts for PPO training")

# -----------------------------
# 4. Helper functions
# -----------------------------
def get_reward(text):
    """Compute scalar reward from reward model safely"""
    inputs = reward_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["attention_mask"] = inputs["attention_mask"].to(device)

    # 🔧 FIX: Prevent index out of range (clamp token IDs)
    vocab_size = reward_model.config.vocab_size
    inputs["input_ids"] = torch.clamp(inputs["input_ids"], 0, vocab_size - 1)

    # Debug (print once)
    if not hasattr(get_reward, "_checked"):
        max_id = inputs["input_ids"].max().item()
        print(f"[DEBUG] Reward model vocab check → Max token ID: {max_id} / {vocab_size}")
        get_reward._checked = True

    with torch.no_grad():
        outputs = reward_model(**inputs)
        cls_embeds = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        reward = reward_head(cls_embeds).squeeze().item()
    return reward


def generate_response(prompt):
    """Generate a response using the policy model"""
    input_ids = policy_tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = policy_model.generate(
        input_ids,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=policy_tokenizer.eos_token_id
    )
    return policy_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# 5. PPO training loop (simplified)
# -----------------------------
EPOCHS = 1
BATCH_SIZE = 2
KL_COEFF = 0.1

print("🚀 Starting PPO fine-tuning...")

# Optional: Sanity check reward model before loop
test_text = "The weather is nice and pleasant today."
print("Reward test output:", get_reward(test_text))

policy_model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        batch_loss = 0

        for prompt in batch_prompts:
            # Generate response
            response = generate_response(prompt)
            full_text = f"{prompt}\n{response}"

            # Get reward
            reward = get_reward(full_text)

            # Compute policy loss
            inputs = policy_tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            outputs = policy_model(**inputs, labels=inputs["input_ids"])
            log_prob = -outputs.loss  # negative loss ≈ log-prob
            kl_penalty = KL_COEFF * outputs.loss
            ppo_loss = -(reward - kl_penalty)  # maximize (reward - KL)

            batch_loss += ppo_loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    avg_loss = total_loss / (len(prompts) / BATCH_SIZE)
    print(f"📘 Epoch {epoch+1}/{EPOCHS} | Avg PPO Loss: {avg_loss:.4f}")

# -----------------------------
# 6. Save final PPO fine-tuned model
# -----------------------------
policy_model.save_pretrained(OUTPUT_MODEL_PATH)
policy_tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
print(f"🏁 PPO fine-tuned model saved at: {OUTPUT_MODEL_PATH}")
