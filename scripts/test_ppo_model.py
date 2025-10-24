import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import os

# -----------------------------
# 1. Paths and device setup
# -----------------------------
POLICY_MODEL_PATH = "models/ppo"          # Path to PPO fine-tuned model
REWARD_MODEL_PATH = "models/reward/reward_model.pt"  # Reward model checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -----------------------------
# 2. Load the PPO fine-tuned model
# -----------------------------
print("📦 Loading PPO fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH).to(device)

# Ensure pad token exists (important for generation)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print("✅ Model and tokenizer loaded successfully!")

# -----------------------------
# 3. Define text generation function
# -----------------------------
def generate_response(prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Generate a response using the PPO fine-tuned model"""
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the generated answer part
    if "### Response:" in generated_text:
        generated_text = generated_text.split("### Response:")[-1].strip()
    return generated_text

# -----------------------------
# 4. Load reward model for confidence scoring
# -----------------------------
print("🏆 Loading reward model for confidence scoring...")
reward_checkpoint = torch.load(REWARD_MODEL_PATH, map_location=device)

reward_base_name = reward_checkpoint["tokenizer"]
reward_tokenizer = AutoTokenizer.from_pretrained(reward_base_name)
reward_model = AutoModel.from_pretrained(reward_base_name).to(device)

# Add linear head and load weights
reward_head = torch.nn.Linear(reward_model.config.hidden_size, 1).to(device)
reward_model.load_state_dict(reward_checkpoint["model_state_dict"])
reward_head.load_state_dict(reward_checkpoint["reward_head_state_dict"])
reward_model.eval()
reward_head.eval()

print("✅ Reward model loaded successfully!")

# -----------------------------
# 5. Confidence computation function
# -----------------------------
def compute_confidence(text):
    """Compute confidence score (0–1) using reward model"""
    inputs = reward_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = reward_model(**inputs)
        cls_embeds = outputs.last_hidden_state[:, 0, :]
        reward_value = reward_head(cls_embeds).squeeze().item()
        confidence = float(torch.sigmoid(torch.tensor(reward_value, device=device)))
    return confidence

# -----------------------------
# 6. Run sample tests
# -----------------------------
test_prompts = [
    "Explain reinforcement learning in simple terms.",
    "Write a short motivational message for students.",
    "What are the advantages of using renewable energy?",
    "Tell a fun fact about space exploration.",
    "How does human feedback help improve AI models?"
]

print("\n🚀 Starting PPO Model Testing...\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"🧠 Prompt {i}: {prompt}")
    response = generate_response(prompt)
    conf = compute_confidence(f"{prompt}\n{response}")
    print(f"💬 Response:\n{response}")
    print(f"📈 Confidence Score: {conf:.3f}")
    print('-' * 80)

# -----------------------------
# 7. Optional: Interactive Chat
# -----------------------------
print("\n💬 Enter prompts (type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower().strip() in ["exit", "quit"]:
        print("👋 Exiting chat.")
        break
    response = generate_response(user_input)
    conf = compute_confidence(f"{user_input}\n{response}")
    print(f"AI: {response}")
    print(f"📈 Confidence: {conf:.3f}\n")
