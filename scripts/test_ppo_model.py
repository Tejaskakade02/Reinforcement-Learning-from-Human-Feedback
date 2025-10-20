import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# -----------------------------
# 1. Paths and device setup
# -----------------------------
MODEL_PATH = "models/ppo"  # Path to your PPO fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -----------------------------
# 2. Load the PPO fine-tuned model
# -----------------------------
print("📦 Loading PPO fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# Ensure pad token exists (important for generation)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print("✅ Model and tokenizer loaded successfully!")

# -----------------------------
# 3. Define the testing function
# -----------------------------
def generate_response(prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Generate a response using the PPO fine-tuned model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
    return generated_text

# -----------------------------
# 4. Run sample tests
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
    print(f"💬 Response:\n{response}\n{'-'*80}\n")

# -----------------------------
# 5. Optional: Interactive Chat
# -----------------------------
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Exiting chat.")
        break
    response = generate_response(user_input)
    print(f"AI: {response}\n")
