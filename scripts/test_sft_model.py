import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. Configuration
# -----------------------------
MODEL_PATH = "models/policy"  # path where your SFT model is saved
BASE_MODEL = "gpt2"               # base model for comparison
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -----------------------------
# 2. Load models
# -----------------------------
print("📦 Loading models and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Ensure padding token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load SFT fine-tuned model
sft_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
sft_model.eval()

# Load base model for comparison
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
base_model.eval()

print("✅ Both models loaded successfully!")

# -----------------------------
# 3. Define helper for generation
# -----------------------------
def generate(model, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# -----------------------------
# 4. Define test prompts
# -----------------------------
test_prompts = [
    "Explain what reinforcement learning is.",
    "Write a short inspirational message for students.",
    "List three benefits of renewable energy.",
    "Describe how AI learns from human feedback."
]

# -----------------------------
# 5. Compare outputs
# -----------------------------
print("\n🚀 Running SFT Validation Test...\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"🧠 Prompt {i}: {prompt}\n")

    base_output = generate(base_model, prompt)
    sft_output = generate(sft_model, prompt)

    print(f"💬 Base Model:\n{base_output}\n")
    print(f"✨ SFT Model:\n{sft_output}\n")
    print("-" * 100)

print("\n✅ Test completed! Compare if the SFT model gives more coherent, aligned, or helpful responses.\n")
