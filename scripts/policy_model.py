
# scripts/policy_model.py
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
import os, json, random
from tqdm import tqdm
import torch

# -----------------------------
# 1. Paths
# -----------------------------
DATA_PROCESSED_PATH = "data/processed"
MODEL_SAVE_PATH = "models/policy"
os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# -----------------------------
# 2. Load small Alpaca dataset
# -----------------------------
print("Loading dataset...")
dataset = load_dataset("yahma/alpaca-cleaned")

# Take only 1000 samples for quick run
N_SAMPLES = 1000
if len(dataset["train"]) > N_SAMPLES:
    dataset["train"] = dataset["train"].select(range(N_SAMPLES))

# -----------------------------
# 3. Prepare tokenizer/model
# -----------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format examples
def format_data(example):
    instr = example.get("instruction") or example.get("input") or ""
    out = example.get("output") or example.get("response") or ""
    text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
    return {"text": text, "instruction": instr, "response": out}

dataset = dataset.map(format_data)

# -----------------------------
# 4. Tokenize (mask prompt tokens in labels)
# -----------------------------
max_length = 256

def tokenize_and_make_labels(example):
    # split on the marker to find prompt length
    split_marker = "### Response:\n"
    full_text = example["text"]
    if split_marker in full_text:
        prompt_text, response_text = full_text.split(split_marker, 1)
        prompt_text = prompt_text + split_marker
    else:
        prompt_text = ""
    # tokenize prompt to get prompt length
    prompt_ids = tokenizer(prompt_text, truncation=False, padding=False)["input_ids"]
    # tokenize full text to max_length
    enc = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.copy()
    # mask prompt tokens so loss is only computed on response
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Use non-batched mapping to keep prompt-length logic simple
tokenized = dataset["train"].map(tokenize_and_make_labels, batched=False, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# -----------------------------
# 5. Training setup
# -----------------------------
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    overwrite_output_dir=True,
    num_train_epochs=2,               # increase for better SFT
    per_device_train_batch_size=2,
    save_total_limit=1,
    logging_steps=50,
    learning_rate=5e-5,
    report_to="none",
    fp16=torch.cuda.is_available()
)

model = AutoModelForCausalLM.from_pretrained(model_name)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

print("Starting fine-tuning (SFT)...")
trainer.train()

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"✅ Policy model saved at {MODEL_SAVE_PATH}")

# -----------------------------
# 6. Differentiator stage (generate candidate pairs)
# -----------------------------
print("Running differentiator stage (generating candidate responses)...")
accepted, rejected = [], []

# pick a small subset for generation
gen_count = min(200, len(dataset["train"]))
for sample in tqdm(dataset["train"].select(range(gen_count))):
    instr = sample.get("instruction") or sample.get("input") or sample.get("prompt") or ""
    if not instr:
        continue
    formatted = f"### Instruction:\n{instr}\n\n### Response:\n"
    input_ids = tokenizer.encode(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=80,
        num_return_sequences=2,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    # extract only the generated response part (after marker)
    def extract_resp(full_text):
        if "### Response:" in full_text:
            return full_text.split("### Response:")[-1].strip()
        return full_text.strip()
    resp0 = extract_resp(decoded[0])
    resp1 = extract_resp(decoded[1])

    # simple differentiator (placeholder): prefer longer
    if len(resp0) >= len(resp1):
        acc, rej = resp0, resp1
    else:
        acc, rej = resp1, resp0

    accepted.append({"prompt": instr, "response": acc, "label": "accepted"})
    rejected.append({"prompt": instr, "response": rej, "label": "rejected"})

# Save processed files (UTF-8)
with open(os.path.join(DATA_PROCESSED_PATH, "accepted_data.jsonl"), "w", encoding="utf-8") as fa:
    for e in accepted:
        fa.write(json.dumps(e, ensure_ascii=False) + "\n")

with open(os.path.join(DATA_PROCESSED_PATH, "rejected_data.jsonl"), "w", encoding="utf-8") as fr:
    for e in rejected:
        fr.write(json.dumps(e, ensure_ascii=False) + "\n")

print("✅ Saved accepted_data.jsonl and rejected_data.jsonl.")
