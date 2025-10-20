# import os
# import json
# import random
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# from tqdm import tqdm

# # -----------------------------
# # 1. Basic setup
# # -----------------------------
# DATA_RAW_PATH = "data/raw/dolly_15k"
# DATA_PROCESSED_PATH = "data/processed"
# MODEL_SAVE_PATH = "models/policy"
# os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
# os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# # -----------------------------
# # 2. Load Dolly-15k dataset
# # -----------------------------
# print("Loading dataset...")
# dataset = load_dataset("databricks/databricks-dolly-15k")

# # -----------------------------
# # 3. Prepare model and tokenizer
# # -----------------------------
# model_name = "gpt2"  # you can change to "EleutherAI/pythia-70m" or similar
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# def format_data(example):
#     prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
#     return {"text": prompt}

# dataset = dataset.map(format_data)

# # -----------------------------
# # 4. Tokenize data
# # -----------------------------
# def tokenize(example):
#     return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

# tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# # -----------------------------
# # 5. Data collator and training args
# # -----------------------------
# data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# training_args = TrainingArguments(
#     output_dir=MODEL_SAVE_PATH,
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_device_train_batch_size=2,
#     save_steps=500,
#     save_total_limit=2,
#     logging_steps=100,
#     learning_rate=5e-5,
#     fp16=True,
#     report_to="none"
# )

# # -----------------------------
# # 6. Fine-tune the model
# # -----------------------------
# model = AutoModelForCausalLM.from_pretrained(model_name)

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     data_collator=data_collator
# )

# print("Starting fine-tuning...")
# trainer.train()

# # Save fine-tuned model
# trainer.save_model(MODEL_SAVE_PATH)
# tokenizer.save_pretrained(MODEL_SAVE_PATH)
# print(f"✅ Policy model saved at {MODEL_SAVE_PATH}")

# # -----------------------------
# # 7. Differentiator stage (simulation)
# # -----------------------------
# print("Running differentiator stage...")

# accepted_data = []
# rejected_data = []

# for sample in tqdm(dataset["train"].select(range(300))):  # small subset for example
#     prompt = f"{sample['instruction']}"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")

#     # Generate multiple responses
#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=80,
#         num_return_sequences=2,
#         do_sample=True,
#         temperature=0.8,
#         top_p=0.95
#     )

#     decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

#     # Simple differentiator rule (placeholder)
#     # For now: choose the longer response as "accepted"
#     if len(decoded[0]) > len(decoded[1]):
#         accepted = decoded[0]
#         rejected = decoded[1]
#     else:
#         accepted = decoded[1]
#         rejected = decoded[0]

#     accepted_data.append({
#         "prompt": prompt,
#         "response": accepted,
#         "label": "accepted"
#     })
#     rejected_data.append({
#         "prompt": prompt,
#         "response": rejected,
#         "label": "rejected"
#     })

# # -----------------------------
# # 8. Save clean filtered data
# # -----------------------------
# with open(os.path.join(DATA_PROCESSED_PATH, "accepted_data.jsonl"), "w") as fa:
#     for entry in accepted_data:
#         fa.write(json.dumps(entry) + "\n")

# with open(os.path.join(DATA_PROCESSED_PATH, "rejected_data.jsonl"), "w") as fr:
#     for entry in rejected_data:
#         fr.write(json.dumps(entry) + "\n")

# print(f"✅ Created accepted_data.jsonl and rejected_data.jsonl in {DATA_PROCESSED_PATH}")


from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
import os, json, random
from tqdm import tqdm

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
dataset["train"] = dataset["train"].select(range(1000))

# -----------------------------
# 3. Prepare tokenizer/model
# -----------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_data(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

dataset = dataset.map(format_data)

# -----------------------------
# 4. Tokenize
# -----------------------------
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# -----------------------------
# 5. Training setup
# -----------------------------
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_total_limit=1,
    logging_steps=50,
    learning_rate=5e-5,
    report_to="none",
    fp16=True
)

model = AutoModelForCausalLM.from_pretrained(model_name)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

print("Starting fine-tuning...")
trainer.train()

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"✅ Policy model saved at {MODEL_SAVE_PATH}")

# -----------------------------
# 6. Differentiator stage (simple simulation)
# -----------------------------
print("Running differentiator stage...")
accepted, rejected = [], []

for sample in tqdm(dataset["train"].select(range(100))):
    prompt = f"{sample['instruction']}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_new_tokens=80,
        num_return_sequences=2,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    if len(decoded[0]) > len(decoded[1]):
        accepted.append({"prompt": prompt, "response": decoded[0], "label": "accepted"})
        rejected.append({"prompt": prompt, "response": decoded[1], "label": "rejected"})
    else:
        accepted.append({"prompt": prompt, "response": decoded[1], "label": "accepted"})
        rejected.append({"prompt": prompt, "response": decoded[0], "label": "rejected"})

# Save to processed folder
with open(os.path.join(DATA_PROCESSED_PATH, "accepted_data.jsonl"), "w") as fa:
    for e in accepted: fa.write(json.dumps(e) + "\n")

with open(os.path.join(DATA_PROCESSED_PATH, "rejected_data.jsonl"), "w") as fr:
    for e in rejected: fr.write(json.dumps(e) + "\n")

print("✅ Saved accepted_data.jsonl and rejected_data.jsonl.")
