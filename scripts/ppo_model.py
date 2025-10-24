# scripts/ppo_model.py
import os
import json
import torch
from tqdm import tqdm
import torch.nn.functional as F
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
policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH).to(device)

# make sure pad token exists (avoids attention mask warning)
if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

# Load reward checkpoint
reward_checkpoint = torch.load(REWARD_MODEL_PATH, map_location=device)
reward_base_name = reward_checkpoint["tokenizer"]
reward_tokenizer = AutoTokenizer.from_pretrained(reward_base_name)
reward_model = AutoModel.from_pretrained(reward_base_name).to(device)
reward_head = torch.nn.Linear(reward_model.config.hidden_size, 1).to(device)
reward_model.load_state_dict(reward_checkpoint["model_state_dict"])
reward_head.load_state_dict(reward_checkpoint["reward_head_state_dict"])
reward_model.eval()
reward_head.eval()

# PPO optimizer (we update the policy model parameters)
optimizer = AdamW(policy_model.parameters(), lr=1e-6)

# -----------------------------
# 3. Load prompts
# -----------------------------
prompts = []
accepted_file = os.path.join(DATA_PATH, "accepted_data.jsonl")
with open(accepted_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        prompts.append(data["prompt"])
print(f"✅ Loaded {len(prompts)} prompts for PPO training")

# -----------------------------
# 4. Helper functions
# -----------------------------
def get_reward(text: str) -> float:
    inputs = reward_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    ).to(device)

    with torch.no_grad():
        out = reward_model(**inputs).last_hidden_state[:, 0, :]
        r = reward_head(out).squeeze().item()
    return float(r)


def generate_response_and_ids(prompt: str):
    """
    Generate response; return (generated_text, full_sequence_ids, prompt_token_length)
    full_sequence_ids is a 1D tensor on device containing prompt + generated tokens
    prompt_token_length is integer number of tokens in the formatted prompt
    """
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    # get prompt token IDs (without adding special tokens)
    prompt_input_ids = policy_tokenizer.encode(formatted, add_special_tokens=False)
    prompt_len = len(prompt_input_ids)

    input_ids = torch.tensor([prompt_input_ids], device=device)

    # generate; include return_dict_in_generate so we also have sequences
    outputs = policy_model.generate(
        input_ids,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=policy_tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    seq_ids = outputs.sequences[0]  # 1D tensor on device: prompt + generated
    full_text = policy_tokenizer.decode(seq_ids, skip_special_tokens=True)
    # extract generated part (after the marker)
    gen_part = full_text.split("### Response:")[-1].strip()
    return gen_part, seq_ids, prompt_len


def compute_sequence_logprob_from_ids(ids_tensor: torch.Tensor, prompt_len: int):
    """
    Compute sum of log-probabilities of the GENERATED tokens only.
    - ids_tensor: 1D tensor (seq_len) on device containing prompt+generated IDs
    - prompt_len: int number of tokens belonging to prompt (so generated tokens start at index prompt_len)
    Returns scalar tensor (sum logprobs) that requires grad.
    """
    # make batch
    input_ids = ids_tensor.unsqueeze(0).to(device)  # shape [1, seq_len]
    # forward pass to get logits
    outputs = policy_model(input_ids)
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # shift to get predicted logits for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()  # predict token at position i+1 using tokens <= i
    shift_labels = input_ids[:, 1:].contiguous()   # labels are tokens at positions 1..L-1

    # compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [1, seq_len-1, vocab]
    # gather logprobs of the true next tokens
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]

    # determine indices of labels that correspond to generated tokens
    # labels index k corresponds to token at position k+1.
    # generated tokens are at positions >= prompt_len (0-based)
    # thus we need label indices k = prompt_len .. seq_len-1 -> in token_logprobs these are indices (k-1) = prompt_len-1 .. seq_len-2
    seq_len = input_ids.size(1)
    if prompt_len <= 0:
        # if prompt_len is 0, include all token_logprobs
        gen_token_logprobs = token_logprobs
    else:
        start_idx = max(0, prompt_len - 1)
        gen_token_logprobs = token_logprobs[:, start_idx:]  # shape [1, gen_len]

    # sum generated token logprobs -> scalar tensor (requires grad)
    seq_logprob = gen_token_logprobs.sum()
    return seq_logprob


# -----------------------------
# 5. PPO-like training loop (simple, small-batch)
# -----------------------------
EPOCHS = 1
BATCH_SIZE = 2
KL_COEFF = 0.1

# moving average baseline for reward
baseline = None
alpha = 0.9  # baseline smoothing

print("🚀 Starting PPO-style fine-tuning... (this is a small, illustrative loop)")

policy_model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        losses = []

        for prompt in batch_prompts:
            # Generate (no-grad)
            with torch.no_grad():
                gen_text, seq_ids, prompt_len = generate_response_and_ids(prompt)

            full_text = f"{prompt}\n{gen_text}"
            reward = get_reward(full_text)

            # update baseline (moving average)
            if baseline is None:
                baseline = reward
            else:
                baseline = alpha * baseline + (1 - alpha) * reward
            advantage = reward - baseline

            # compute sequence logprob (with grad) for generated tokens
            seq_ids_var = seq_ids.detach().clone().to(device)
            seq_logprob = compute_sequence_logprob_from_ids(seq_ids_var, prompt_len)  # scalar tensor

            # policy gradient style loss: minimize -logprob * advantage
            # advantage can be positive (increase prob) or negative (decrease prob)
            loss_term = - seq_logprob * (advantage)

            # optional KL penalty placeholder (better to compute against reference policy)
            kl_penalty = KL_COEFF * 0.0
            total_term = loss_term + kl_penalty

            losses.append(total_term)

        if not losses:
            continue

        batch_loss_tensor = torch.stack(losses).mean()
        optimizer.zero_grad()
        batch_loss_tensor.backward()
        optimizer.step()

        total_loss += batch_loss_tensor.item()

    avg_loss = total_loss / max(1.0, (len(prompts) / BATCH_SIZE))
    print(f"📘 Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Baseline: {baseline:.4f}")

# -----------------------------
# 6. Save final PPO fine-tuned model
# -----------------------------
policy_model.save_pretrained(OUTPUT_MODEL_PATH)
policy_tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
print(f"🏁 PPO fine-tuned model saved at: {OUTPUT_MODEL_PATH}")
