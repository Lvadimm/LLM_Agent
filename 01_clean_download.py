# Downloads instruction datasets, formats them for MLX, and creates train/valid splits.

import json
import random
import os
from datasets import load_dataset, concatenate_datasets

# Ensure the output directory exists
os.makedirs("data", exist_ok=True)

print("Loading datasets...")

# 1. Load source datasets
ds1 = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
ds2 = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")

print(f"Total examples loaded: {len(ds1) + len(ds2):,}")

# 2. Combine and shuffle
combined = concatenate_datasets([ds1, ds2]).shuffle(seed=42)

def format_for_mlx(ex):
    """
    Normalizes data into the MLX-LM chat format:
    {'messages': [{'role': 'user', ...}, {'role': 'assistant', ...}]}
    """
    instruction = ex.get("instruction") or ex.get("prompt", "").strip()
    output = ex.get("output") or ex.get("response", "").strip()

    if not instruction or not output:
        return None

    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": f"```python\n{output}\n```"}
        ]
    }

# 3. Process and format data
print("Formatting data for MLX...")
entries = []
for ex in combined:
    formatted_entry = format_for_mlx(ex)
    if formatted_entry:
        entries.append(formatted_entry)

# 4. Create train/validation split (90/10)
# MLX-LM requires a validation set to monitor loss.
random.seed(42)
random.shuffle(entries)

split_idx = int(0.9 * len(entries))
train_entries = entries[:split_idx]
valid_entries = entries[split_idx:]

print(f"Split results: {len(train_entries):,} train, {len(valid_entries):,} validation.")

# 5. Save files
print("Saving to ./data/...")

with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for entry in train_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open("data/valid.jsonl", "w", encoding="utf-8") as f:
    for entry in valid_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Done. Ready for training.")
