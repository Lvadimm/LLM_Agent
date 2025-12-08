import json
import glob
import os
import random

CHAT_DIR = "./chats"
OUTPUT_FILE = "my_custom_finetune.jsonl"

def convert_chats_to_training_data():
    files = glob.glob(f"{CHAT_DIR}/*.json")
    print(f"Found {len(files)} conversation logs.")
    
    training_data = []
    
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            history = data.get("history", [])
            
            # Skip empty or short conversations
            if not history or len(history) < 1:
                continue
                
            # Convert to MLX / OpenAI Chat Format
            # Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
            messages = []
            
            # Add a system prompt to help guide the style
            messages.append({
                "role": "system",
                "content": "You are an Expert Senior Software Engineer."
            })
            
            for turn in history:
                user_msg, ai_msg = turn
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": ai_msg})
            
            # Only save if we have a valid conversation pair
            if len(messages) > 1:
                training_data.append({"messages": messages})
                
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    # Shuffle to prevent bias
    random.shuffle(training_data)
    
    # Save to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Success! Exported {len(training_data)} training examples to {OUTPUT_FILE}")
    print(f"🚀 You can now fine-tune using: python -m mlx_lm.lora --train --data {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_chats_to_training_data()
