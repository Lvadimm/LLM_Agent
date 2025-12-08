# Loads the base model and fuses the LoRA adapters in memory for inference.

from mlx_lm import load, generate

print("Loading base model and adapters...")

# 1. Load the model
# The adapter_path argument automatically fuses the LoRA weights
# into the base model at runtime.
model, tokenizer = load(
    "./base_model",
    adapter_path="./adapters_final"
)

print("Model loaded successfully.")

# 2. Define a test prompt
# Note: Since the model was trained with a chat template, we manually format
# the prompt here to resemble the expected input structure.
prompt_content = "Write a modern FastAPI app with JWT authentication, PostgreSQL + SQLAlchemy 2.0, rate limiting, and proper error handling. Use Python 3.11+ syntax."

# Construct the prompt using the standard chat format (User/Assistant)
prompt = f"<|im_start|>user\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n```python"

print(f"\nGenerating response for: {prompt_content[:50]}...\n")

# 3. Generate response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=1800,
    verbose=True # Prints generation speed/stats
)

print("\n--- Generated Output ---\n")
print(response)
