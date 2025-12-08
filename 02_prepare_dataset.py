
# Downloads the quantized Qwen 2.5 Coder model for MLX usage.

from huggingface_hub import snapshot_download
import os

# Ensure the download directory exists (optional, but good practice)
os.makedirs("base_model", exist_ok=True)

print("Starting model download (approx. 4GB)...")

# 1. Download the model repository
# We use local_dir_use_symlinks=False to ensure we get actual files,
# not just links to the Hugging Face cache.
snapshot_download(
    repo_id="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    local_dir="./base_model",
    local_dir_use_symlinks=False,
    resume_download=True
)

# 2. Verify download completion
if os.path.exists("./base_model/config.json"):
    print("Download complete. Model files are in ./base_model")
else:
    print("Warning: Download may have failed. Please check your connection.")

print("Ready for training.")
