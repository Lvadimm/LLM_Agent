#!/bin/bash
# Fine-tunes Qwen2.5-Coder-7B using LoRA on MLX.
# Configuration optimized for memory efficiency on Apple Silicon.

echo "Starting training workflow..."
echo "Configuration: Effective batch size of 48 (1 x 48 accumulation)."

# Run the training command
mlx_lm.lora \
  --model ./base_model \
  --train \
  --data ./data \
  --adapter-path ./adapters_final \  #  adapters_v2_self_taught for chat fine tunning training (export to training) to keep versions
  --fine-tune-type lora \
  --num-layers 28 \
  --iters 7200 \          # 600 for chat fine tunning training (export to training)
  --batch-size 1 \
  --grad-accumulation-steps 48 \   # 16 for chat fine tunning training (export to training)
  --learning-rate 1e-5 \
  --optimizer adamw \
  --max-seq-length 2048 \
  --grad-checkpoint \
  --save-every 1000 \
  --steps-per-report 100 \
  --seed 42

echo "Training complete. Adapters saved to ./adapters_final"
