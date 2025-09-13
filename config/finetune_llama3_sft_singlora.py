# Configuration for supervised fine-tuning Llama 3.1 with SingleLoRA adapters
out_dir = 'out-llama3-sft-singlora'
init_from = 'meta-llama/Llama-3.1-8B'
dataset = 'tulu'  # SFT dataset (prepared by data/sft/prepare.py)
block_size = 4096
batch_size = 4  # Larger batch size for SingleLoRA (less memory intensive)
learning_rate = 2e-5  # Smaller LR for fine-tuning
max_iters = 2000  # More iterations for SFT

# SingleLoRA parameters
singlora_r = 16
singlora_alpha = 32
singlora_dropout = 0.05

# SFT-specific settings
dropout = 0.0
weight_decay = 0.01  # Lighter weight decay for SFT
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Evaluation settings for SFT
eval_interval = 500
eval_iters = 100
