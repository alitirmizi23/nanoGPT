# Configuration for supervised fine-tuning Llama 3.1 with QSingleLoRA adapters
out_dir = 'out-llama3-sft-qsinglora'
init_from = 'meta-llama/Llama-3.1-8B'
dataset = 'tulu'  # SFT dataset (prepared by data/sft/prepare.py)
block_size = 4096
batch_size = 2  # Smaller batch size for QSingleLoRA (memory efficient but slower)
learning_rate = 2e-5  # Smaller LR for fine-tuning
max_iters = 2000  # More iterations for SFT

# QSingleLoRA parameters
qsinglora_r = 16
qsinglora_alpha = 32
qsinglora_dropout = 0.05
qsinglora_bits = 4
qsinglora_blocksize = 64

# SFT-specific settings
dropout = 0.0
weight_decay = 0.01  # Lighter weight decay for SFT
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Evaluation settings for SFT
eval_interval = 500
eval_iters = 100
