# Configuration for fine-tuning GPT-2 with QSingleLoRA adapters
out_dir = 'out-gpt2-qsinglora'
init_from = 'gpt2'
block_size = 1024
batch_size = 4
learning_rate = 3e-4
max_iters = 1000

# QSingleLoRA parameters
qsinglora_r = 8
qsinglora_alpha = 16
qsinglora_dropout = 0.05
qsinglora_bits = 4
qsinglora_blocksize = 64

# Note: Only one LoRA variant can be enabled at a time
# To use QLoRA instead, comment out qsinglora_r and uncomment qlora_r below:
# qlora_r = 8
# qlora_alpha = 16
# qlora_dropout = 0.05
# qlora_bits = 4
# qlora_blocksize = 64
