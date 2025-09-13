# Configuration for fine-tuning Llama 3.1 with QLoRA adapters
out_dir = 'out-llama3-qlora'
init_from = 'meta-llama/Llama-3.1-8B'  # or 'meta-llama/Llama-3.1-70B'
block_size = 4096
batch_size = 4
learning_rate = 3e-4
max_iters = 1000

# QLoRA parameters
qlora_r = 8
qlora_alpha = 16
qlora_dropout = 0.05
qlora_bits = 4
qlora_blocksize = 64

# Note: Only one LoRA variant can be enabled at a time
# To use QSingleLoRA instead, comment out qlora_r and uncomment qsinglora_r below:
# qsinglora_r = 8
# qsinglora_alpha = 16
# qsinglora_dropout = 0.05
# qsinglora_bits = 4
# qsinglora_blocksize = 64
