# Configuration for fine-tuning Llama 3.2-1B with QLoRA adapters
out_dir = 'out-llama32-qlora'
init_from = 'meta-llama/Llama-3.2-1B'
block_size = 4096
batch_size = 16  # Even larger batch size for quantized 1B model
learning_rate = 3e-4
max_iters = 1000

# QLoRA parameters
qlora_r = 8
qlora_alpha = 16
qlora_dropout = 0.05
qlora_bits = 4
qlora_blocksize = 64

# Alternative QLoRA variants (uncomment to use):
# For QSingleLoRA (quantized SingleLoRA):
# qsinglora_r = 8
# qsinglora_alpha = 16
# qsinglora_dropout = 0.05
# qsinglora_bits = 4
# qsinglora_blocksize = 64
