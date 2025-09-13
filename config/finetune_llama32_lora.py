# Configuration for fine-tuning Llama 3.2-1B with LoRA adapters
out_dir = 'out-llama32-lora'
init_from = 'meta-llama/Llama-3.2-1B'
block_size = 4096
batch_size = 8  # Larger batch size for small 1B model
learning_rate = 3e-4
max_iters = 1000
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

# Alternative LoRA variants (uncomment to use):
# set singlora_r > 0 instead of lora_r to use SingLoRA
# singlora_r = 8
# singlora_alpha = 16
# singlora_dropout = 0.05

# For QLoRA (quantized LoRA):
# qlora_r = 8
# qlora_alpha = 16
# qlora_dropout = 0.05
# qlora_bits = 4
# qlora_blocksize = 64

# For QSingleLoRA (quantized SingleLoRA):
# qsinglora_r = 8
# qsinglora_alpha = 16
# qsinglora_dropout = 0.05
# qsinglora_bits = 4
# qsinglora_blocksize = 64
