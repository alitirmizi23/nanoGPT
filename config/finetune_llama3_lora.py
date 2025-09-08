# Configuration for fine-tuning Llama 3.1 with LoRA adapters
out_dir = 'out-llama3'
init_from = 'meta-llama/Llama-3.1-8B'
block_size = 4096
batch_size = 4
learning_rate = 3e-4
max_iters = 1000
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
# set singlora_r > 0 instead of lora_r to use SingLoRA
