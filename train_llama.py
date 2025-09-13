"""
Training script for fine-tuning Llama 3.1 models with LoRA or SingLoRA adapters.
The script follows the minimal style of nanoGPT's train.py but uses HuggingFace
Transformers to load the pretrained Llama models.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from llama_model import load_pretrained_llama

# -----------------------------------------------------------------------------
# default config values
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'meta-llama/Llama-3.1-8B'  # or 'meta-llama/Llama-3.1-70B'
wandb_log = False
wandb_project = 'llama3'
wandb_run_name = 'llama3_finetune'

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 4096

dropout = 0.0
bias = False
lora_r = 0
lora_alpha = 1
lora_dropout = 0.0
singlora_r = 0
singlora_alpha = 1
singlora_dropout = 0.0
# QLoRA parameters
qlora_r = 0
qlora_alpha = 1
qlora_dropout = 0.0
qlora_bits = 4
qlora_blocksize = 64
# QSingleLoRA parameters
qsinglora_r = 0
qsinglora_alpha = 1
qsinglora_dropout = 0.0
qsinglora_bits = 4
qsinglora_blocksize = 64

learning_rate = 3e-4
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 1e-5

backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# Check mutual exclusivity between all LoRA variants
lora_variants = [lora_r > 0, singlora_r > 0, qlora_r > 0, qsinglora_r > 0]
if sum(lora_variants) > 1:
    raise ValueError('Only one LoRA variant can be enabled at a time')
# -----------------------------------------------------------------------------

# ddp setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader
meta_path = os.path.join('data', dataset, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

data_dir = os.path.join('data', dataset)
data_dtype = np.uint16 if (meta_vocab_size or 0) <= np.iinfo(np.uint16).max else np.uint32
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=data_dtype, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=data_dtype, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# model init
model = load_pretrained_llama(init_from,
                              lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              singlora_r=singlora_r, singlora_alpha=singlora_alpha, singlora_dropout=singlora_dropout,
                              qlora_r=qlora_r, qlora_alpha=qlora_alpha, qlora_dropout=qlora_dropout,
                              qlora_bits=qlora_bits, qlora_blocksize=qlora_blocksize,
                              qsinglora_r=qsinglora_r, qsinglora_alpha=qsinglora_alpha, qsinglora_dropout=qsinglora_dropout,
                              qsinglora_bits=qsinglora_bits, qsinglora_blocksize=qsinglora_blocksize,
                              device=device, dtype=dtype)

# Handle parameter freezing for all LoRA variants
has_lora = any([lora_r > 0, singlora_r > 0, qlora_r > 0, qsinglora_r > 0])
if has_lora:
    for name, param in model.named_parameters():
        # Freeze parameters that are not LoRA adapters or quantization parameters
        is_adapter = ('lora_' in name or 'singlora_' in name or
                     'quantized_' in name or 'weight_scale' in name or 'weight_zero' in name)
        if not is_adapter:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Determine which mode is enabled
    if qlora_r > 0:
        mode = 'QLoRA'
    elif qsinglora_r > 0:
        mode = 'QSingleLoRA'
    elif singlora_r > 0:
        mode = 'SingLoRA'
    else:
        mode = 'LoRA'

    print(f"{mode} enabled, trainable parameters: {trainable}")

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

if compile:
    print('compiling the model... (takes a ~minute)')
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X).logits
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

iter_num = 0
best_val_loss = 1e9

while True:
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'llama3_ckpt.pt'))
    if iter_num == max_iters:
        break

    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        with ctx:
            logits = model(X).logits
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1)) / gradient_accumulation_steps
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    iter_num += 1

if ddp:
    destroy_process_group()
