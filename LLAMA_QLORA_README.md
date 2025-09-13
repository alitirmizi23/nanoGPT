# QLoRA and QSingleLoRA for Llama 3 Training

This document describes how to use QLoRA (Quantized Low-Rank Adaptation) and QSingleLoRA implementations with Llama 3 models in nanoGPT.

## Overview

The Llama implementation supports all four LoRA variants:
- **LoRA**: Standard Low-Rank Adaptation
- **SingLoRA**: Single-matrix Low-Rank Adaptation
- **QLoRA**: Quantized Low-Rank Adaptation (4-bit quantization)
- **QSingleLoRA**: Quantized Single-matrix Low-Rank Adaptation

## Setup

### Prerequisites

1. **HuggingFace Authentication**: You need access to meta-llama models
   ```bash
   export HF_TOKEN=your_huggingface_token
   # or login via CLI
   huggingface-cli login
   ```

2. **Training Data**: Prepare your dataset (e.g., OpenWebText)
   ```bash
   python data/openwebtext/prepare.py
   ```

## Usage

### Training with QLoRA

#### Using Configuration Files

1. **QLoRA Configuration**:
   ```python
   # config/finetune_llama3_qlora.py
   out_dir = 'out-llama3-qlora'
   init_from = 'meta-llama/Llama-3.1-8B'
   qlora_r = 8
   qlora_alpha = 16
   qlora_dropout = 0.05
   qlora_bits = 4
   qlora_blocksize = 64
   ```

2. **Train with QLoRA**:
   ```bash
   torchrun --standalone --nproc_per_node=1 train_llama.py config/finetune_llama3_qlora.py
   ```

#### Using Command Line Overrides

```bash
torchrun --standalone --nproc_per_node=1 train_llama.py config/finetune_llama3_lora.py \
    --qlora_r=8 \
    --qlora_alpha=16 \
    --qlora_dropout=0.05 \
    --qlora_bits=4 \
    --qlora_blocksize=64
```

### Training with QSingleLoRA

#### Configuration File:
```python
# config/finetune_llama3_qsinglora.py
out_dir = 'out-llama3-qsinglora'
init_from = 'meta-llama/Llama-3.1-8B'
qsinglora_r = 8
qsinglora_alpha = 16
qsinglora_dropout = 0.05
qsinglora_bits = 4
qsinglora_blocksize = 64
```

#### Training Command:
```bash
torchrun --standalone --nproc_per_node=1 train_llama.py config/finetune_llama3_qsinglora.py
```

## Configuration Parameters

### QLoRA Parameters
- `qlora_r`: Rank of LoRA adaptation matrices (default: 0, disabled)
- `qlora_alpha`: Scaling factor (default: 1)
- `qlora_dropout`: Dropout rate (default: 0.0)
- `qlora_bits`: Quantization bits (default: 4)
- `qlora_blocksize`: Block size for quantization (default: 64)

### QSingleLoRA Parameters
- `qsinglora_r`: Rank of SingleLoRA adaptation matrix (default: 0, disabled)
- `qsinglora_alpha`: Scaling factor (default: 1)
- `qsinglora_dropout`: Dropout rate (default: 0.0)
- `qsinglora_bits`: Quantization bits (default: 4)
- `qsinglora_blocksize`: Block size for quantization (default: 64)

## Memory Benefits

| Method | Memory Reduction | Training Params |
|--------|------------------|-----------------|
| Full Fine-tuning | ~0% | All parameters |
| LoRA | ~90% | ~0.5M parameters |
| SingLoRA | ~90% | ~0.25M parameters |
| **QLoRA** | **~95%** | **~0.5M parameters** |
| **QSingleLoRA** | **~95%** | **~0.25M parameters** |

## Implementation Details

### Quantization Process

1. **Model Loading**: Llama model loaded from HuggingFace
2. **Linear Replacement**: All `nn.Linear` layers replaced with QLoRA/QSingleLoRA
3. **Weight Copying**: Original weights copied to quantized layers
4. **Quantization**: Weights quantized to 4-bit with block-wise scaling
5. **Parameter Freezing**: Only adapter parameters remain trainable

### Automatic Weight Quantization

Unlike GPT models that require manual quantization after loading, Llama models automatically quantize weights during the linear replacement process in `_replace_linear()`.

## Multi-GPU Training

All variants support distributed training:

```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 train_llama.py config/finetune_llama3_qlora.py

# Multi-GPU (8 GPUs)
torchrun --standalone --nproc_per_node=8 train_llama.py config/finetune_llama3_qlora.py
```

## Model Compatibility

- **Llama 3.1-8B**: Fully supported
- **Llama 3.1-70B**: Fully supported
- **Llama 3-8B/70B**: Should work (not tested)
- **Other Llama variants**: May require minor adjustments

## Best Practices

### Memory Optimization

1. **Use QLoRA/QSingleLoRA** for large models (>7B parameters)
2. **Smaller batch sizes** may be needed with quantization
3. **Monitor GPU memory** usage during training
4. **Use gradient checkpointing** if needed

### Hyperparameter Tuning

1. **qlora_r = 8-16**: Good balance of performance vs. memory
2. **qlora_alpha = 2*qlora_r**: Standard scaling factor
3. **qlora_blocksize = 64**: Optimal for most use cases
4. **Learning rate**: 3e-4 works well for fine-tuning

### Performance Tips

1. **Use bfloat16** for better numerical stability
2. **Enable gradient scaling** for mixed precision
3. **Use compiled model** (`compile=True`) for speed
4. **Monitor trainable parameters** to ensure correct setup

## Troubleshooting

### Common Issues

1. **Authentication Error**:
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. **Out of Memory**:
   - Reduce batch size
   - Use smaller `qlora_r`
   - Enable gradient checkpointing

3. **Slow Training**:
   - Use `compile=True`
   - Use larger batch sizes if memory allows
   - Use bfloat16 instead of float16

### Debugging

Check that the correct variant is enabled:
```python
# In training logs, you should see:
# "QLoRA enabled, trainable parameters: XXX"
# or
# "QSingleLoRA enabled, trainable parameters: XXX"
```

## Files Modified

- `llama_model.py`: Added QLoRA/QSingleLoRA support
- `train_llama.py`: Updated training script
- `config/finetune_llama3_qlora.py`: QLoRA configuration
- `config/finetune_llama3_qsinglora.py`: QSingleLoRA configuration
- `test_llama_qlora.py`: Test script

## Testing

Run the test script to verify implementations:
```bash
python test_llama_qlora.py
```

Note: Requires HuggingFace authentication for full testing.
