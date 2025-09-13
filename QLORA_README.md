# QLoRA and QSingleLoRA Implementation

This document describes the QLoRA (Quantized Low-Rank Adaptation) and QSingleLoRA implementations added to the nanoGPT codebase.

## Overview

QLoRA combines quantization with Low-Rank Adaptation to reduce memory usage while maintaining fine-tuning performance. QSingleLoRA is a quantized version of the existing SingleLoRA implementation.

## Key Features

- **4-bit quantization**: Base model weights are quantized to 4 bits to reduce memory footprint
- **LoRA adaptation**: Trainable low-rank adapters are applied on top of quantized weights
- **Block-wise quantization**: Weights are quantized in blocks for better precision
- **Mutual exclusivity**: Only one LoRA variant can be enabled at a time

## Usage

### Configuration Parameters

#### QLoRA Parameters
- `qlora_r`: Rank of the LoRA adaptation matrices (default: 0, disabled)
- `qlora_alpha`: Scaling factor for LoRA (default: 1)
- `qlora_dropout`: Dropout rate for LoRA layers (default: 0.0)
- `qlora_bits`: Quantization bits (default: 4)
- `qlora_blocksize`: Block size for quantization (default: 64)

#### QSingleLoRA Parameters
- `qsinglora_r`: Rank of the SingleLoRA adaptation matrix (default: 0, disabled)
- `qsinglora_alpha`: Scaling factor for SingleLoRA (default: 1)
- `qsinglora_dropout`: Dropout rate for SingleLoRA layers (default: 0.0)
- `qsinglora_bits`: Quantization bits (default: 4)
- `qsinglora_blocksize`: Block size for quantization (default: 64)

### Example Configurations

#### Using QLoRA
```python
from model import GPT, GPTConfig

config = GPTConfig(
    # Model architecture
    n_layer=12,
    n_head=12,
    n_embd=768,

    # QLoRA parameters
    qlora_r=8,
    qlora_alpha=16,
    qlora_dropout=0.05,
    qlora_bits=4,
    qlora_blocksize=64
)

model = GPT(config)
```

#### Using QSingleLoRA
```python
config = GPTConfig(
    # Model architecture
    n_layer=12,
    n_head=12,
    n_embd=768,

    # QSingleLoRA parameters
    qsinglora_r=8,
    qsinglora_alpha=16,
    qsinglora_dropout=0.05,
    qsinglora_bits=4,
    qsinglora_blocksize=64
)

model = GPT(config)
```

### Training with Pretrained Models

When loading from pretrained models, you need to quantize the weights after loading:

```python
# Load pretrained model with QLoRA
model = GPT.from_pretrained('gpt2', {
    'qlora_r': 8,
    'qlora_alpha': 16,
    'qlora_dropout': 0.05
})

# Quantize the weights
model.quantize_weights()
```

### Configuration Files

Example configuration files are provided:

- `config/finetune_gpt2_qlora.py`: QLoRA configuration
- `config/finetune_gpt2_qsinglora.py`: QSingleLoRA configuration

### Running Training

```bash
# Train with QLoRA
python train.py config/finetune_gpt2_qlora.py

# Train with QSingleLoRA
python train.py config/finetune_gpt2_qsinglora.py
```

## Implementation Details

### Quantization Process

1. **Weight Loading**: Original weights are loaded from pretrained models
2. **Block-wise Quantization**: Weights are divided into blocks and quantized to 4 bits
3. **Scale and Zero-point Storage**: Quantization parameters are stored for dequantization
4. **Dynamic Dequantization**: Weights are dequantized on-the-fly during forward passes

### Memory Benefits

- **4-bit quantization**: ~75% reduction in model weight memory
- **LoRA adaptation**: Only adapter parameters are trained, further reducing memory
- **Combined effect**: Significant memory savings while maintaining training performance

### Performance Considerations

- **Forward pass overhead**: Dequantization adds computational overhead
- **Block size**: Smaller blocks provide better precision but increase memory usage for storing quantization parameters
- **Bits**: 4-bit quantization provides good balance between memory savings and precision

## Differences from Standard LoRA

1. **Quantization**: Base weights are quantized, reducing memory footprint
2. **Dequantization**: Dynamic dequantization during forward passes
3. **Storage**: Additional storage for quantization parameters (scales and zero-points)
4. **Compatibility**: Drop-in replacement for standard LoRA with additional quantization parameters

## Testing

Run the test script to verify implementations:

```bash
python test_qlora.py
```

This will test:
- QLoRA functionality
- QSingleLoRA functionality
- Mutual exclusivity between different LoRA variants

## Notes

- Only one LoRA variant (LoRA, SingLoRA, QLoRA, QSingleLoRA) can be enabled at a time
- Quantization is applied after loading pretrained weights
- The implementations are designed to be compatible with existing training and inference pipelines
