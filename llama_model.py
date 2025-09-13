import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from model import LoRALinear, SingLoRALinear, QLoRALinear, QSingLoRALinear


def _replace_linear(module: nn.Module,
                    lora_r: int, lora_alpha: int, lora_dropout: float,
                    singlora_r: int, singlora_alpha: int, singlora_dropout: float,
                    qlora_r: int, qlora_alpha: int, qlora_dropout: float, qlora_bits: int, qlora_blocksize: int,
                    qsinglora_r: int, qsinglora_alpha: int, qsinglora_dropout: float, qsinglora_bits: int, qsinglora_blocksize: int):
    """Recursively replace nn.Linear layers with LoRA/SingLoRA/QLoRA/QSingLoRA versions."""

    # Check mutual exclusivity
    lora_variants = [lora_r > 0, singlora_r > 0, qlora_r > 0, qsinglora_r > 0]
    if sum(lora_variants) > 1:
        raise ValueError('Only one LoRA variant can be enabled at a time')

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            bias = child.bias is not None

            # Determine which variant to use
            if qlora_r > 0:
                new_module = QLoRALinear(child.in_features, child.out_features,
                                        r=qlora_r, lora_alpha=qlora_alpha,
                                        lora_dropout=qlora_dropout, bias=bias,
                                        bits=qlora_bits, blocksize=qlora_blocksize)
                # Copy weights and quantize
                new_module.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_module.bias.data = child.bias.data.clone()
                new_module.quantize_weight()  # Quantize after copying weights

            elif qsinglora_r > 0:
                new_module = QSingLoRALinear(child.in_features, child.out_features,
                                            r=qsinglora_r, alpha=qsinglora_alpha,
                                            dropout=qsinglora_dropout, bias=bias,
                                            bits=qsinglora_bits, blocksize=qsinglora_blocksize)
                # Copy weights and quantize
                new_module.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_module.bias.data = child.bias.data.clone()
                new_module.quantize_weight()  # Quantize after copying weights

            elif singlora_r > 0:
                new_module = SingLoRALinear(child.in_features, child.out_features,
                                            r=singlora_r, alpha=singlora_alpha,
                                            dropout=singlora_dropout, bias=bias)
                new_module.weight = child.weight
                if child.bias is not None:
                    new_module.bias = child.bias

            else:  # Standard LoRA or no LoRA
                new_module = LoRALinear(child.in_features, child.out_features,
                                        r=lora_r, lora_alpha=lora_alpha,
                                        lora_dropout=lora_dropout, bias=bias)
                new_module.weight = child.weight
                if child.bias is not None:
                    new_module.bias = child.bias

            new_module = new_module.to(child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, new_module)
        else:
            _replace_linear(child, lora_r, lora_alpha, lora_dropout,
                            singlora_r, singlora_alpha, singlora_dropout,
                            qlora_r, qlora_alpha, qlora_dropout, qlora_bits, qlora_blocksize,
                            qsinglora_r, qsinglora_alpha, qsinglora_dropout, qsinglora_bits, qsinglora_blocksize)


def load_pretrained_llama(model_name: str,
                           lora_r: int = 0,
                           lora_alpha: int = 1,
                           lora_dropout: float = 0.0,
                           singlora_r: int = 0,
                           singlora_alpha: int = 1,
                           singlora_dropout: float = 0.0,
                           qlora_r: int = 0,
                           qlora_alpha: int = 1,
                           qlora_dropout: float = 0.0,
                           qlora_bits: int = 4,
                           qlora_blocksize: int = 64,
                           qsinglora_r: int = 0,
                           qsinglora_alpha: int = 1,
                           qsinglora_dropout: float = 0.0,
                           qsinglora_bits: int = 4,
                           qsinglora_blocksize: int = 64,
                           device: str = 'cuda',
                           dtype: str = 'bfloat16'):
    """Load a pretrained Llama model and optionally enable LoRA/SingLoRA/QLoRA/QSingLoRA."""
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype))
    model.to(device)

    # Check if any LoRA variant is enabled
    has_lora = any([lora_r > 0, singlora_r > 0, qlora_r > 0, qsinglora_r > 0])
    if has_lora:
        _replace_linear(model, lora_r, lora_alpha, lora_dropout,
                        singlora_r, singlora_alpha, singlora_dropout,
                        qlora_r, qlora_alpha, qlora_dropout, qlora_bits, qlora_blocksize,
                        qsinglora_r, qsinglora_alpha, qsinglora_dropout, qsinglora_bits, qsinglora_blocksize)
    return model
