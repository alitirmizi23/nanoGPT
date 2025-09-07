import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from model import LoRALinear, SingLoRALinear


def _replace_linear(module: nn.Module,
                    lora_r: int, lora_alpha: int, lora_dropout: float,
                    singlora_r: int, singlora_alpha: int, singlora_dropout: float):
    """Recursively replace nn.Linear layers with LoRA/SingLoRA versions."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            bias = child.bias is not None
            if singlora_r > 0:
                new_module = SingLoRALinear(child.in_features, child.out_features,
                                            r=singlora_r, alpha=singlora_alpha,
                                            dropout=singlora_dropout, bias=bias)
            else:
                new_module = LoRALinear(child.in_features, child.out_features,
                                        r=lora_r, lora_alpha=lora_alpha,
                                        lora_dropout=lora_dropout, bias=bias)
            new_module = new_module.to(child.weight.device, dtype=child.weight.dtype)
            new_module.weight = child.weight
            if child.bias is not None:
                new_module.bias = child.bias
            setattr(module, name, new_module)
        else:
            _replace_linear(child, lora_r, lora_alpha, lora_dropout,
                            singlora_r, singlora_alpha, singlora_dropout)


def load_pretrained_llama(model_name: str,
                           lora_r: int = 0,
                           lora_alpha: int = 1,
                           lora_dropout: float = 0.0,
                           singlora_r: int = 0,
                           singlora_alpha: int = 1,
                           singlora_dropout: float = 0.0,
                           device: str = 'cuda',
                           dtype: str = 'bfloat16'):
    """Load a pretrained Llama model and optionally enable LoRA/SingLoRA."""
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype))
    model.to(device)
    if lora_r > 0 or singlora_r > 0:
        _replace_linear(model, lora_r, lora_alpha, lora_dropout,
                        singlora_r, singlora_alpha, singlora_dropout)
    return model
