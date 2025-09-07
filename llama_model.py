"""Implementation of LLaMA 3.1 models with optional LoRA and SingLoRA adapters."""

from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from lora import LoRALinear, SingLoRALinear


@dataclass
class LlamaConfig:
    model_name: str
    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    single_lora: bool = False


PRETRAINED_LLAMA3_1 = {
    "8B": "meta-llama/Meta-Llama-3.1-8B",
    "70B": "meta-llama/Meta-Llama-3.1-70B",
}


class Llama3(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaForCausalLM.from_pretrained(config.model_name)
        if config.lora_r > 0 or config.single_lora:
            self._inject_lora()

    def _factory(self, in_features, out_features, bias=True):
        if self.config.single_lora:
            return SingLoRALinear(
                in_features,
                out_features,
                r=self.config.lora_r,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
                bias=bias,
            )
        else:
            return LoRALinear(
                in_features,
                out_features,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=bias,
            )

    def _convert_linear(self, module: nn.Linear) -> nn.Module:
        new_linear = self._factory(
            module.in_features, module.out_features, bias=module.bias is not None
        )
        new_linear.weight = module.weight
        if module.bias is not None:
            new_linear.bias = module.bias
        return new_linear

    def _inject_lora(self):
        for layer in self.model.model.layers:
            layer.self_attn.q_proj = self._convert_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self._convert_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self._convert_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self._convert_linear(layer.self_attn.o_proj)
            layer.mlp.gate_proj = self._convert_linear(layer.mlp.gate_proj)
            layer.mlp.up_proj = self._convert_linear(layer.mlp.up_proj)
            layer.mlp.down_proj = self._convert_linear(layer.mlp.down_proj)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    @classmethod
    def from_pretrained(cls, size: str = "8B", **kwargs):
        model_name = PRETRAINED_LLAMA3_1[size]
        config = LlamaConfig(model_name=model_name, **kwargs)
        return cls(config)
