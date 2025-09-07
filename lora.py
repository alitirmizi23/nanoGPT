import math
import torch
import torch.nn as nn

class LoRALinear(nn.Linear):
    """Linear layer with Low-Rank Adaptation (LoRA)."""

    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0.0, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.r = r
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = lora_alpha / r
            self.lora_dropout = nn.Dropout(lora_dropout)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    def forward(self, x):
        result = super().forward(x)
        if self.r > 0:
            result = result + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result

class SingLoRALinear(nn.Linear):
    """Linear layer with Single-matrix LoRA (SingLoRA)."""

    def __init__(self, in_features, out_features, r=0, alpha=1, dropout=0.0, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        if r > 0:
            dim = max(in_features, out_features)
            self.singlora_A = nn.Parameter(torch.empty(dim, r))
            nn.init.kaiming_uniform_(self.singlora_A, a=math.sqrt(5))
            self.scaling = alpha / r
            self.singlora_dropout = nn.Dropout(dropout)
        else:
            self.singlora_A = None
            self.singlora_dropout = nn.Identity()

    def forward(self, x):
        result = super().forward(x)
        if self.r > 0:
            A_in = self.singlora_A[: self.in_features]
            A_out = self.singlora_A[: self.out_features]
            result = result + (self.singlora_dropout(x) @ A_in) @ A_out.t() * self.scaling
        return result
