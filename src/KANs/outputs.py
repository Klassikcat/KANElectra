import torch
from torch import (
    nn,
    Tensor,
    FloatTensor
)
from transformers import PretrainedConfig
from kan import KANLayer
    

class ElectraKANAttentionOutput(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        
    def forward(self, hidden_states: torch.Tensor, attention_output: torch.Tensor) -> torch.Tensor:
        pass 