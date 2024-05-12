from typing import *
import torch
from torch import nn, Tensor, FloatTensor
from transformers import PretrainedConfig
from kan import KANLayer


class ElectraKANAttentionBlock(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Tuple[torch.Tensor]:
        pass
    
    

class ElectraKANIntermediate(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass
    
   
class ElectraKANSelfAttention(nn.Module):
    def __init__(
        self, 
        config: PretrainedConfig, 
        position_embedding_type: Optional[str] = None
        ) -> None:
        super().__init__()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Tuple[torch.Tensor]:
        pass
    
    