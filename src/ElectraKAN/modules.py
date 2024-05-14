from typing import *
import torch
from torch import nn, Tensor, FloatTensor, LongTensor
from kan import KANLayer
from .outputs import ElectraKANSelfOutput


class ElectraEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        max_len: int
    ) -> None:
        self.layers = nn.ModuleList([
            EncoderLayer(dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.input_ids_embedding = PositionalEncoding(dim, max_len)
        self.pos_embedding = PositionalEncoding(dim, max_len)
        self.token_type_ids_embedding = PositionalEncoding(dim, max_len)
        
    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: Optional[LongTensor] = None,
        token_type_ids: Optional[LongTensor] = None
    ):
        if not attention_mask:
            attention_mask = torch.ones_like(input_ids)
        if not token_type_ids:
            token_type_ids = torch.zeros_like(input_ids)
        hidden_states = self.pos_enc(input_ids)
        


    
class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ff = PositionWideFeedForward(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(
        self, 
        x: FloatTensor, 
        mask: Optional[Tensor] = None
        ) -> FloatTensor:
        x = self.attn(x, x, x, mask) + x
        x = self.norm1(x)
        x = self.ff(x) + x
        return self.norm2(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = KANLayer(dim, dim)
        self.key = KANLayer(dim, dim)
        self.value = KANLayer(dim, dim)
        self.out = KANLayer(dim, dim)
        
    def scaled_dot_production_attn(self, query: FloatTensor, key: FloatTensor, value: FloatTensor, mask: Optional[Tensor] = None) -> Tuple[FloatTensor, FloatTensor]:
        scores = query @ key.transpose(-2, -1) * self.scale
        if mask is not None:
            scores.masked_fill_(mask, float('-inf'))
        attn = scores.softmax(dim=-1)
        return attn @ value, attn   # Thanks copilot!
    
    def split_heads(self, x: FloatTensor) -> Tensor:
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # Thanks copilot! - 2
    
    def combine_heads(self, x: FloatTensor) -> Tensor:
        return x.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.dim)  # Thanks copilot!
    
    def forward(
        self, 
        query: FloatTensor, 
        key: FloatTensor, 
        value: FloatTensor, 
        mask: Optional[Tensor] = None
        ) -> FloatTensor:
        query = self.split_heads(self.query(query))
        key = self.split_heads(self.key(key))
        value = self.split_heads(self.value(value))
        
        x, attn = self.scaled_dot_production_attn(query, key, value, mask)
        
        return self.out(self.combine_heads(x))


class PositionWideFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = KANLayer(dim, hidden_dim)
        self.fc2 = KANLayer(hidden_dim, dim)
        self.activation = nn.ReLU()
        
    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.fc2(self.activation(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros(max_len, dim))
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        self.pos_enc[:, 0::2] = torch.sin(pos * div)
        self.pos_enc[:, 1::2] = torch.cos(pos * div)
        
        self.register_buffer('pos_enc', self.pos_enc)
        
    def forward(self, x: FloatTensor) -> FloatTensor:
        return x + self.pos_enc[:x.size(1)]