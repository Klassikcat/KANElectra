from typing import *
import torch
from torch import (
    nn, 
    Tensor, 
    FloatTensor, 
    LongTensor
)
from kan import KANLayer
import torch.nn.functional as F


class ElectraGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        vocab_type_size: int,
        layernorm_eps: float,
        embedding_dropout_p: float,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        max_pos_embedding: int = 512,
    ) -> None:
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.embedding_layer = Embedding(vocab_size, embedding_dim, max_pos_embedding, vocab_type_size, layernorm_eps, embedding_dropout_p)
        self.head = GeneratorHead(hidden_dim, embedding_dim, vocab_size, layernorm_eps)
        
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
        hidden_states = self.embedding_layer(input_ids, attention_mask, token_type_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.head(hidden_states)


class ElectraDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        vocab_type_size: int,
        layernorm_eps: float,
        embedding_dropout_p: float,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        max_pos_embedding: int = 512,
        num_labels: int = 1 
    ):
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.embedding_layer = Embedding(vocab_size, embedding_dim, max_pos_embedding, vocab_type_size, layernorm_eps, embedding_dropout_p)
        self.head = Classifier(hidden_dim, embedding_dim, num_labels, layernorm_eps)
        
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
        hidden_states = self.embedding_layer(input_ids, attention_mask, token_type_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.head(hidden_states)
    

class GeneratorHead(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, vocab_size: int, eps: float) -> None:
        super().__init__()
        self.kan = KANLayer(hidden_dim, embedding_dim)
        self.out = KANLayer(embedding_dim, vocab_size)
        self.eps = eps
        
    def forward(self, hidden: FloatTensor) -> FloatTensor:
        hidden = self.kan(hidden)
        hidden = F.gelu(hidden)
        hidden = F.layer_norm(hidden, eps=self.eps)
        return self.out(hidden)
        

class Classifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
    ):
        self.kan = KANLayer(hidden_dim, hidden_dim)
        self.out = KANLayer(hidden_dim, num_labels)
        
    def forward(
        self,
        hidden: FloatTensor
    ):
        hidden = self.kan(hidden)
        hidden = F.gelu(hidden)
        return self.out(hidden).squeeze(-1)


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
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

    
class EncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ff = PositionWideFeedForward(dim, ff_dim)
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
    def __init__(self, dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.fc1 = KANLayer(dim, intermediate_dim)
        self.fc2 = KANLayer(intermediate_dim, dim)
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
    
    
class Embedding(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        max_pos_embedding: int, 
        vocab_type_size: Optional[int] = 2,
        eps: Optional[float] = 1e-12,
        dropout_p: Optional[float] = .1,
        positional_embedding_type: str = 'absolute'
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)  
        self.pos_embedding = nn.Embedding(max_pos_embedding, embedding_dim)
        self.token_type_embedding = nn.Embedding(vocab_type_size, embedding_dim)
        
        self.layernorm = nn.LayerNorm(embedding_dim, eps=eps)
        self.dropout_p = dropout_p
        self.positional_embedding_type = positional_embedding_type
        
    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ) -> FloatTensor:
        input_embedding = self.word_embedding(input_ids)
        token_type_embedding = self.token_type_embedding(attention_mask)
        embedding = input_embedding + token_type_embedding
        if self.positional_embedding_type in ['absolute', 'abs']:
            pos_embedding = self.pos_embedding(attention_mask) 
            embedding += pos_embedding
        embedding = self.layernorm(embedding)
        embedding = F.dropout(embedding, p=self.dropout_p)
        return embedding