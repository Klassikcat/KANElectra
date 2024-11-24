from typing import *
import math
import torch
from torch import (
    nn,
    Tensor,
    FloatTensor,
    LongTensor
)
from .kan import KAN
import torch.nn.functional as F


class ElectraGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        vocab_type_size: int,
        embedding_dropout_p: float,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        max_pos_embedding: int
    ):
        super().__init__()
        self.embedding = InputEmbedding(
            vocab_size,
            embedding_dim,
            vocab_type_size,
            embedding_dropout_p,
            max_pos_embedding
        )
        self.encoder = ElectraEncoder(
            hidden_dim,
            num_heads,
            num_layers,
            0.1,
            ff_dim
        )
        self.generator = GeneratorOutput(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ) -> Tensor:
        embeddings = self.embedding(input_ids, token_type_ids)
        seq_out = self.encoder(embeddings, attention_mask)
        dropouted_seq_output = F.dropout(seq_out, p=0.1)
        return self.generator(dropouted_seq_output)


class GeneratorOutput(nn.Module):
    def __init__(self, hidden, vocab_size) :
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, x) :
        return self.softmax(self.linear(x))


class ElectraDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        vocab_type_size: int,
        embedding_dropout_p: float,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        max_pos_embedding: int,
        num_labels: int
    ):
        super().__init__()
        self.embedding = InputEmbedding(
            vocab_size,
            embedding_dim,
            vocab_type_size,
            embedding_dropout_p,
            max_pos_embedding
        )
        self.encoder = ElectraEncoder(
            hidden_dim,
            num_heads,
            num_layers,
            0.1,
            ff_dim
        )
        self.classifier = KAN(width=[hidden_dim, num_labels])

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ) -> Tensor:
        embeddings = self.embedding(input_ids, token_type_ids)
        seq_out = self.encoder(embeddings, attention_mask)
        dropouted_seq_output = F.dropout(seq_out, p=0.1)
        return self.classifier(dropouted_seq_output)


class ElectraEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        dropout_p: float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if not hidden_dim:
            hidden_dim = dim * 4 # default hidden_dim on paper
        self.layers = nn.ModuleList([
            EncoderLayer(dim, num_heads, hidden_dim, dropout_p) for i in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: Tensor,
        mask: Tensor
    ) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return hidden_states


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        vocab_type_size: int,
        embedding_dropout_p: float,
        max_pos_embedding: int
        ):
       super().__init__()
       self.embedding = nn.Embedding(vocab_size, embedding_dim)
       self.positional_embedding = nn.Embedding(max_pos_embedding, embedding_dim)
       self.token_type_embedding = nn.Embedding(vocab_type_size, embedding_dim)
       self.dropout = nn.Dropout(embedding_dropout_p)

    def forward(
        self,
        input_ids: LongTensor,
        token_type_ids: LongTensor,
    ) -> Tensor:
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embeddings = (
            self.embedding(input_ids) +
            self.positional_embedding(position_ids) +
            self.token_type_embedding(token_type_ids)
        )
        return self.dropout(embeddings)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p: float):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[LongTensor] = None
    ) -> Tensor:
        batch_size, n_head, length, d_tensor = query.shape
        multiplied_kv = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(d_tensor)
        if attention_mask is not None:
            broadcased_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            masked_attention = multiplied_kv.masked_fill(broadcased_attention_mask == 0, -1e9)
        else:
            masked_attention = multiplied_kv
        attention = self.softmax(masked_attention)
        return torch.matmul(attention, value)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_p: float
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.attention = ScaledDotProductAttention(dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        self.fc_q = KAN(width=[dim, dim])
        self.fc_k = KAN(width=[dim, dim])
        self.fc_v = KAN(width=[dim, dim])
        self.fc_out = KAN(width=[dim, dim])
        self.num_heads = num_heads
        self.dim = dim

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: LongTensor
    ) -> Tensor:
        batch_size = query.size(0)
        length = query.size(1)
        dim = query.size(2)
        d_tensor = dim // self.num_heads

        # split
        query = self.fc_q(query).view(batch_size, self.num_heads, length, d_tensor)
        key = self.fc_k(key).view(batch_size, self.num_heads, length, d_tensor)
        value = self.fc_v(value).view(batch_size, self.num_heads, length, d_tensor)
        attention_output = self.attention(query, key, value, attention_mask)
        # concat
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * (self.dim // self.num_heads))
        output = self.fc_out(attention_output)
        return self.dropout(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        dropout_p: float
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout_p: float
    ):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout_p)
        self.ff = FeedForward(dim, hidden_dim, dropout_p)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: Tensor,
        attention_mask: LongTensor
    ) -> Tensor:
        attention_output = self.attn(x, x, x, attention_mask)
        add_norm = self.norm1(x + attention_output)
        output = self.ff(attention_output)
        ff_add_norm = self.norm2(add_norm + output)
        return self.dropout(ff_add_norm)
