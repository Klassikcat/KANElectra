import unittest
import torch
import sys
sys.path.append('../')
from src.ElectraKAN.modules import *


class TestModules(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_dim = 128
        self.vocab_type_size = 20000
        self.layernorm_eps = 1e-6
        self.embedding_dropout_p = 0.1
        self.hidden_dim = 256
        self.num_heads = 4
        self.ff_dim = 512
        self.num_layers = 3
        self.max_pos_embedding = 512
        self.num_labels = 2
        self.dim = 512
        self.max_len = 100
        self.input_ids = torch.randint(0, 10000, (8, self.dim)).long()
        self.attn_mask = torch.randint(0, 1, (8, self.dim)).long()
        self.token_type_ids = torch.randint(0, 1, (8, self.dim)).long()

    def test_ElectraGenerator(self):
        model = ElectraGenerator(
            self.vocab_size,
            self.embedding_dim,
            self.vocab_type_size,
            self.layernorm_eps,
            self.embedding_dropout_p,
            self.hidden_dim,
            self.num_heads,
            self.ff_dim,
            self.num_layers,
            self.max_pos_embedding
        )
        output = model(
            input_ids=self.input_ids, 
            attention_mask=self.attn_mask, 
            token_type_ids=self.token_type_ids
            )

    def test_ElectraDiscriminator(self):
        model = ElectraDiscriminator(
            self.vocab_size,
            self.embedding_dim,
            self.vocab_type_size,
            self.layernorm_eps,
            self.embedding_dropout_p,
            self.hidden_dim,
            self.num_heads,
            self.ff_dim,
            self.num_layers,
            self.max_pos_embedding,
            self.num_labels
        )
        output = model(
            input_ids=self.input_ids, 
            attention_mask=self.attn_mask, 
            token_type_ids=self.token_type_ids
            )

    def test_ElectraEncoder(self):
        model = ElectraEncoder(
            self.dim,
            self.num_heads,
            self.hidden_dim,
            self.num_layers,
            self.max_len
        )
        output = model(
            input_ids=self.input_ids, 
            attention_mask=self.attn_mask, 
            token_type_ids=self.token_type_ids
            )

    # Add tests for other classes...

if __name__ == '__main__':
    unittest.main()