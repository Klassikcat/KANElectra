import unittest
import torch
import sys
sys.path.append('../')
from src.ElectraKAN.modules import *


class TestModules(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_dim = 128
        self.vocab_type_size = 2
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
        input_ids = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        output = model(input_ids)

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
        input_ids = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        output = model(input_ids)

    def test_ElectraEncoder(self):
        model = ElectraEncoder(
            self.dim,
            self.num_heads,
            self.hidden_dim,
            self.num_layers,
            self.max_len
        )
        input_ids = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
        output = model(input_ids)

    # Add tests for other classes...

if __name__ == '__main__':
    unittest.main()