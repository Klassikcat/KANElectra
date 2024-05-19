import unittest 
import torch
import sys
sys.path.append('../')
from src.modules import *


class ModuleTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hidden_dim = 512
        cls.num_labels = 2
        cls.dim = 512
        cls.num_heads = 8
        cls.num_layers = 12
        cls.max_len = 512
        cls.input_ids = torch.randint(0, 100, (1, 512))
        cls.attention_mask = torch.ones_like(cls.input_ids)
        cls.token_type_ids = torch.zeros_like(cls.input_ids)
        
    def test_SelfAttention(self):
        attention = SelfAttention(self.dim, self.num_heads)
    