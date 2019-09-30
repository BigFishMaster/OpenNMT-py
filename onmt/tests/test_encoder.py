import unittest
import torch
import numpy as np
from onmt.encoders.egat_encoder import EvolutionaryGAT as eGAT
torch.random.manual_seed(123)


class Opt(object):
    def __init__(self):
       self.egat_max_seq_len = 6
       self.egat_input_size = 10
       self.egat_hidden_size = 20
       self.egat_output_size = 10
       self.egat_dropout = 0.5
       self.egat_alpha = 0.2
       self.egat_heads = 4
       self.tran_num_layers = 2
       self.train_d_model = 10
       self.tran_heads = 2
       self.tran_d_ff = 40
       self.tran_dropout = 0.3
       self.tran_attention_dropout = 0.4
       self.tran_max_relation_position = 0
       self.tran_vocab_size = 10

class Batch():
    steps = 5
    seq_len = 6
    batch_size = 8
    feat_dim = 10
    src = torch.rand(seq_len, batch_size, steps, feat_dim).float()
    lengths = torch.Tensor([5, 5, 5, 4, 4, 3, 3, 2]).long()
    t = torch.Tensor([2,2,1,3,1,2,1,1]).long()
    adj = torch.randint(0, 2, [8, 6, 6])
    egat = [src, lengths, adj, t]
    tran_src = torch.randint(0, 10, [5, 8, 1])
    # note: max_len must be the same as sequence length.
    tran_lengths = torch.Tensor([3,4,5,1,3,2,3,2]).long()
    tran = [tran_src, tran_lengths]


class TestEncoders(unittest.TestCase):
    def test_egat_encoder(self):
        opt = Opt()
        kwargs = {
          "egat": [
            opt.egat_max_seq_len,
            opt.egat_input_size,
            opt.egat_hidden_size,
            opt.egat_output_size,
            opt.egat_dropout,
            opt.egat_alpha,
            opt.egat_heads],
          "transformer": [
            opt.tran_num_layers,
            opt.train_d_model,
            opt.tran_heads,
            opt.tran_d_ff,
            opt.tran_dropout,
            opt.tran_attention_dropout,
            opt.tran_max_relation_position,
            opt.tran_vocab_size
          ]
        }
        print(kwargs)
        model = eGAT(kwargs)
        batch = Batch()
        model(batch)
