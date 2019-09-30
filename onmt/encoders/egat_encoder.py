import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.encoders.gat_encoder import GAT
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.modules import VecEmbedding


class EvolutionaryGAT(nn.Module):
    def __init__(self, max_seq_len, input_size, hidden_size, output_size,
                 dropout, alpha, heads):
        super(EvolutionaryGAT, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings = VecEmbedding(
            input_size,
            emb_dim=hidden_size,
            position_encoding=False,
            dropout=dropout
        )
        rnn_encoders = [RNNEncoder(rnn_type="LSTM",
                                      bidirectional=True,
                                      num_layers=2,
                                      hidden_size=hidden_size,
                                      dropout=dropout,
                                      embeddings=self.embeddings,
                                      use_bridge=False) for _ in range(max_seq_len)]
        self.rnn_encoders = nn.ModuleList(rnn_encoders)
        assert hidden_size % heads == 0, \
            "hidden_size(%s) must be divided by heads(%s)." % (hidden_size, heads)
        each_head_size = int(hidden_size / heads)
        print(each_head_size)
        self.gat_encoder = GAT(hidden_size, each_head_size, output_size,
                               dropout, alpha, heads)

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.egat_max_seq_len,
            opt.egat_input_size,
            opt.egat_hidden_size,
            opt.egat_output_size,
            opt.egat_dropout,
            opt.egat_alpha,
            opt.egat_heads
        )

    def forward(self, src, lengths, adj, t):
        """
        :param src: seq_len x batch x steps x dims
        :param lengths: batch, LongTensor
        :param adj:
        :param t: batch, LongTensor
        :return:
        """
        # shape: seq_len, batch, hidden_size,
        t = t.view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        # shape: seq_len x steps x batch x dims
        src = src.transpose(1, 2)
        print("t:", t.size())
        print("src:", src.size())
        mbs = []
        seq_len, batch, steps, dims = src.size()
        for i in range(seq_len):
            # shape: steps x batch x hidden_size
            _, mb, _ = self.rnn_encoders[i](src[i, :, :, :])
            # shape: batch x steps x hidden_size
            mb = mb.transpose(0, 1).contiguous()
            # batch_size x 1 x hidden_size
            mb = torch.gather(mb, 1, t)
            mbs.append(mb)
        # shape: batch_size x seq_len x hidden_size
        memory_bank = torch.cat(mbs, 1)
        print("memory_bank:", memory_bank.size())
        out = self.gat_encoder(memory_bank, adj)
        return memory_bank.transpose(0, 1), out.transpose(0, 1).contiguous(), lengths

