import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.encoders.gat_encoder import GAT
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.modules import Embeddings, VecEmbedding


class EvolutionaryGAT(nn.Module):
    def __init__(self, kwargs):
        super(EvolutionaryGAT, self).__init__()
        egat_param = kwargs["egat"]
        transformer_param = kwargs["transformer"]
        self.init_egat(*egat_param)
        self.init_transformer(*transformer_param)

    def init_egat(self, max_seq_len, input_size, hidden_size, output_size,
                 dropout, alpha, heads):
        print(max_seq_len, input_size, hidden_size, output_size, dropout, alpha, heads)
        self.egat_hidden_size = hidden_size
        self.egat_embeddings = VecEmbedding(
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
                                      embeddings=self.egat_embeddings,
                                      use_bridge=False) for _ in range(max_seq_len)]
        self.rnn_encoders = nn.ModuleList(rnn_encoders)
        assert hidden_size % heads == 0, \
            "hidden_size(%s) must be divided by heads(%s)." % (hidden_size, heads)
        each_head_size = int(hidden_size / heads)
        print(each_head_size)
        self.gat_encoder = GAT(hidden_size, each_head_size, output_size,
                               dropout, alpha, heads)

    def init_transformer(self, num_layers, d_model, heads, d_ff, dropout,
                         attention_dropout, max_relative_positions, vocab_size):
        self.tran_embeddings = Embeddings(
            word_vec_size=d_model,
            position_encoding=True,
            feat_merge="concat",
            feat_vec_exponent=0.7,
            feat_vec_size=-1,
            dropout=dropout,
            word_padding_idx=1,
            feat_padding_idx=[],
            word_vocab_size=vocab_size,
            feat_vocab_sizes=[],
            sparse=False,
            fix_word_vecs=False
        )
        self.tran_encoder = TransformerEncoder(num_layers, d_model,
            heads, d_ff, dropout, attention_dropout,
            self.tran_embeddings, max_relative_positions)

    @classmethod
    def from_opt(cls, opt):
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
        return cls(kwargs)

    def forward(self, batch):
        # batch: torchtext.data.batch
        egat_src, egat_lengths, egat_adj, egat_t = batch.egat
        tran_src, tran_lengths = batch.tran
        _, egat_out, _ = self.forward_egat(egat_src, egat_lengths, egat_adj, egat_t)
        _, tran_out, _ = self.tran_encoder(tran_src, tran_lengths)
        out = torch.cat([egat_out, tran_out], 0)
        out_lengths = egat_lengths + tran_lengths
        print("egat_out:", egat_out.size())
        print("tran_out:", tran_out.size())
        print("final out:", out.size())
        return out, out, out_lengths


    def forward_egat(self, src, lengths, adj, t):
        """
        :param src: seq_len x batch x steps x dims
        :param lengths: batch, LongTensor
        :param adj:
        :param t: batch, LongTensor
        :return:
        """
        # shape: seq_len, batch, hidden_size,
        t = t.view(-1, 1, 1).repeat(1, 1, self.egat_hidden_size)
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

