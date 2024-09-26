#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""rnn: RNN layers.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import math

# Third party's modules
import numpy as np

import torch
from torch import nn

# Local modules
from .misc import (
    Identity,
    GPoolRecognitionHead,
    TemporalAttention)

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class RNNEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 rnn_type,
                 num_layers,
                 activation,
                 bidir,
                 dropout,
                 apply_mask,
                 proj_size=0):
        super().__init__()
        assert rnn_type in ["srnn", "lstm", "gru"]

        if rnn_type == "srnn":
            self.rnn = nn.RNN(input_size=in_channels,
                              hidden_size=out_channels,
                              num_layers=num_layers,
                              nonlinearity=activation,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=bidir)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=in_channels,
                               hidden_size=out_channels,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidir,
                               proj_size=proj_size)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=in_channels,
                              hidden_size=out_channels,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=bidir)
        self.num_layers = num_layers
        self.apply_mask = apply_mask

    def sep_state_layerwise(self, last_state):
        # `[D * num_layers, N, C] -> [N, D * num_layers, C]`
        last_state = last_state.permute(1, 0, 2)
        # `[N, D * num_layers, C] -> (num_layers, [N, D, C]) -> [N, D, C]`
        if self.num_layers > 1:
            last_state = torch.split(last_state, self.num_layers, dim=1)
        else:
            last_state = (last_state,)
        return last_state

    def forward(self, feature, feature_pad_mask=None):
        if feature_pad_mask is not None and self.apply_mask:
            tlength = feature_pad_mask.sum(axis=-1).detach().cpu()
            feature = nn.utils.rnn.pack_padded_sequence(
                feature, tlength, batch_first=True, enforce_sorted=False)

        if isinstance(self.rnn, nn.LSTM):
            hidden_seqs, (last_hstate, last_cstate) = self.rnn(feature)
        else:
            hidden_seqs, last_hstate = self.rnn(feature)
            last_cstate = None
        # Unpack hidden sequence.
        if isinstance(hidden_seqs, nn.utils.rnn.PackedSequence):
            hidden_seqs = nn.utils.rnn.unpack_sequence(hidden_seqs)
            # Back list to padded batch.
            hidden_seqs = nn.utils.rnn.pad_sequence(
                hidden_seqs, batch_first=True, padding_value=0.0)

        return hidden_seqs, last_hstate, last_cstate


class RNNISLR(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 rnn_type="lstm",
                 rnn_num_layers=1,
                 rnn_activation="tanh",
                 rnn_bidir=False,
                 rnn_dropout=0.1,
                 apply_mask=True,
                 masking_type="both",
                 attention_type="none",
                 attention_post_scale=False,
                 head_type="gpool"):
        super().__init__()
        assert rnn_type in ["srnn", "lstm", "gru"]
        assert masking_type in ["none", "rnn", "head", "both"]
        assert attention_type in ["none", "sigmoid", "softmax"]
        assert head_type in ["gpool", "last_state"]

        self.linear = nn.Linear(in_channels, hidden_channels)
        self.activation = nn.ReLU()

        apply_mask = True if masking_type in ["rnn", "both"] else False
        self.rnn = RNNEncoder(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            rnn_type=rnn_type,
            num_layers=rnn_num_layers,
            activation=rnn_activation,
            bidir=rnn_bidir,
            dropout=rnn_dropout,
            apply_mask=apply_mask)

        if attention_type != "none":
            if rnn_bidir:
                self.att = TemporalAttention(hidden_channels * 2, attention_type,
                    post_scale=attention_post_scale)
            else:
                self.att = TemporalAttention(hidden_channels, attention_type,
                    post_scale=attention_post_scale)
        else:
            self.att = Identity()
        self.attw = None

        if rnn_bidir:
            self.head = GPoolRecognitionHead(hidden_channels * 2, out_channels)
        else:
            self.head = GPoolRecognitionHead(hidden_channels, out_channels)

        self.masking_type = masking_type
        self.head_type = head_type

    def forward(self, feature, feature_pad_mask=None):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        feature = self.linear(feature)
        feature = self.activation(feature)

        hidden_seqs, last_hstate = self.rnn(feature, feature_pad_mask)[:2]

        # Apply attention.
        hidden_seqs = self.att(hidden_seqs, feature_pad_mask)
        if isinstance(hidden_seqs, (tuple, list)):
            hidden_seqs, self.attw = hidden_seqs[0], hidden_seqs[1]

        if self.head_type == "gpool":
            # `[N, T, C'] -> [N, C', T]`
            feature = hidden_seqs.permute(0, 2, 1)
        else:  # "last_state"
            last_hstate = self.rnn.sep_state_layerwise(last_hstate)
            # `(num_layers, [N, D, C]) -> [N, D, C] -> [N, T(=1), D*C]`
            last_hstate = last_hstate[-1]
            feature = last_hstate.reshape([last_hstate.shape[0], 1, -1])
            # `[N, T, D*C] -> [N, D*C, T]`
            feature = feature.permute(0, 2, 1)

        if feature_pad_mask is not None and self.masking_type in ["head", "both"]:
            logit = self.head(feature, feature_pad_mask)
        else:
            logit = self.head(feature)
        return logit


class BahdanauAttentionEnergy(nn.Module):
    def __init__(self,
                 key_dim,
                 query_dim,
                 att_dim,
                 add_bias=False):
        super().__init__()

        self.w_key = nn.Linear(key_dim, att_dim, bias=add_bias)
        self.w_query = nn.Linear(query_dim, att_dim, bias=add_bias)
        self.w_out = nn.Linear(att_dim, 1, bias=add_bias)

    def forward(self, key, query):
        # print("key:", key.shape)
        # print("query:", query.shape)
        # key: `[N, key_len, key_dim]`
        # query: `[N, 1, query_dim]`
        key = self.w_key(key)
        query = self.w_query(query)
        # Adding with broadcasting.
        # key: `[N, key_len, key_dim]`
        # query: `[N, 1, query_dim]`
        # query should be broadcasted to `[N, key_len, query_dim]`
        temp = key + query
        # `[N, key_len, att_dim] -> [N, key_len, 1] -> [N, 1, key_len]`
        energy = self.w_out(torch.tanh(temp))
        energy = torch.permute(energy, [0, 2, 1])
        return energy


class LuongDotAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, key, query):
        # key: `[N, key_len, key_dim]`
        # query: `[N, 1, query_dim]`
        # key_dim == query_dim
        # bmm: `[n, a, b] x [n, b, c] -> [n, a, c]`
        # `[N, key_len, 1] -> [N, 1, key_len]`
        energy = torch.bmm(key, torch.permute(query, [0, 2, 1]))
        energy = torch.permute(energy, [0, 2, 1])
        return energy


class LuongGeneralAttention(nn.Module):
    def __init__(self,
                 key_dim,
                 query_dim,
                 add_bias=False):
        super().__init__()

        self.w_key = nn.Linear(key_dim, query_dim, bias=add_bias)

    def forward(self, key, query):
        key = self.w_key(key)
        # key: `[N, key_len, query_dim]`
        # query: `[N, 1, query_dim]`
        # bmm: `[n, a, b] x [n, b, c] -> [n, a, c]`
        # `[N, key_len, 1] -> [N, 1, key_len]`
        energy = torch.bmm(key, torch.permute(query, [0, 2, 1]))
        energy = torch.permute(energy, [0, 2, 1])
        return energy


class SingleHeadAttention(nn.Module):
    def __init__(self,
                 key_dim,
                 query_dim,
                 att_dim,
                 add_bias,
                 att_type):
        super().__init__()
        assert att_type in ["bahdanau", "luong_dot", "luong_general"]

        if att_type == "bahdanau":
            self.att_energy = BahdanauAttentionEnergy(
                key_dim=key_dim,
                query_dim=query_dim,
                att_dim=att_dim,
                add_bias=add_bias)
        elif att_type == "luong_dot":
            self.att_energy = LuongDotAttention()
        elif att_type == "luong_general":
            self.att_energy = LuongGeneralAttention(
                key_dim=key_dim,
                query_dim=query_dim,
                add_bias=add_bias)

        self.neg_inf = None

    def forward(self,
                key,
                value,
                query,
                mask=None):
        if self.neg_inf is None:
            self.neg_inf = float(np.finfo(
                torch.tensor(0, dtype=key.dtype).numpy().dtype).min)

        batch, klen, kdim = key.shape
        _, qlen, qdim = query.shape
        energy = self.att_energy(key=key, query=query)
        assert energy.shape == (batch, qlen, klen)

        # Apply mask.
        if mask is not None:
            if len(mask.shape) == 2:
                # `[N, klen] -> [N, qlen(=1), klen]`
                mask = mask.unsqueeze(1)
            # Negative infinity should be 0 in softmax.
            energy = energy.masked_fill_(mask==0, self.neg_inf)

        # Compute attention mask.
        attw = torch.softmax(energy, dim=-1)
        # attw: `[N, qlen, klen]`
        # value: `[N, klen, kdim]`
        # bmm: `[N, qlen, klen] x [N, klen, kdim] -> [N, qlen, kdim]`
        cvec = torch.bmm(attw, value)
        return cvec, attw


class BahdanauRNNDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 emb_channels,
                 att_type,
                 att_dim,
                 att_add_bias,
                 rnn_type,
                 num_layers,
                 activation,
                 dropout,
                 padding_val,
                 proj_size=0):
        super().__init__()
        assert rnn_type in ["srnn", "lstm", "gru"]

        self.emb_layer = nn.Embedding(
            num_embeddings=out_channels,
            embedding_dim=emb_channels,
            padding_idx=padding_val)

        self.att_layer = SingleHeadAttention(
            key_dim=in_channels,
            query_dim=hidden_channels,
            att_dim=att_dim,
            add_bias=att_add_bias,
            att_type=att_type)

        if rnn_type == "srnn":
            self.rnn = nn.RNN(input_size=in_channels + emb_channels,
                              hidden_size=hidden_channels,
                              num_layers=num_layers,
                              nonlinearity=activation,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=False)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=in_channels + emb_channels,
                               hidden_size=hidden_channels,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=False,
                               proj_size=proj_size)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=in_channels + emb_channels,
                              hidden_size=hidden_channels,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=False)
        self.head = nn.Linear(hidden_channels, out_channels)

        self.num_layers = num_layers
        self.dec_hstate = None
        self.attw = None

        self.reset_parameters(emb_channels, padding_val)

    def reset_parameters(self, embedding_dim, padding_val):
        # Bellow initialization has strong effect to performance.
        # Please refer.
        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py#L189
        nn.init.normal_(self.emb_layer.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(self.emb_layer.weight[padding_val], 0)

        # Please refer.
        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_decoder.py
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def init_dec_hstate(self, enc_hstate, init_as_zero=False):
        if init_as_zero:
            dec_hstate = torch.zeros_like(enc_hstate)
        else:
            dec_hstate = enc_hstate
        # To avoid error at RNN layer.
        self.dec_hstate = dec_hstate.contiguous()

    def forward(self,
                dec_inputs,
                enc_seqs,
                enc_mask):
        assert self.dec_hstate is not None, f"dec_hstate has not been initialized."
        dec_hstate = self.dec_hstate
        # print("dec_hstate:", dec_hstate.shape)

        # Attention layer requires hidden state of 2nd rnn layer.
        # as `[N, 1, C]`
        query = dec_hstate[-1].unsqueeze(1)
        cvec, self.attw = self.att_layer(
            key=enc_seqs,
            value=enc_seqs,
            query=query,
            mask=enc_mask)

        emb_out = self.emb_layer(dec_inputs)
        # `[N, C] -> [N, 1, C]`
        emb_out = emb_out.reshape([-1, 1, emb_out.shape[-1]])
        feature = torch.cat([cvec, emb_out], dim=-1)
        if isinstance(self.rnn, nn.LSTM):
            hidden_seqs, (last_hstate, last_cstate) = self.rnn(feature,
                                                               dec_hstate)
        else:
            hidden_seqs, last_hstate = self.rnn(feature,
                                                dec_hstate)
            last_cstate = None

        output_dec = self.head(hidden_seqs)
        self.dec_hstate = last_hstate
        return output_dec


class LuongRNNDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 emb_channels,
                 att_type,
                 att_dim,
                 att_add_bias,
                 rnn_type,
                 num_layers,
                 activation,
                 dropout,
                 padding_val,
                 proj_size=0):
        super().__init__()
        assert rnn_type in ["srnn", "lstm", "gru"]

        self.emb_layer = nn.Embedding(
            num_embeddings=out_channels,
            embedding_dim=emb_channels,
            padding_idx=padding_val)
        self.vocab_size = out_channels

        if rnn_type == "srnn":
            self.rnn = nn.RNN(input_size=emb_channels,
                              hidden_size=hidden_channels,
                              num_layers=num_layers,
                              nonlinearity=activation,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=False)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=emb_channels,
                               hidden_size=hidden_channels,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=False,
                               proj_size=proj_size)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=emb_channels,
                              hidden_size=hidden_channels,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=False)

        self.att_layer = SingleHeadAttention(
            key_dim=in_channels,
            query_dim=hidden_channels,
            att_dim=att_dim,
            add_bias=att_add_bias,
            att_type=att_type)

        self.head = nn.Linear(hidden_channels * 2,  # hstate + cvec
                              out_channels)

        self.num_layers = num_layers
        self.dec_hstate = None
        self.attw = None

        self.reset_parameters(emb_channels, padding_val)

    def reset_parameters(self, embedding_dim, padding_val):
        # Bellow initialization has strong effect to performance.
        # Please refer.
        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py#L189
        nn.init.normal_(self.emb_layer.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(self.emb_layer.weight[padding_val], 0)

        # Please refer.
        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_decoder.py
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def init_dec_hstate(self, enc_hstate, init_as_zero=False):
        if init_as_zero:
            dec_hstate = torch.zeros_like(enc_hstate)
        else:
            dec_hstate = enc_hstate
        # To avoid error at RNN layer.
        self.dec_hstate = dec_hstate.contiguous()

    def forward(self,
                dec_inputs,
                enc_seqs,
                enc_mask):

        assert self.dec_hstate is not None, f"dec_hstate has not been initialized."
        dec_hstate = self.dec_hstate

        emb_out = self.emb_layer(dec_inputs) * math.sqrt(self.vocab_size)
        if isinstance(self.rnn, nn.LSTM):
            hidden_seqs, (last_hstate, last_cstate) = self.rnn(emb_out,
                                                               dec_hstate)
        else:
            hidden_seqs, last_hstate = self.rnn(emb_out,
                                                dec_hstate)
            last_cstate = None

        # Attention layer requires hidden state of 2nd rnn layer.
        # as `[N, 1, C]`
        query = last_hstate[-1].unsqueeze(1)
        cvec, self.attw = self.att_layer(
            key=enc_seqs,
            value=enc_seqs,
            query=query,
            mask=enc_mask)

        # `[N, 1, C]`
        feature = torch.cat([cvec, hidden_seqs], dim=-1)

        output_dec = self.head(feature)
        self.dec_hstate = last_hstate
        return output_dec


class RNNCSLR(nn.Module):
    def __init__(self,
                 enc_in_channels,
                 enc_hidden_channels,
                 enc_rnn_type,
                 enc_num_layers,
                 enc_activation,
                 enc_bidir,
                 enc_dropout,
                 enc_apply_mask,
                 enc_proj_size,
                 dec_type,
                 dec_in_channels,
                 dec_hidden_channels,
                 dec_out_channels,
                 dec_emb_channels,
                 dec_att_type,
                 dec_att_dim,
                 dec_att_add_bias,
                 dec_rnn_type,
                 dec_num_layers,
                 dec_activation,
                 dec_dropout,
                 dec_padding_val,
                 dec_proj_size):
        super().__init__()
        assert dec_type in ["bahdanau", "luong"]
        self.enc_bidir = enc_bidir

        self.linear = nn.Linear(enc_in_channels, enc_hidden_channels)
        self.enc_activation = nn.ReLU()

        self.encoder = RNNEncoder(
            in_channels=enc_hidden_channels,
            out_channels=enc_hidden_channels,
            rnn_type=enc_rnn_type,
            num_layers=enc_num_layers,
            activation=enc_activation,
            bidir=enc_bidir,
            dropout=enc_dropout,
            apply_mask=enc_apply_mask,
            proj_size=enc_proj_size)

        if enc_bidir:
            dec_in_channels *= 2
            dec_hidden_channels *= 2
            dec_att_dim *= 2

        if dec_type == "bahdanau":
            self.decoder = BahdanauRNNDecoder(
                in_channels=dec_in_channels,
                hidden_channels=dec_hidden_channels,
                out_channels=dec_out_channels,
                emb_channels=dec_emb_channels,
                att_type=dec_att_type,
                att_dim=dec_att_dim,
                att_add_bias=dec_att_add_bias,
                rnn_type=dec_rnn_type,
                num_layers=dec_num_layers,
                activation=dec_activation,
                dropout=dec_dropout,
                padding_val=dec_padding_val,
                proj_size=dec_proj_size)
        elif dec_type == "luong":
            self.decoder = LuongRNNDecoder(
                in_channels=dec_in_channels,
                hidden_channels=dec_hidden_channels,
                out_channels=dec_out_channels,
                emb_channels=dec_emb_channels,
                att_type=dec_att_type,
                att_dim=dec_att_dim,
                att_add_bias=dec_att_add_bias,
                rnn_type=dec_rnn_type,
                num_layers=dec_num_layers,
                activation=dec_activation,
                dropout=dec_dropout,
                padding_val=dec_padding_val,
                proj_size=dec_proj_size)

        self.attws = None

    def _apply_encoder(self, feature, feature_pad_mask=None):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        # print("feature0:", feature.shape)
        feature = self.linear(feature)
        feature = self.enc_activation(feature)
        # print("feature1:", feature.shape)

        # Apply encoder.
        enc_seqs, enc_hstate = self.encoder(feature, feature_pad_mask)[:2]

        # Basically, decoder should not be bidirectional.
        # So, we should concatenate backwarded feature.
        if self.enc_bidir:
            # `[2*layers, N, C] -> [layers, N, 2*C]`
            enc_hstate = torch.permute(enc_hstate, [1, 0, 2])
            enc_hstate = enc_hstate.reshape([enc_hstate.shape[0],
                                             enc_hstate.shape[1] // 2,
                                             -1])
            enc_hstate = torch.permute(enc_hstate, [1, 0, 2])
        return enc_seqs, enc_hstate

    def forward(self,
                feature, tokens,
                feature_pad_mask=None, tokens_pad_mask=None):
        """Forward computation for train.
        """
        # print("feature:", feature.shape)
        # print("tokens:", tokens.shape)

        enc_seqs, enc_hstate = self._apply_encoder(feature, feature_pad_mask)

        # Apply decoder.
        # print("enc_seqs:", enc_seqs.shape)
        # print("enc_hstate:", enc_hstate.shape)
        self.decoder.init_dec_hstate(enc_hstate)
        preds = None
        for t_index in range(0, tokens.shape[-1]):
            # Teacher forcing.
            dec_inputs = tokens[:, t_index].reshape([-1, 1])
            pred = self.decoder(
                dec_inputs=dec_inputs,
                enc_seqs=enc_seqs,
                enc_mask=feature_pad_mask)
            if preds is None:
                preds = pred
            else:
                # `[N, T, C]`
                preds = torch.cat([preds, pred], dim=1)
        return preds

    def inference(self,
                  feature,
                  start_id,
                  end_id,
                  feature_pad_mask=None,
                  max_seqlen=62):
        """Forward computation for test.
        """
        self.attws = None

        enc_seqs, enc_hstate = self._apply_encoder(feature, feature_pad_mask)

        # Apply decoder.
        self.decoder.init_dec_hstate(enc_hstate)
        dec_inputs = torch.tensor([start_id]).to(feature.device)
        # `[N, T]`
        dec_inputs = dec_inputs.reshape([1, 1])
        preds = None
        pred_ids = [start_id]
        for _ in range(max_seqlen):
            pred = self.decoder(
                dec_inputs=dec_inputs,
                enc_seqs=enc_seqs,
                enc_mask=feature_pad_mask)
            if preds is None:
                preds = pred
            else:
                # `[N, T, C]`
                preds = torch.cat([preds, pred], dim=1)
            # Store attention.
            # attw: `[N, qlen, klen]`
            attw = self.decoder.attw.detach().cpu()
            if self.attws is None:
                self.attws = attw
            else:
                self.attws = torch.cat([self.attws, attw], dim=1)

            pid = torch.argmax(pred, dim=-1)
            dec_inputs = pid

            pid = pid.reshape([1]).detach().cpu().numpy()[0]
            pred_ids.append(int(pid))
            if int(pid) == end_id:
                break

        # `[N, T]`
        pred_ids = np.array([pred_ids])
        return pred_ids, preds


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
