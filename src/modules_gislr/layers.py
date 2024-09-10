#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""layers: Neural network layers.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import copy
import math
from inspect import signature

# Third party's modules
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

# Local modules
from .utils import (
    make_san_mask,
    make_causal_mask,
    select_reluwise_activation)

# Execution settings
VERSION = u"%(prog)s dev"


class Zero(nn.Module):
    """Place holder layer to return zero vector.
    """
    # This design is on purpose.
    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature, *args, **kwargs):
        """Perform forward computation.
        """
        return 0


class Identity(nn.Module):
    """Place holder layer to return identity vector.
    """
    # This design is on purpose.
    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature, *args, **kwargs):
        """Perform forward computation.
        """
        return feature


class TemporalAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 attention_type,
                 post_scale):
        super().__init__()
        assert attention_type in ["sigmoid", "softmax"]
        self.linear = nn.Linear(in_channels, 1)
        self.attention_type = attention_type

        if attention_type == "sigmoid":
            self.scale_layer = nn.Sigmoid()
        elif attention_type == "softmax":
            self.scale_layer = nn.Softmax(dim=1)

        self.neg_inf = None
        self.post_scale = post_scale

    def calc_attw(self, attw, mask):
        # Initialize masking value.
        if self.neg_inf is None:
            self.neg_inf = float(np.finfo(
                torch.tensor(0, dtype=attw.dtype).numpy().dtype).min)
        if mask is not None:
            attw = attw.masked_fill_(mask[:, :, None] == 0, self.neg_inf)
        attw = self.scale_layer(attw)
        if self.post_scale:
            if mask is None:
                tlength = torch.tensor(attw.shape[1], dtype=attw.dtype, device=attw.device)
                tlength = tlength.reshape([1, 1, 1])
            else:
                tlength = mask.sum(dim=1)
                tlength = tlength.reshape([-1, 1, 1])
            scale = tlength / attw.sum(dim=1, keepdims=True)
            attw = attw * scale
        return attw

    def forward(self, feature, mask=None):
        # `[N, T, C]`
        attw = self.linear(feature)
        attw = self.calc_attw(attw, mask)
        feature = attw * feature
        return feature, attw


class GPoolRecognitionHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.head = nn.Linear(in_channels, out_channels)
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.head.weight,
                        mean=0.,
                        std=math.sqrt(1. / self.out_channels))

    def forward(self, feature, feature_pad_mask=None):
        # Averaging over temporal axis.
        # `[N, C, T] -> [N, C, 1] -> [N, C]`
        if feature_pad_mask is not None:
            tlength = feature_pad_mask.sum(dim=-1)
            feature = feature * feature_pad_mask.unsqueeze(1)
            feature = feature.sum(dim=-1) / tlength.unsqueeze(-1)
        else:
            feature = F.avg_pool1d(feature, kernel_size=feature.shape[-1])
        feature = feature.reshape(feature.shape[0], -1)

        # Predict.
        feature = self.head(feature)
        return feature


class SimpleISLR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, 64)
        self.activation = nn.ReLU()
        self.head = GPoolRecognitionHead(64, out_channels)

    def forward(self, feature):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        feature = self.linear(feature)
        feature = self.activation(feature)

        # `[N, T, C'] -> [N, C', T]`
        feature = feature.permute(0, 2, 1)
        logit = self.head(feature)
        return logit


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
        dec_inputs = tokens[:, 0:1]
        preds = None
        for t_index in range(1, tokens.shape[-1]):
            pred = self.decoder(
                dec_inputs=dec_inputs,
                enc_seqs=enc_seqs,
                enc_mask=feature_pad_mask)
            if preds is None:
                preds = pred
            else:
                # `[N, T, C]`
                preds = torch.cat([preds, pred], dim=1)

            # Teacher forcing.
            dec_inputs = tokens[:, t_index:t_index+1]
        return preds

    def inference(self,
                  feature,
                  start_id,
                  end_id,
                  feature_pad_mask=None,
                  max_seqlen=10):
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


class PositionalEncoding(nn.Module):
    def __init__(self,
                 dim_model: int,
                 dropout: float,
                 max_len: int = 5000):
        super().__init__()
        self.dim_model = dim_model
        # Compute the positional encodings once in log space.
        pose = torch.zeros(max_len, dim_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float()
                             * -(math.log(10000.0) / dim_model))
        pose[:, 0::2] = torch.sin(position * div_term)
        pose[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pose", pose)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                feature):
        feature = feature + self.pose[None, :feature.shape[1], :]
        feature = self.dropout(feature)
        return feature


class MultiheadAttention(nn.Module):
    """Multi-headed attention (MHA) layer.

    # Args:
      - key_dim: The dimension of key.
      - query_dim: The dimension of query.
      - att_dim: The dimension of attention space.
      - out_dim: The dimension of output.
      - num_heads: The number of heads.
      - dropout: The dropout probability for attention weights.
      - add_bias: If True, use bias term in linear layers.
    """
    def __init__(self,
                 key_dim,
                 query_dim,
                 att_dim,
                 out_dim,
                 num_heads,
                 dropout,
                 add_bias):
        super().__init__()

        assert att_dim % num_heads == 0
        self.head_dim = att_dim // num_heads
        self.num_heads = num_heads
        self.scale = math.sqrt(self.head_dim)

        self.w_key = nn.Linear(key_dim, att_dim, bias=add_bias)
        self.w_value = nn.Linear(key_dim, att_dim, bias=add_bias)
        self.w_query = nn.Linear(query_dim, att_dim, bias=add_bias)

        self.w_out = nn.Linear(att_dim, out_dim, bias=add_bias)

        self.dropout_attn = nn.Dropout(p=dropout)

        self.neg_inf = None

        self.qkv_same_dim = key_dim == query_dim
        self.reset_parameters(add_bias)

    def reset_parameters(self, add_bias):
        """Initialize parameters with Xavier uniform distribution.

        # NOTE: For this initialization, please refer
        https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py  # pylint: disable=line-too-long

        """
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.w_key.weight)
            nn.init.xavier_uniform_(self.w_value.weight)
            nn.init.xavier_uniform_(self.w_query.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        if add_bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)
            nn.init.constant_(self.w_out.bias, 0.)

    def forward(self,
                key: torch.Tensor,
                value: torch.Tensor,
                query: torch.Tensor,
                mask: torch.Tensor):
        """Perform forward computation.

        # Args:
          - key: `[N, klen, key_dim]`
          - value: `[N, klen, vdim]`
          - query: `[N, qlen, query_dim]`
          - mask: `[N, qlen, klen]`
        # Returns:
          - cvec: The context vector. `[N, qlen, vdim]`
          - aws: The attention weights. `[N, H, qlen, klen]`
        """
        if self.neg_inf is None:
            self.neg_inf = float(np.finfo(
                torch.tensor(0, dtype=key.dtype).numpy().dtype).min)

        bsize, klen = key.size()[: 2]
        qlen = query.size(1)

        # key: `[N, klen, kdim] -> [N, klen, adim] -> [N, klen, H, adim/H(=hdim)]`
        # value: `[N, klen, vdim] -> [N, klen, adim] -> [N, klen, H, adim/H(=hdim)]`
        # query: `[N, qlen, qdim] -> [N, qlen, adim] -> [N, qlen, H, adim/H(=hdim)]`
        key = self.w_key(key).reshape([bsize, -1, self.num_heads, self.head_dim])
        value = self.w_value(value).reshape([bsize, -1, self.num_heads, self.head_dim])
        query = self.w_query(query).reshape([bsize, -1, self.num_heads, self.head_dim])

        # qk_score: `[N, qlen, H, hdim] x [N, klen, H, hdim] -> [N, qlen, klen, H]`
        qk_score = torch.einsum("bihd,bjhd->bijh", (query, key)) / self.scale

        # Apply mask.
        if mask is not None:
            # `[N, qlen, klen] -> [N, qlen, klen, H]`
            mask = mask.unsqueeze(3).repeat([1, 1, 1, self.num_heads])
            mask_size = (bsize, qlen, klen, self.num_heads)
            assert mask.size() == mask_size, f"{mask.size()}:{mask_size}"
            # Negative infinity should be 0 in softmax.
            qk_score = qk_score.masked_fill_(mask == 0, self.neg_inf)
        # Compute attention weight.
        attw = torch.softmax(qk_score, dim=2)
        attw = self.dropout_attn(attw)

        # cvec: `[N, qlen, klen, H] x [N, qlen, h, hdim] -> [N, qlen, H, hdim]
        # -> [N, qlen, H * hdim]`
        cvec = torch.einsum("bijh,bjhd->bihd", (attw, value))
        cvec = cvec.reshape([bsize, -1, self.num_heads * self.head_dim])
        cvec = self.w_out(cvec)
        # attw: `[N, qlen, klen, H]` -> `[N, H, qlen, klen]`
        attw = attw.permute(0, 3, 1, 2)
        return cvec, attw


class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 dim_model,
                 dim_ffw,
                 dropout,
                 activation,
                 add_bias):
       super().__init__()

       self.w_1 = nn.Linear(dim_model, dim_ffw, bias=add_bias)
       self.w_2 = nn.Linear(dim_ffw, dim_model, bias=add_bias)

       self.dropout = nn.Dropout(p=dropout)

       self.activation = select_reluwise_activation(activation)

    def forward(self, feature):
        feature = self.w_1(feature)
        feature = self.activation(feature)
        feature = self.dropout(feature)
        feature = self.w_2(feature)
        return feature


def create_norm(norm_type, dim_model, eps=1e-5, add_bias=None):
    # The argument `bias` was added at v2.1.0.
    # So, we check whether LayerNorm has this.
    sig = signature(nn.LayerNorm)
    available_bias = bool("bias" in sig.parameters)
    if norm_type == "layer":
        if available_bias:
            norm = nn.LayerNorm(dim_model, eps=eps, bias=add_bias)
        else:
            norm = nn.LayerNorm(dim_model, eps=eps)
    elif norm_type == "batch":
        norm = nn.BatchNorm1d(dim_model, eps=eps)
    return norm


def apply_norm(norm_layer, feature, channel_first=False):
    if isinstance(norm_layer, nn.LayerNorm):
        if channel_first:
            # `[N, C, T] -> [N, T, C] -> [N, C, T]`
            feature = feature.permute([0, 2, 1]).contiguous()
            feature = norm_layer(feature)
            feature = feature.permute([0, 2, 1]).contiguous()
        else:
            feature = norm_layer(feature)
    elif isinstance(norm_layer, nn.BatchNorm1d):
        if channel_first:
            feature = norm_layer(feature)
        else:
            # `[N, T, C] -> [N, C, T]`
            feature = feature.permute([0, 2, 1]).contiguous()
            feature = norm_layer(feature)
            # `[N, C, T] -> [N, T, C]`
            feature = feature.permute([0, 2, 1]).contiguous()
    return feature


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_ffw,
                 dropout,
                 activation,
                 norm_type_sattn,
                 norm_type_ffw,
                 norm_eps,
                 norm_first,
                 add_bias):
        super().__init__()

        self.norm_first = norm_first

        #################################################
        # MHA.
        #################################################
        self.self_attn = MultiheadAttention(
            key_dim=dim_model,
            query_dim=dim_model,
            att_dim=dim_model,
            out_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            add_bias=add_bias)
        self.norm_sattn = create_norm(norm_type_sattn, dim_model, norm_eps, add_bias)

        #################################################
        # PFFN.
        #################################################
        self.ffw = PositionwiseFeedForward(
            dim_model=dim_model,
            dim_ffw=dim_ffw,
            dropout=dropout,
            activation=activation,
            add_bias=add_bias)
        self.norm_ffw = create_norm(norm_type_ffw, dim_model, norm_eps, add_bias)

        self.dropout = nn.Dropout(p=dropout)

        # To store attention weights.
        self.attw = None

    def _forward_prenorm(self,
                         feature,
                         san_mask):
        """Pre-normalization structure.

        For the details, please refer
        https://arxiv.org/pdf/2002.04745v1.pdf
        """
        #################################################
        # self-attention
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_sattn, feature)
        feature, self.attw = self.self_attn(
            key=feature,
            value=feature,
            query=feature,
            mask=san_mask)
        feature = self.dropout(feature) + residual

        #################################################
        # FFW
        #################################################
        residual = feature
        # `[N, qlen, dim_model]`
        feature = apply_norm(self.norm_ffw, feature)
        feature = self.ffw(feature)
        feature = self.dropout(feature) + residual
        return feature

    def _forward_postnorm(self,
                          feature,
                          san_mask):
        """Post-normalization structure (standard).

        """
        #################################################
        # self-attention
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature, self.attw = self.self_attn(
            key=feature,
            value=feature,
            query=feature,
            mask=san_mask)
        feature = self.dropout(feature) + residual
        feature = apply_norm(self.norm_sattn, feature)

        #################################################
        # FFW
        #################################################
        residual = feature
        # `[N, qlen, dim_model]`
        feature = self.ffw(feature)
        feature = self.dropout(feature) + residual
        feature = apply_norm(self.norm_ffw, feature)
        return feature

    def forward(self,
                feature,
                causal_mask=None,
                src_key_padding_mask=None):
        bsize, qlen = feature.shape[:2]
        if src_key_padding_mask is not None:
            san_mask = make_san_mask(src_key_padding_mask, causal_mask)
        elif causal_mask is not None:
            san_mask = causal_mask
        else:
            san_mask = None

        if self.norm_first:
            feature = self._forward_prenorm(feature, san_mask)
        else:
            feature = self._forward_postnorm(feature, san_mask)

        return feature


class TransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers,
                 dim_model,
                 dropout_pe,
                 norm_type_tail,
                 norm_eps,
                 norm_first,
                 add_bias,
                 add_tailnorm):
        super().__init__()

        self.pos_encoder = PositionalEncoding(dim_model, dropout_pe)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        # Add LayerNorm at tail position.
        # This is applied only when norm_first is True because
        # post-normalization structure includes tail-normalization in encoder
        # layers.
        if add_tailnorm and norm_first:
            self.norm_tail = create_norm(norm_type_tail, dim_model, norm_eps, add_bias)
        else:
            self.norm_tail = Identity()

    def forward(self,
                feature,
                causal_mask,
                src_key_padding_mask):
        feature = self.pos_encoder(feature)
        for layer in self.layers:
            feature = layer(feature,
                            causal_mask,
                            src_key_padding_mask)
        feature = apply_norm(self.norm_tail, feature)
        return feature


class TransformerEnISLR(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 activation="relu",
                 pooling_type="none",
                 tren_num_layers=1,
                 tren_num_heads=1,
                 tren_dim_ffw=256,
                 tren_dropout_pe=0.1,
                 tren_dropout=0.1,
                 tren_norm_type_sattn="layer",
                 tren_norm_type_ffw="layer",
                 tren_norm_type_tail="layer",
                 tren_norm_eps=1e-5,
                 tren_norm_first=True,
                 tren_add_bias=True,
                 tren_add_tailnorm=True):
        super().__init__()

        # Feature extraction.
        self.linear = nn.Linear(in_channels, inter_channels)
        self.activation = select_reluwise_activation(activation)

        if pooling_type == "none":
            self.pooling = Identity()
        elif pooling_type == "average":
            self.pooling = nn.AvgPool2d(
                kernel_size=[2, 1],
                stride=[2, 1],
                padding=0)
        elif pooling_type == "max":
            self.pooling = nn.MaxPool2d(
                kernel_size=[2, 1],
                stride=[2, 1],
                padding=0)

        # Transformer-Encoder.
        enlayer = TransformerEncoderLayer(
            dim_model=inter_channels,
            num_heads=tren_num_heads,
            dim_ffw=tren_dim_ffw,
            dropout=tren_dropout,
            activation=activation,
            norm_type_sattn=tren_norm_type_sattn,
            norm_type_ffw=tren_norm_type_ffw,
            norm_eps=tren_norm_eps,
            norm_first=tren_norm_first,
            add_bias=tren_add_bias)
        self.tr_encoder = TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=tren_num_layers,
            dim_model=inter_channels,
            dropout_pe=tren_dropout_pe,
            norm_type_tail=tren_norm_type_tail,
            norm_eps=tren_norm_eps,
            norm_first=tren_norm_first,
            add_bias=tren_add_bias,
            add_tailnorm=tren_add_tailnorm)

        self.head = GPoolRecognitionHead(inter_channels, out_channels)

    def forward(self,
                feature,
                feature_causal_mask=None,
                feature_pad_mask=None):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        feature = self.linear(feature)
        if torch.isnan(feature).any():
            raise ValueError()
        feature = self.activation(feature)
        if torch.isnan(feature).any():
            raise ValueError()

        # Apply pooling.
        feature = self.pooling(feature)
        if feature_pad_mask is not None:
            # Cast to apply pooling.
            feature_pad_mask = feature_pad_mask.to(feature.dtype)
            feature_pad_mask = self.pooling(feature_pad_mask.unsqueeze(-1)).squeeze(-1)
            # Binarization.
            # This removes averaged signals with padding.
            feature_pad_mask = feature_pad_mask > 0.5
            if feature_causal_mask is not None:
                feature_causal_mask = make_causal_mask(feature_pad_mask)

        feature = self.tr_encoder(
            feature=feature,
            causal_mask=feature_causal_mask,
            src_key_padding_mask=feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()

        # `[N, T, C] -> [N, C, T]`
        logit = self.head(feature.permute([0, 2, 1]), feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()
        return logit


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_ffw,
                 dropout,
                 activation,
                 norm_type_sattn,
                 norm_type_cattn,
                 norm_type_ffw,
                 norm_eps,
                 norm_first,
                 add_bias):
        super().__init__()

        self.norm_first = norm_first

        #################################################
        # MHSA.
        #################################################
        self.self_attn = MultiheadAttention(
            key_dim=dim_model,
            query_dim=dim_model,
            att_dim=dim_model,
            out_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            add_bias=add_bias)
        self.norm_sattn = create_norm(norm_type_sattn, dim_model, norm_eps, add_bias)

        #################################################
        # MHCA.
        #################################################
        self.cross_attn = MultiheadAttention(
            key_dim=dim_model,
            query_dim=dim_model,
            att_dim=dim_model,
            out_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            add_bias=add_bias)
        self.norm_cattn = create_norm(norm_type_cattn, dim_model, norm_eps, add_bias)

        #################################################
        # PFFN.
        #################################################
        self.ffw = PositionwiseFeedForward(
            dim_model=dim_model,
            dim_ffw=dim_ffw,
            dropout=dropout,
            activation=activation,
            add_bias=add_bias)
        self.norm_ffw = create_norm(norm_type_ffw, dim_model, norm_eps, add_bias)

        self.dropout = nn.Dropout(p=dropout)

        # To store attention weights.
        self.sattw = None
        self.cattw = None

    def _forward_prenorm(self,
                         tgt_feature,
                         enc_feature,
                         tgt_san_mask,
                         enc_tgt_mask):
        """Pre-normalization structure.

        For the details, please refer
        https://arxiv.org/pdf/2002.04745v1.pdf
        """
        #################################################
        # self-attention
        #################################################
        residual = tgt_feature
        tgt_feature = apply_norm(self.norm_sattn, tgt_feature)
        tgt_feature, self.sattw = self.self_attn(
            key=tgt_feature,
            value=tgt_feature,
            query=tgt_feature,
            mask=tgt_san_mask)
        tgt_feature = self.dropout(tgt_feature) + residual

        #################################################
        # cross-attention
        #################################################
        residual = tgt_feature
        tgt_feature = apply_norm(self.norm_cattn, tgt_feature)
        tgt_feature, self.cattw = self.cross_attn(
            key=enc_feature,
            value=enc_feature,
            query=tgt_feature,
            mask=enc_tgt_mask)
        tgt_feature = self.dropout(tgt_feature) + residual

        #################################################
        # FFW
        #################################################
        residual = tgt_feature
        tgt_feature = apply_norm(self.norm_ffw, tgt_feature)
        tgt_feature = self.ffw(tgt_feature)
        tgt_feature = self.dropout(tgt_feature) + residual
        return tgt_feature

    def _forward_postnorm(self,
                          tgt_feature,
                          enc_feature,
                          tgt_san_mask,
                          enc_tgt_mask):
        """Post-normalization structure (standard).

        """
        #################################################
        # self-attention
        #################################################
        residual = tgt_feature
        tgt_feature, self.sattw = self.self_attn(
            key=tgt_feature,
            value=tgt_feature,
            query=tgt_feature,
            mask=tgt_san_mask)
        tgt_feature = self.dropout(tgt_feature) + residual
        tgt_feature = apply_norm(self.norm_sattn, tgt_feature)

        #################################################
        # cross-attention
        #################################################
        residual = tgt_feature
        tgt_feature, self.cattw = self.cross_attn(
            key=enc_feature,
            value=enc_feature,
            query=tgt_feature,
            mask=enc_tgt_mask)
        tgt_feature = self.dropout(tgt_feature) + residual
        tgt_feature = apply_norm(self.norm_cattn, tgt_feature)

        #################################################
        # FFW
        #################################################
        residual = tgt_feature
        tgt_feature = self.ffw(tgt_feature)
        tgt_feature = self.dropout(tgt_feature) + residual
        tgt_feature = apply_norm(self.norm_ffw, tgt_feature)

        return tgt_feature

    def forward(self,
                tgt_feature,
                enc_feature,
                tgt_causal_mask=None,
                enc_tgt_causal_mask=None,
                tgt_key_padding_mask=None,
                enc_key_padding_mask=None):

        # Create mask.
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = torch.ones(tgt_feature.shape[:2],
                                              dtype=enc_feature.dtype,
                                              device=enc_feature.device)
        tgt_san_mask = make_san_mask(tgt_key_padding_mask, tgt_causal_mask)
        if enc_key_padding_mask is None:
            enc_key_padding_mask = torch.ones(enc_feature.shape[:2],
                                              dtype=enc_feature.dtype,
                                              device=enc_feature.device)
        enc_tgt_mask = enc_key_padding_mask.unsqueeze(1).repeat(
            [1, tgt_feature.shape[1], 1])
        if enc_tgt_causal_mask is not None:
            enc_tgt_mask = enc_tgt_mask & enc_tgt_causal_mask

        if self.norm_first:
            tgt_feature = self._forward_prenorm(tgt_feature, enc_feature,
                                                tgt_san_mask, enc_tgt_mask)
        else:
            tgt_feature = self._forward_postnorm(tgt_feature, enc_feature,
                                                 tgt_san_mask, enc_tgt_mask)

        return tgt_feature


class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer,
                 out_channels,
                 num_layers,
                 dim_model,
                 dropout_pe,
                 norm_type_tail,
                 norm_eps,
                 norm_first,
                 add_bias,
                 add_tailnorm,
                 padding_val):
        super().__init__()

        self.emb_layer = nn.Embedding(out_channels,
                                      dim_model,
                                      padding_idx=padding_val)
        self.vocab_size = out_channels

        self.pos_encoder = PositionalEncoding(dim_model, dropout_pe)
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

        # Add LayerNorm at tail position.
        # This is applied only when norm_first is True because
        # post-normalization structure includes tail-normalization in encoder
        # layers.
        if add_tailnorm and norm_first:
            self.norm_tail = create_norm(norm_type_tail, dim_model, norm_eps, add_bias)
        else:
            self.norm_tail = Identity()

        self.head = nn.Linear(dim_model, out_channels)

    def forward(self,
                tgt_feature,
                enc_feature,
                tgt_causal_mask,
                enc_tgt_causal_mask,
                tgt_key_padding_mask,
                enc_key_padding_mask):

        tgt_feature = self.emb_layer(tgt_feature) * math.sqrt(self.vocab_size)

        tgt_feature = self.pos_encoder(tgt_feature)
        for layer in self.layers:
            tgt_feature = layer(
                tgt_feature=tgt_feature,
                enc_feature=enc_feature,
                tgt_causal_mask=tgt_causal_mask,
                enc_tgt_causal_mask=enc_tgt_causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                enc_key_padding_mask=enc_key_padding_mask)
        tgt_feature = apply_norm(self.norm_tail, tgt_feature)

        logit = self.head(tgt_feature)
        return logit


class TransformerCSLR(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 padding_val,
                 activation="relu",
                 tren_num_layers=1,
                 tren_num_heads=1,
                 tren_dim_ffw=256,
                 tren_dropout_pe=0.1,
                 tren_dropout=0.1,
                 tren_norm_type_sattn="layer",
                 tren_norm_type_ffw="layer",
                 tren_norm_type_tail="layer",
                 tren_norm_eps=1e-5,
                 tren_norm_first=True,
                 tren_add_bias=True,
                 tren_add_tailnorm=True,
                 trde_num_layers=1,
                 trde_num_heads=1,
                 trde_dim_ffw=256,
                 trde_dropout_pe=0.1,
                 trde_dropout=0.1,
                 trde_norm_type_sattn="layer",
                 trde_norm_type_cattn="layer",
                 trde_norm_type_ffw="layer",
                 trde_norm_type_tail="layer",
                 trde_norm_eps=1e-5,
                 trde_norm_first=True,
                 trde_add_bias=True,
                 trde_add_tailnorm=True):
        super().__init__()

        # Feature extraction.
        self.linear = nn.Linear(in_channels, inter_channels)
        self.activation = select_reluwise_activation(activation)

        # Transformer-Encoder.
        enlayer = TransformerEncoderLayer(
            dim_model=inter_channels,
            num_heads=tren_num_heads,
            dim_ffw=tren_dim_ffw,
            dropout=tren_dropout,
            activation=activation,
            norm_type_sattn=tren_norm_type_sattn,
            norm_type_ffw=tren_norm_type_ffw,
            norm_eps=tren_norm_eps,
            norm_first=tren_norm_first,
            add_bias=tren_add_bias)
        self.tr_encoder = TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=tren_num_layers,
            dim_model=inter_channels,
            dropout_pe=tren_dropout_pe,
            norm_type_tail=tren_norm_type_tail,
            norm_eps=tren_norm_eps,
            norm_first=tren_norm_first,
            add_bias=tren_add_bias,
            add_tailnorm=tren_add_tailnorm)

        # Transformer-Decoder.
        delayer = TransformerDecoderLayer(
            dim_model=inter_channels,
            num_heads=trde_num_heads,
            dim_ffw=trde_dim_ffw,
            dropout=trde_dropout,
            activation=activation,
            norm_type_sattn=trde_norm_type_sattn,
            norm_type_cattn=trde_norm_type_cattn,
            norm_type_ffw=trde_norm_type_ffw,
            norm_eps=trde_norm_eps,
            norm_first=trde_norm_first,
            add_bias=trde_add_bias)
        self.tr_decoder = TransformerDecoder(
            decoder_layer=delayer,
            out_channels=out_channels,
            num_layers=trde_num_layers,
            dim_model=inter_channels,
            dropout_pe=trde_dropout_pe,
            norm_type_tail=trde_norm_type_tail,
            norm_eps=trde_norm_eps,
            norm_first=trde_norm_first,
            add_bias=trde_add_bias,
            add_tailnorm=trde_add_tailnorm,
            padding_val=padding_val)

    def forward(self,
                src_feature,
                tgt_feature,
                src_causal_mask,
                src_padding_mask,
                tgt_causal_mask,
                tgt_padding_mask):
        """Forward computation for train.
        """
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = src_feature.shape
        src_feature = src_feature.permute([0, 2, 1, 3])
        src_feature = src_feature.reshape(N, T, -1)

        src_feature = self.linear(src_feature)

        enc_feature = self.tr_encoder(
            feature=src_feature,
            causal_mask=src_causal_mask,
            src_key_padding_mask=src_padding_mask)

        preds = self.tr_decoder(tgt_feature=tgt_feature,
                                enc_feature=enc_feature,
                                tgt_causal_mask=tgt_causal_mask,
                                enc_tgt_causal_mask=None,
                                tgt_key_padding_mask=tgt_padding_mask,
                                enc_key_padding_mask=src_padding_mask)
        # `[N, T, C]`
        return preds

    def inference(self,
                  src_feature,
                  start_id,
                  end_id,
                  src_padding_mask=None,
                  max_seqlen=10):
        """Forward computation for test.
        """

        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = src_feature.shape
        src_feature = src_feature.permute([0, 2, 1, 3])
        src_feature = src_feature.reshape(N, T, -1)

        src_feature = self.linear(src_feature)

        enc_feature = self.tr_encoder(
            feature=src_feature,
            causal_mask=None,
            src_key_padding_mask=src_padding_mask)

        # Apply decoder.
        dec_inputs = torch.tensor([start_id]).to(src_feature.device)
        # `[N, T]`
        dec_inputs = dec_inputs.reshape([1, 1])
        preds = None
        pred_ids = [start_id]
        for _ in range(max_seqlen):
            pred = self.tr_decoder(
                tgt_feature=dec_inputs,
                enc_feature=enc_feature,
                tgt_causal_mask=None,
                enc_tgt_causal_mask=None,
                tgt_key_padding_mask=None,
                enc_key_padding_mask=src_padding_mask)
            # Extract last prediction.
            pred = pred[:, -1:, :]
            # `[N, T, C]`
            if preds is None:
                preds = pred
            else:
                # Concatenate last elements.
                preds = torch.cat([preds, pred], dim=1)

            pid = torch.argmax(pred, dim=-1)
            dec_inputs = torch.cat([dec_inputs, pid], dim=-1)

            pid = pid.reshape([1]).detach().cpu().numpy()[0]
            pred_ids.append(int(pid))
            if int(pid) == end_id:
                break

        # `[N, T]`
        pred_ids = np.array([pred_ids])
        return pred_ids, preds


class MacaronNetEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_ffw,
                 dropout,
                 activation,
                 norm_type_sattn,
                 norm_type_ffw,
                 norm_eps,
                 add_bias,
                 fc_factor=0.5,
                 shared_ffw=False):
        super().__init__()

        self.fc_factor = fc_factor

        # =====================================================================
        # First half PFFN.
        # =====================================================================
        self.norm_ffw1 = create_norm(norm_type_ffw, dim_model, norm_eps, add_bias)
        self.ffw1 = PositionwiseFeedForward(
            dim_model=dim_model,
            dim_ffw=dim_ffw,
            dropout=dropout,
            activation=activation,
            add_bias=add_bias)
        # =====================================================================
        # MHA.
        # =====================================================================
        self.norm_sattn = create_norm(norm_type_sattn, dim_model, norm_eps, add_bias)
        self.self_attn = MultiheadAttention(
            key_dim=dim_model,
            query_dim=dim_model,
            att_dim=dim_model,
            out_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            add_bias=add_bias)
        # =====================================================================
        # Second half PFFN.
        # =====================================================================
        self.norm_ffw2 = create_norm(norm_type_ffw, dim_model, norm_eps, add_bias)
        if shared_ffw:
            self.ffw2 = self.ffw1
        else:
            self.ffw2 = PositionwiseFeedForward(
                dim_model=dim_model,
                dim_ffw=dim_ffw,
                dropout=dropout,
                activation=activation,
                add_bias=add_bias)

        self.dropout = nn.Dropout(p=dropout)

        # To store attention weights.
        self.attw = None

    def forward(self,
                feature,
                causal_mask=None,
                src_key_padding_mask=None):
        bsize, qlen = feature.shape[:2]
        if src_key_padding_mask is not None:
            san_mask = make_san_mask(src_key_padding_mask, causal_mask)
        elif causal_mask is not None:
            san_mask = causal_mask
        else:
            san_mask = None

        #################################################
        # First half PFFN.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_ffw1, feature)
        feature = self.ffw1(feature)
        feature = self.fc_factor * self.dropout(feature) + residual

        #################################################
        # MHA.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_sattn, feature)
        feature, self.attw = self.self_attn(
            key=feature,
            value=feature,
            query=feature,
            mask=san_mask)
        feature = self.dropout(feature) + residual

        #################################################
        # Second half PFFN.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_ffw2, feature)
        feature = self.ffw2(feature)
        feature = self.fc_factor * self.dropout(feature) + residual

        return feature


class MacaronNetEnISLR(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 activation="relu",
                 pooling_type="none",
                 tren_num_layers=1,
                 tren_num_heads=1,
                 tren_dim_ffw=256,
                 tren_dropout_pe=0.1,
                 tren_dropout=0.1,
                 tren_norm_type_sattn="layer",
                 tren_norm_type_ffw="layer",
                 tren_norm_type_tail="layer",
                 tren_norm_eps=1e-5,
                 tren_add_bias=True,
                 tren_add_tailnorm=True,
                 fc_factor=0.5,
                 shared_ffw=False):
        super().__init__()

        # Feature extraction.
        self.linear = nn.Linear(in_channels, inter_channels)
        self.activation = select_reluwise_activation(activation)

        if pooling_type == "none":
            self.pooling = Identity()
        elif pooling_type == "average":
            self.pooling = nn.AvgPool2d(
                kernel_size=[2, 1],
                stride=[2, 1],
                padding=0)
        elif pooling_type == "max":
            self.pooling = nn.MaxPool2d(
                kernel_size=[2, 1],
                stride=[2, 1],
                padding=0)

        # Transformer-Encoder.
        enlayer = MacaronNetEncoderLayer(
            dim_model=inter_channels,
            num_heads=tren_num_heads,
            dim_ffw=tren_dim_ffw,
            dropout=tren_dropout,
            activation=activation,
            norm_type_sattn=tren_norm_type_sattn,
            norm_type_ffw=tren_norm_type_ffw,
            norm_eps=tren_norm_eps,
            add_bias=tren_add_bias,
            fc_factor=fc_factor,
            shared_ffw=shared_ffw)
        self.tr_encoder = TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=tren_num_layers,
            dim_model=inter_channels,
            dropout_pe=tren_dropout_pe,
            norm_type_tail=tren_norm_type_tail,
            norm_eps=tren_norm_eps,
            norm_first=True,  # Fixed for MacaronNet.
            add_bias=tren_add_bias,
            add_tailnorm=tren_add_tailnorm)

        self.head = GPoolRecognitionHead(inter_channels, out_channels)

    def forward(self,
                feature,
                feature_causal_mask=None,
                feature_pad_mask=None):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        feature = self.linear(feature)
        if torch.isnan(feature).any():
            raise ValueError()
        feature = self.activation(feature)
        if torch.isnan(feature).any():
            raise ValueError()

        # Apply pooling.
        feature = self.pooling(feature)
        if feature_pad_mask is not None:
            # Cast to apply pooling.
            feature_pad_mask = feature_pad_mask.to(feature.dtype)
            feature_pad_mask = self.pooling(feature_pad_mask.unsqueeze(-1)).squeeze(-1)
            # Binarization.
            # This removes averaged signals with padding.
            feature_pad_mask = feature_pad_mask > 0.5
            if feature_causal_mask is not None:
                feature_causal_mask = make_causal_mask(feature_pad_mask)

        feature = self.tr_encoder(
            feature=feature,
            causal_mask=feature_causal_mask,
            src_key_padding_mask=feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()

        # `[N, T, C] -> [N, C, T]`
        logit = self.head(feature.permute([0, 2, 1]), feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()
        return logit


class ConformerConvBlock(nn.Module):
    def __init__(self,
                 dim_model,
                 kernel_size,
                 conv_type="separable",
                 norm_type="layer",
                 activation="swish",
                 padding_mode="zeros",
                 add_tail_conv=True,
                 causal=False):
        super().__init__()

        assert conv_type in ["separable", "standard", "predepth"]
        assert norm_type in ["layer", "batch"]
        assert (kernel_size - 1) % 2 == 0, f"kernel_size:{kernel_size} must be the odd number."
        assert kernel_size >= 3, f"kernel_size: {kernel_size} must be larger than 3."
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.causal = causal

        if causal:
            self.padding = (kernel_size - 1)
        else:
            self.padding = (kernel_size - 1) // 2

        if conv_type == "separable":
            self.conv_module = nn.Sequential(
                collections.OrderedDict([
                    # Point-wise.
                    ("pconv",
                     nn.Conv1d(
                        in_channels=dim_model,
                        out_channels=dim_model * 2,  # for GLU
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True)),
                    ("glu", nn.GLU(dim=1)),
                    # Depth-wise
                    ("dconv",
                     nn.Conv1d(
                        in_channels=dim_model,
                        out_channels=dim_model,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=self.padding,
                        groups=dim_model,
                        padding_mode=padding_mode,
                        bias=True))])
                )
        elif conv_type == "predepth":
            self.conv_module = nn.Sequential(
                collections.OrderedDict([
                    # Depth-wise
                    ("dconv",
                     nn.Conv1d(
                        in_channels=dim_model,
                        out_channels=dim_model,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=self.padding,
                        groups=dim_model,
                        padding_mode=padding_mode,
                        bias=True)),
                    # Point-wise.
                    ("pconv",
                     nn.Conv1d(
                        in_channels=dim_model,
                        out_channels=dim_model * 2,  # for GLU
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True)),
                    ("glu", nn.GLU(dim=1))])
                )
        elif conv_type == "standard":
            self.conv_module = nn.Sequential(
                collections.OrderedDict([
                    ("conv",
                     nn.Conv1d(
                         in_channels=dim_model,
                         out_channels=dim_model * 2,  # for GLU
                         kernel_size=kernel_size,
                         stride=1,
                         padding=self.padding,
                         bias=True)),
                    ("glu", nn.GLU(dim=1))])
                )

        self.norm = create_norm(norm_type, dim_model)

        self.activation = select_reluwise_activation(activation)

        if add_tail_conv:
            self.tail_pointwise_conv = nn.Conv1d(
                in_channels=dim_model,
                out_channels=dim_model,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.tail_pointwise_conv = nn.Identity()

        self.add_tail_conv = add_tail_conv

    def forward(self,
                feature):
        # `[N, T, C] -> [N, C, T]`
        feature = feature.permute([0, 2, 1]).contiguous()
        feature = self.conv_module(feature)

        if self.causal:
            feature = feature[:, :, :-self.padding]

        # `[N, C, T]`: channel_first
        feature = apply_norm(self.norm, feature, channel_first=True)

        feature = self.activation(feature)
        feature = self.tail_pointwise_conv(feature)

        # `[N, C, T] -> [N, T, C]`
        feature = feature.permute([0, 2, 1]).contiguous()
        return feature


class ConformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model,
                 num_heads,
                 dim_ffw,
                 dropout,
                 activation,
                 norm_type_sattn,
                 norm_type_ffw,
                 norm_eps,
                 add_bias,
                 conv_kernel_size=3,
                 conv_type="separable",
                 conv_norm_type="layer",
                 conv_activation="swish",
                 conv_padding_mode="zeros",
                 conv_add_tail_conv=True,
                 conv_causal=False,
                 conv_layout="post"):
        super().__init__()
        assert conv_layout in ["pre", "post"]
        self.conv_layout = conv_layout

        self.fc_factor = 0.5

        # =====================================================================
        # First half PFFN.
        # =====================================================================
        self.norm_ffw1 = create_norm(norm_type_ffw, dim_model, norm_eps, add_bias)
        self.ffw1 = PositionwiseFeedForward(
            dim_model=dim_model,
            dim_ffw=dim_ffw,
            dropout=dropout,
            activation=activation,
            add_bias=add_bias)
        # =====================================================================
        # MHA.
        # =====================================================================
        self.norm_sattn = create_norm(norm_type_sattn, dim_model, norm_eps, add_bias)
        self.self_attn = MultiheadAttention(
            key_dim=dim_model,
            query_dim=dim_model,
            att_dim=dim_model,
            out_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            add_bias=add_bias)
        # =====================================================================
        # Conv module.
        # =====================================================================
        self.norm_conv = create_norm(conv_norm_type, dim_model, norm_eps, add_bias)
        self.conv = ConformerConvBlock(
            dim_model=dim_model,
            kernel_size=conv_kernel_size,
            conv_type=conv_type,
            norm_type=conv_norm_type,
            activation=conv_activation,
            padding_mode=conv_padding_mode,
            add_tail_conv=conv_add_tail_conv,
            causal=conv_causal)

        # =====================================================================
        # Second half PFFN.
        # =====================================================================
        self.norm_ffw2 = create_norm(norm_type_ffw, dim_model, norm_eps, add_bias)
        self.ffw2 = PositionwiseFeedForward(
            dim_model=dim_model,
            dim_ffw=dim_ffw,
            dropout=dropout,
            activation=activation,
            add_bias=add_bias)

        self.dropout = nn.Dropout(p=dropout)

        # To store attention weights.
        self.attw = None

    def _forward_preconv(self, feature, san_mask):
        #################################################
        # First half PFFN.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_ffw1, feature)
        feature = self.ffw1(feature)
        feature = self.fc_factor * self.dropout(feature) + residual

        #################################################
        # Conv.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_conv, feature)
        feature = self.conv(feature)
        feature = self.dropout(feature) + residual

        #################################################
        # MHA.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_sattn, feature)
        feature, self.attw = self.self_attn(
            key=feature,
            value=feature,
            query=feature,
            mask=san_mask)
        feature = self.dropout(feature) + residual

        #################################################
        # Second half PFFN.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_ffw2, feature)
        feature = self.ffw2(feature)
        feature = self.fc_factor * self.dropout(feature) + residual
        return feature

    def _forward_postconv(self, feature, san_mask):
        #################################################
        # First half PFFN.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_ffw1, feature)
        feature = self.ffw1(feature)
        feature = self.fc_factor * self.dropout(feature) + residual

        #################################################
        # MHA.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_sattn, feature)
        feature, self.attw = self.self_attn(
            key=feature,
            value=feature,
            query=feature,
            mask=san_mask)
        feature = self.dropout(feature) + residual

        #################################################
        # Conv.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_conv, feature)
        feature = self.conv(feature)
        feature = self.dropout(feature) + residual

        #################################################
        # Second half PFFN.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_ffw2, feature)
        feature = self.ffw2(feature)
        feature = self.fc_factor * self.dropout(feature) + residual
        return feature

    def forward(self,
                feature,
                causal_mask=None,
                src_key_padding_mask=None):
        bsize, qlen = feature.shape[:2]
        if src_key_padding_mask is not None:
            san_mask = make_san_mask(src_key_padding_mask, causal_mask)
        elif causal_mask is not None:
            san_mask = causal_mask
        else:
            san_mask = None

        if self.conv_layout == "pre":
            feature = self._forward_preconv(feature, san_mask)
        else:
            feature = self._forward_postconv(feature, san_mask)
        return feature


class ConformerEnISLR(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 activation="relu",
                 pooling_type="none",
                 tren_num_layers=1,
                 tren_num_heads=1,
                 tren_dim_ffw=256,
                 tren_dropout_pe=0.1,
                 tren_dropout=0.1,
                 tren_norm_type_sattn="layer",
                 tren_norm_type_ffw="layer",
                 tren_norm_type_tail="layer",
                 tren_norm_eps=1e-5,
                 tren_add_bias=True,
                 tren_add_tailnorm=True,
                 conv_kernel_size=3,
                 conv_type="separable",
                 conv_norm_type="layer",
                 conv_activation="swish",
                 conv_padding_mode="zeros",
                 conv_add_tail_conv=True,
                 conv_causal=False,
                 conv_layout="post"):
        super().__init__()

        # Feature extraction.
        self.linear = nn.Linear(in_channels, inter_channels)
        self.activation = select_reluwise_activation(activation)

        if pooling_type == "none":
            self.pooling = Identity()
        elif pooling_type == "average":
            self.pooling = nn.AvgPool2d(
                kernel_size=[2, 1],
                stride=[2, 1],
                padding=0)
        elif pooling_type == "max":
            self.pooling = nn.MaxPool2d(
                kernel_size=[2, 1],
                stride=[2, 1],
                padding=0)

        # Transformer-Encoder.
        enlayer = ConformerEncoderLayer(
            dim_model=inter_channels,
            num_heads=tren_num_heads,
            dim_ffw=tren_dim_ffw,
            dropout=tren_dropout,
            activation=activation,
            norm_type_sattn=tren_norm_type_sattn,
            norm_type_ffw=tren_norm_type_ffw,
            norm_eps=tren_norm_eps,
            add_bias=tren_add_bias,
            conv_kernel_size=conv_kernel_size,
            conv_type=conv_type,
            conv_activation=conv_activation,
            conv_norm_type=conv_norm_type,
            conv_padding_mode=conv_padding_mode,
            conv_causal=conv_causal,
            conv_add_tail_conv=conv_add_tail_conv,
            conv_layout=conv_layout)
        self.tr_encoder = TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=tren_num_layers,
            dim_model=inter_channels,
            dropout_pe=tren_dropout_pe,
            norm_type_tail=tren_norm_type_tail,
            norm_eps=tren_norm_eps,
            norm_first=True,  # Fixed for MacaronNet.
            add_bias=tren_add_bias,
            add_tailnorm=tren_add_tailnorm)

        self.head = GPoolRecognitionHead(inter_channels, out_channels)

    def forward(self,
                feature,
                feature_causal_mask=None,
                feature_pad_mask=None):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        feature = self.linear(feature)
        if torch.isnan(feature).any():
            raise ValueError()
        feature = self.activation(feature)
        if torch.isnan(feature).any():
            raise ValueError()

        # Apply pooling.
        feature = self.pooling(feature)
        if feature_pad_mask is not None:
            # Cast to apply pooling.
            feature_pad_mask = feature_pad_mask.to(feature.dtype)
            feature_pad_mask = self.pooling(feature_pad_mask.unsqueeze(-1)).squeeze(-1)
            # Binarization.
            # This removes averaged signals with padding.
            feature_pad_mask = feature_pad_mask > 0.5
            if feature_causal_mask is not None:
                feature_causal_mask = make_causal_mask(feature_pad_mask)

        feature = self.tr_encoder(
            feature=feature,
            causal_mask=feature_causal_mask,
            src_key_padding_mask=feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()

        # `[N, T, C] -> [N, C, T]`
        logit = self.head(feature.permute([0, 2, 1]), feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()
        return logit


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
