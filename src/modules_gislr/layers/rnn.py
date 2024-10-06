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

from pydantic import (
    Field)

from torch import nn

# Local modules
from .misc import (
    ConfiguredModel,
    GPoolRecognitionHeadSettings,
    TemporalAttentionSettings)

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class RNNSettings(ConfiguredModel):
    in_channels: int = 64
    out_channels: int = 64
    rnn_type: str = Field(default="gru", pattern=r"srnn|lstm|gru")
    num_layers: int = 1
    activation: str = Field(default="tanh", pattern=r"tanh|relu")
    bidir: bool = True
    dropout: float = 0.1
    proj_size: int = 0

    def build_layer(self):
        if self.rnn_type == "srnn":
            rnn = nn.RNN(input_size=self.in_channels,
                         hidden_size=self.out_channels,
                         num_layers=self.num_layers,
                         nonlinearity=self.activation,
                         batch_first=True,
                         dropout=self.dropout,
                         bidirectional=self.bidir)
        elif self.rnn_type == "lstm":
            rnn = nn.LSTM(input_size=self.in_channels,
                          hidden_size=self.out_channels,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=self.bidir,
                          proj_size=self.proj_size)
        elif self.rnn_type == "gru":
            rnn = nn.GRU(input_size=self.in_channels,
                         hidden_size=self.out_channels,
                         num_layers=self.num_layers,
                         batch_first=True,
                         dropout=self.dropout,
                         bidirectional=self.bidir)
        return rnn


class RNNEncoderSettings(ConfiguredModel):
    in_channels: int = 64
    out_channels: int = 64
    rnn_type: str = Field(default="gru", pattern=r"srnn|lstm|gru")
    num_layers: int = 1
    activation: str = Field(default="tanh", pattern=r"tanh|relu")
    bidir: bool = True
    dropout: float = 0.1
    proj_size: int = 0
    apply_mask: bool = True

    def build_layer(self):
        return RNNEncoder(self)


class RNNEncoder(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, RNNEncoderSettings)
        self.settings = settings

        rnn_settings = settings.model_dump(exclude={"apply_mask"})
        rnn_settings = RNNSettings.model_validate(rnn_settings)
        self.rnn = rnn_settings.build_layer()
        self.num_layers = settings.num_layers
        self.apply_mask = settings.apply_mask

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


class RNNISLRSettings(ConfiguredModel):
    in_channels: int = 64
    hidden_channels: int = 64
    out_channels: int = 64
    masking_type: str = Field(default="both", pattern=r"none|rnn|head|both")
    head_type: str = Field(default="gpool", pattern=r"gpool|last_state")

    rnnen_settings: RNNEncoderSettings = Field(
        default_factory=lambda: RNNEncoderSettings())
    att_settings: TemporalAttentionSettings = Field(
        default_factory=lambda: TemporalAttentionSettings())
    head_settings: GPoolRecognitionHeadSettings = Field(
        default_factory=lambda: GPoolRecognitionHeadSettings())

    def model_post_init(self, __context):
        # Adjust rnn_settings.
        self.rnnen_settings.in_channels = self.hidden_channels
        self.rnnen_settings.out_channels = self.hidden_channels
        apply_mask = True if self.masking_type in ["rnn", "both"] else False
        self.rnnen_settings.apply_mask = apply_mask
        # Adjust att_settings.
        if self.rnnen_settings.bidir:
            self.att_settings.in_channels = self.hidden_channels * 2
        else:
            self.att_settings.in_channels = self.hidden_channels
        # Adjust head_settings.
        if self.rnnen_settings.bidir:
            self.head_settings.in_channels = self.hidden_channels * 2
        else:
            self.head_settings.in_channels = self.hidden_channels
        self.head_settings.out_channels = self.out_channels

        # Propagate.
        self.rnnen_settings.model_post_init(__context)
        self.att_settings.model_post_init(__context)
        self.head_settings.model_post_init(__context)

    def build_layer(self):
        return RNNISLR(self)


class RNNISLR(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, RNNISLRSettings)
        self.settings = settings

        self.linear = nn.Linear(settings.in_channels, settings.hidden_channels)
        self.activation = nn.ReLU()

        self.rnn = settings.rnnen_settings.build_layer()
        self.att = settings.att_settings.build_layer()
        self.head = settings.head_settings.build_layer()

        self.attw = None
        self.masking_type = settings.masking_type
        self.head_type = settings.head_type

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


class LuongDotAttentionEnergy(nn.Module):
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


class LuongGeneralAttentionEnergy(nn.Module):
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


class SingleHeadAttentionSettings(ConfiguredModel):
    key_dim: int = 64
    query_dim: int = 64
    att_dim: int = 64
    add_bias: bool = True
    att_type: str = Field(default="bahdanau",
        pattern=r"bahdanau|luong_dot|luong_general")

    def build_layer(self):
        return SingleHeadAttention(self)


class SingleHeadAttention(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, SingleHeadAttentionSettings)
        self.settings = settings

        if settings.att_type == "bahdanau":
            self.att_energy = BahdanauAttentionEnergy(
                key_dim=settings.key_dim,
                query_dim=settings.query_dim,
                att_dim=settings.att_dim,
                add_bias=settings.add_bias)
        elif settings.att_type == "luong_dot":
            self.att_energy = LuongDotAttentionEnergy()
        elif settings.att_type == "luong_general":
            self.att_energy = LuongGeneralAttentionEnergy(
                key_dim=settings.key_dim,
                query_dim=settings.query_dim,
                add_bias=settings.add_bias)

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


class RNNDecoderSettings(ConfiguredModel):
    dec_type: str = Field(default="bahdanau", pattern=r"bahdanau|luong")
    in_channels: int = 64
    hidden_channels: int = 64
    out_channels: int = 64

    # nn.Embedding
    num_embeddings: int = 64
    emb_channels: int = 4
    padding_idx: int = 0

    rnn_settings: RNNSettings = Field(default_factory=lambda: RNNSettings())
    att_settings: SingleHeadAttentionSettings = Field(
        default_factory=lambda: SingleHeadAttentionSettings())

    def model_post_init(self, __context):
        # Adjust emb_settings.
        self.num_embeddings = self.out_channels

        # Adjust att_settings.
        self.att_settings.key_dim = self.in_channels
        self.att_settings.query_dim = self.hidden_channels

        # Adjust rnn_settings.
        if self.dec_type == "bahdanau":
            self.rnn_settings.in_channels = self.in_channels + self.emb_channels
        elif self.dec_type == "luong":
            self.rnn_settings.in_channels = self.emb_channels

        self.rnn_settings.out_channels = self.hidden_channels
        # Force unidirection.
        self.rnn_settings.bidir = False

        # Propagate.
        self.rnn_settings.model_post_init(__context)
        self.att_settings.model_post_init(__context)

    def build_layer(self):
        if self.dec_type == "bahdanau":
            decoder = BahdanauRNNDecoder(self)
        elif self.dec_type == "luong":
            decoder = LuongRNNDecoder(self)
        return decoder


class BahdanauRNNDecoder(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, RNNDecoderSettings)
        self.settings = settings

        self.emb_layer = nn.Embedding(
            num_embeddings=settings.num_embeddings,
            embedding_dim=settings.emb_channels,
            padding_idx=settings.padding_idx)

        self.att_layer = settings.att_settings.build_layer()
        self.rnn = settings.rnn_settings.build_layer()

        self.head = nn.Linear(settings.hidden_channels, settings.out_channels)

        self.num_layers = settings.rnn_settings.num_layers
        self.dec_hstate = None
        self.attw = None

        self.reset_parameters(settings.emb_channels, settings.padding_idx)

    def reset_parameters(self, embedding_dim, padding_idx):
        # Bellow initialization has strong effect to performance.
        # Please refer.
        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py#L189
        nn.init.normal_(self.emb_layer.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(self.emb_layer.weight[padding_idx], 0)

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
                 settings):
        super().__init__()
        assert isinstance(settings, RNNDecoderSettings)
        self.settings = settings

        self.emb_layer = nn.Embedding(
            num_embeddings=settings.num_embeddings,
            embedding_dim=settings.emb_channels,
            padding_idx=settings.padding_idx)

        self.rnn = settings.rnn_settings.build_layer()
        self.att_layer = settings.att_settings.build_layer()

        self.head = nn.Linear(settings.hidden_channels * 2,  # hstate + cvec
                              settings.out_channels)

        self.num_layers = settings.rnn_settings.num_layers
        self.dec_hstate = None
        self.attw = None

        self.reset_parameters(settings.emb_channels, settings.padding_idx)

    def reset_parameters(self, embedding_dim, padding_idx):
        # Bellow initialization has strong effect to performance.
        # Please refer.
        # https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py#L189
        nn.init.normal_(self.emb_layer.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(self.emb_layer.weight[padding_idx], 0)

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

        emb_out = self.emb_layer(dec_inputs)
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


class RNNCSLRSettings(ConfiguredModel):
    in_channels: int = 64
    enc_hidden_channels: int = 64
    dec_hidden_channels: int = 64
    dec_att_channels: int = 64
    out_channels: int = 64

    rnnen_settings: RNNEncoderSettings = Field(
        default_factory=lambda: RNNEncoderSettings())
    rnnde_settings: RNNDecoderSettings = Field(
        default_factory=lambda: RNNDecoderSettings())

    def model_post_init(self, __context):
        # Adjust rnnen_settings.
        self.rnnen_settings.in_channels = self.enc_hidden_channels
        self.rnnen_settings.out_channels = self.enc_hidden_channels
        # Adjust rnnde_settings.
        self.rnnde_settings.out_channels = self.out_channels
        if self.rnnen_settings.bidir:
            self.rnnde_settings.in_channels = self.enc_hidden_channels * 2
            self.rnnde_settings.hidden_channels = self.dec_hidden_channels * 2
            self.rnnde_settings.att_settings.att_dim = self.dec_att_channels * 2

        # Propagate.
        self.rnnen_settings.model_post_init(__context)
        self.rnnde_settings.model_post_init(__context)

    def build_layer(self):
        return RNNCSLR(self)


class RNNCSLR(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, RNNCSLRSettings)
        self.settings = settings

        self.linear = nn.Linear(settings.in_channels, settings.enc_hidden_channels)
        self.enc_activation = nn.ReLU()

        self.encoder = settings.rnnen_settings.build_layer()
        self.decoder = settings.rnnde_settings.build_layer()

        self.enc_bidir = settings.rnnen_settings.bidir
        self.attws = None

    def _apply_encoder(self, feature, feature_pad_mask=None):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        feature = self.linear(feature)
        feature = self.enc_activation(feature)

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
        enc_seqs, enc_hstate = self._apply_encoder(feature, feature_pad_mask)

        # Apply decoder.
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
