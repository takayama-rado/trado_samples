#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""transformer: Transformer layers.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import math

# Third party's modules
import numpy as np

import torch

from pydantic import (
    Field)

from torch import nn
from torch.nn import functional as F

# Local modules
from .misc import (
    ConfiguredModel,
    GPoolRecognitionHeadSettings,
    Identity,
    apply_norm,
    create_norm)

from ..utils import (
    make_san_mask,
    make_causal_mask,
    select_reluwise_activation)


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class PositionalEncodingSettings(ConfiguredModel):
    dim_model: int = 64
    dropout: float = 0.1
    max_len: int = 5000

    def build_layer(self):
        return PositionalEncoding(self)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, PositionalEncodingSettings)
        self.settings = settings

        self.dim_model = settings.dim_model
        # Compute the positional encodings once in log space.
        pose = torch.zeros(settings.max_len, settings.dim_model,
            dtype=torch.float32)
        position = torch.arange(0, settings.max_len,
            dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, settings.dim_model, 2).float()
                             * -(math.log(10000.0) / settings.dim_model))
        pose[:, 0::2] = torch.sin(position * div_term)
        pose[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pose", pose)

        self.dropout = nn.Dropout(p=settings.dropout)

    def forward(self,
                feature):
        feature = feature + self.pose[None, :feature.shape[1], :]
        feature = self.dropout(feature)
        return feature


class MultiheadAttentionSettings(ConfiguredModel):
    key_dim: int = 64
    query_dim: int = 64
    att_dim: int = 64
    out_dim: int = 64
    num_heads: int = 1
    dropout: float = 0.1
    add_bias: bool = True

    def build_layer(self):
        return MultiheadAttention(self)


class MultiheadAttention(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, MultiheadAttentionSettings)
        self.settings = settings

        assert settings.att_dim % settings.num_heads == 0
        self.head_dim = settings.att_dim // settings.num_heads
        self.num_heads = settings.num_heads
        self.scale = math.sqrt(self.head_dim)

        self.w_key = nn.Linear(settings.key_dim, settings.att_dim,
            bias=settings.add_bias)
        self.w_value = nn.Linear(settings.key_dim, settings.att_dim,
            bias=settings.add_bias)
        self.w_query = nn.Linear(settings.query_dim, settings.att_dim,
            bias=settings.add_bias)

        self.w_out = nn.Linear(settings.att_dim, settings.out_dim, bias=settings.add_bias)

        self.dropout_attn = nn.Dropout(p=settings.dropout)

        self.neg_inf = None

        self.qkv_same_dim = settings.key_dim == settings.query_dim
        self.reset_parameters(settings.add_bias)

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


class PositionwiseFeedForwardSettings(ConfiguredModel):
    dim_model: int = 64
    dim_pffn: int = 256
    dropout: float = 0.1
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    add_bias: bool = True

    def build_layer(self):
        return PositionwiseFeedForward(self)


class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 settings):
       super().__init__()
       assert isinstance(settings, PositionwiseFeedForwardSettings)

       self.w_1 = nn.Linear(settings.dim_model, settings.dim_pffn,
           bias=settings.add_bias)
       self.w_2 = nn.Linear(settings.dim_pffn, settings.dim_model,
           bias=settings.add_bias)

       self.dropout = nn.Dropout(p=settings.dropout)

       self.activation = select_reluwise_activation(settings.activation)

    def forward(self, feature):
        feature = self.w_1(feature)
        feature = self.activation(feature)
        feature = self.dropout(feature)
        feature = self.w_2(feature)
        return feature


class TransformerEncoderLayerSettings(ConfiguredModel):
    dim_model: int = 64
    dim_pffn: int = 256
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    norm_type_sattn: str = Field(default="layer", pattern=r"layer|batch")
    norm_type_pffn: str = Field(default="layer", pattern=r"layer|batch")
    norm_eps: float = 1e-5
    norm_first: bool = True
    dropout: float = 0.1

    mhsa_settings: MultiheadAttentionSettings = Field(
        default_factory=lambda: MultiheadAttentionSettings())
    pffn_settings: PositionwiseFeedForwardSettings = Field(
        default_factory=lambda: PositionwiseFeedForwardSettings())

    def model_post_init(self, __context):
        # Adjust mhsa_settings.
        self.mhsa_settings.key_dim = self.dim_model
        self.mhsa_settings.query_dim = self.dim_model
        self.mhsa_settings.att_dim = self.dim_model
        self.mhsa_settings.out_dim = self.dim_model
        # Adjust pffn_settings.
        self.pffn_settings.dim_model = self.dim_model
        self.pffn_settings.dim_pffn = self.dim_pffn
        self.pffn_settings.activation = self.activation

        # Propagate.
        self.mhsa_settings.model_post_init(__context)
        self.pffn_settings.model_post_init(__context)

    def build_layer(self):
        if self.norm_first:
            layer = PreNormTransformerEncoderLayer(self)
        else:
            layer = PostNormTransformerEncoderLayer(self)
        return layer


def create_encoder_mask(src_key_padding_mask,
                        causal_mask):
    if src_key_padding_mask is not None:
        san_mask = make_san_mask(src_key_padding_mask, causal_mask)
    elif causal_mask is not None:
        san_mask = causal_mask
    else:
        san_mask = None
    return san_mask


class PreNormTransformerEncoderLayer(nn.Module):
    """Pre-normalization structure.

    For the details, please refer
    https://arxiv.org/pdf/2002.04745v1.pdf
    """
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, TransformerEncoderLayerSettings)
        assert settings.norm_first is True
        self.settings = settings

        #################################################
        # MHSA.
        #################################################
        self.norm_sattn = create_norm(
            settings.norm_type_sattn, settings.dim_model, settings.norm_eps,
            settings.mhsa_settings.add_bias)
        self.self_attn = settings.mhsa_settings.build_layer()

        #################################################
        # PFFN.
        #################################################
        self.norm_pffn = create_norm(settings.norm_type_pffn, settings.dim_model,
            settings.norm_eps, settings.pffn_settings.add_bias)
        self.pffn = settings.pffn_settings.build_layer()

        self.dropout = nn.Dropout(p=settings.dropout)

        # To store attention weights.
        self.attw = None

    def forward(self,
                feature,
                causal_mask=None,
                src_key_padding_mask=None):
        bsize, qlen = feature.shape[:2]
        san_mask = create_encoder_mask(src_key_padding_mask, causal_mask)

        #################################################
        # MHSA
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
        # PFFN
        #################################################
        residual = feature
        # `[N, qlen, dim_model]`
        feature = apply_norm(self.norm_pffn, feature)
        feature = self.pffn(feature)
        feature = self.dropout(feature) + residual

        return feature


class PostNormTransformerEncoderLayer(nn.Module):
    """Post-normalization structure (Standard).

    For the details, please refer
    https://arxiv.org/pdf/2002.04745v1.pdf
    """
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, TransformerEncoderLayerSettings)
        assert settings.norm_first is False
        self.settings = settings

        #################################################
        # MHSA.
        #################################################
        self.self_attn = settings.mhsa_settings.build_layer()
        self.norm_sattn = create_norm(
            settings.norm_type_sattn, settings.dim_model, settings.norm_eps,
            settings.mhsa_settings.add_bias)

        #################################################
        # PFFN.
        #################################################
        self.pffn = settings.pffn_settings.build_layer()
        self.norm_pffn = create_norm(settings.norm_type_pffn, settings.dim_model,
            settings.norm_eps, settings.pffn_settings.add_bias)

        self.dropout = nn.Dropout(p=settings.dropout)

        # To store attention weights.
        self.attw = None

    def forward(self,
                feature,
                causal_mask=None,
                src_key_padding_mask=None):
        bsize, qlen = feature.shape[:2]
        san_mask = create_encoder_mask(src_key_padding_mask, causal_mask)

        #################################################
        # MHSA
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
        # PFFN
        #################################################
        residual = feature
        # `[N, qlen, dim_model]`
        feature = self.pffn(feature)
        feature = self.dropout(feature) + residual
        feature = apply_norm(self.norm_pffn, feature)

        return feature


class TransformerEncoderSettings(ConfiguredModel):
    num_layers: int = 1
    norm_type_tail: str = Field(default="layer", pattern=r"layer|batch")
    norm_eps: float = 1e-5
    add_bias: bool = True
    add_tailnorm: bool = True

    pe_settings: PositionalEncodingSettings = Field(
        default_factory=lambda: PositionalEncodingSettings())

    def build_layer(self, encoder_layer):
        return TransformerEncoder(self, encoder_layer)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 settings,
                 encoder_layer):
        super().__init__()
        assert isinstance(settings, TransformerEncoderSettings)
        dim_model = settings.pe_settings.dim_model
        assert dim_model == encoder_layer.settings.dim_model
        self.settings = settings

        self.pos_encoder = settings.pe_settings.build_layer()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _
            in range(settings.num_layers)])

        # Add LayerNorm at tail position.
        # This is applied only for pre-normalization structure because
        # post-normalization structure includes tail-normalization in encoder
        # layers.
        add_tailnorm0 = settings.add_tailnorm
        add_tailnorm1 = not isinstance(encoder_layer, PostNormTransformerEncoderLayer)
        if add_tailnorm0 and add_tailnorm1:
            self.norm_tail = create_norm(settings.norm_type_tail, dim_model,
                settings.norm_eps, settings.add_bias)
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


class TransformerEnISLRSettings(ConfiguredModel):
    in_channels: int = 64
    inter_channels: int = 64
    out_channels: int = 64
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    pooling_type: str = Field(default="none", pattern=r"none|average|max")

    enlayer_settings: TransformerEncoderLayerSettings = Field(
        default_factory=lambda: TransformerEncoderLayerSettings())
    encoder_settings: TransformerEncoderSettings = Field(
        default_factory=lambda: TransformerEncoderSettings())

    head_settings: GPoolRecognitionHeadSettings = Field(
        default_factory=lambda: GPoolRecognitionHeadSettings())

    def model_post_init(self, __context):
        # Adjust enlayer_settings.
        self.enlayer_settings.dim_model = self.inter_channels
        self.enlayer_settings.activation = self.activation

        # Adjust head_settings.
        self.head_settings.in_channels = self.inter_channels
        self.head_settings.out_channels = self.out_channels

        # Propagate.
        self.enlayer_settings.model_post_init(__context)
        self.encoder_settings.model_post_init(__context)
        self.head_settings.model_post_init(__context)

    def build_layer(self):
        return TransformerEnISLR(self)


def build_pooling_layer(pooling_type):
    if pooling_type == "none":
        pooling = Identity()
    elif pooling_type == "average":
        pooling = nn.AvgPool2d(
            kernel_size=[2, 1],
            stride=[2, 1],
            padding=0)
    elif pooling_type == "max":
        pooling = nn.MaxPool2d(
            kernel_size=[2, 1],
            stride=[2, 1],
            padding=0)
    return pooling


class TransformerEnISLR(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, TransformerEnISLRSettings)
        self.settings = settings

        # Feature extraction.
        self.linear = nn.Linear(settings.in_channels, settings.inter_channels)
        self.activation = select_reluwise_activation(settings.activation)

        self.pooling = build_pooling_layer(settings.pooling_type)

        # Transformer-Encoder.
        enlayer = settings.enlayer_settings.build_layer()
        self.tr_encoder = settings.encoder_settings.build_layer(enlayer)

        self.head = settings.head_settings.build_layer()

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
            if feature_pad_mask.shape[-1] != feature.shape[1]:
                feature_pad_mask = F.interpolate(
                    feature_pad_mask.unsqueeze(1).float(),
                    feature.shape[1],
                    mode="nearest")
                feature_pad_mask = feature_pad_mask.squeeze(1) > 0.5
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


class TransformerDecoderLayerSettings(ConfiguredModel):
    dim_model: int = 64
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    norm_type_sattn: str = Field(default="layer", pattern=r"layer|batch")
    norm_type_cattn: str = Field(default="layer", pattern=r"layer|batch")
    norm_type_pffn: str = Field(default="layer", pattern=r"layer|batch")
    norm_eps: float = 1e-5
    norm_first: bool = True
    dropout: float = 0.1

    mhsa_settings: MultiheadAttentionSettings = Field(
        default_factory=lambda: MultiheadAttentionSettings())
    mhca_settings: MultiheadAttentionSettings = Field(
        default_factory=lambda: MultiheadAttentionSettings())
    pffn_settings: PositionwiseFeedForwardSettings = Field(
        default_factory=lambda: PositionwiseFeedForwardSettings())

    def model_post_init(self, __context):
        # Adjust mhsa_settings.
        self.mhsa_settings.key_dim = self.dim_model
        self.mhsa_settings.query_dim = self.dim_model
        self.mhsa_settings.att_dim = self.dim_model
        self.mhsa_settings.out_dim = self.dim_model
        # Adjust mhca_settings.
        self.mhca_settings.key_dim = self.dim_model
        self.mhca_settings.query_dim = self.dim_model
        self.mhca_settings.att_dim = self.dim_model
        self.mhca_settings.out_dim = self.dim_model
        # Adjust pffn_settings.
        self.pffn_settings.dim_model = self.dim_model
        self.pffn_settings.activation = self.activation

        # Propagate.
        self.mhsa_settings.model_post_init(__context)
        self.mhca_settings.model_post_init(__context)
        self.pffn_settings.model_post_init(__context)

    def build_layer(self):
        if self.norm_first:
            layer = PreNormTransformerDecoderLayer(self)
        else:
            layer = PostNormTransformerDecoderLayer(self)
        return layer


def create_decoder_mask(tgt_feature, enc_feature,
                        tgt_key_padding_mask, tgt_causal_mask,
                        enc_key_padding_mask, enc_tgt_causal_mask):
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
    return tgt_san_mask, enc_tgt_mask


class PreNormTransformerDecoderLayer(nn.Module):
    """Pre-normalization structure.

    For the details, please refer
    https://arxiv.org/pdf/2002.04745v1.pdf
    """
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, TransformerDecoderLayerSettings)
        assert settings.norm_first is True
        self.settings = settings

        #################################################
        # MHSA.
        #################################################
        self.norm_sattn = create_norm(
            settings.norm_type_sattn, settings.dim_model, settings.norm_eps,
            settings.mhsa_settings.add_bias)
        self.self_attn = settings.mhsa_settings.build_layer()

        #################################################
        # MHCA.
        #################################################
        self.norm_cattn = create_norm(
            settings.norm_type_cattn, settings.dim_model, settings.norm_eps,
            settings.mhca_settings.add_bias)
        self.cross_attn = settings.mhca_settings.build_layer()

        #################################################
        # PFFN.
        #################################################
        self.norm_pffn = create_norm(
            settings.norm_type_pffn, settings.dim_model, settings.norm_eps,
            settings.pffn_settings.add_bias)
        self.pffn = settings.pffn_settings.build_layer()

        self.dropout = nn.Dropout(p=settings.dropout)

        # To store attention weights.
        self.sattw = None
        self.cattw = None

    def forward(self,
                tgt_feature,
                enc_feature,
                tgt_causal_mask=None,
                enc_tgt_causal_mask=None,
                tgt_key_padding_mask=None,
                enc_key_padding_mask=None):

        # Create mask.
        tgt_san_mask, enc_tgt_mask = create_decoder_mask(
            tgt_feature, enc_feature,
            tgt_key_padding_mask, tgt_causal_mask,
            enc_key_padding_mask, enc_tgt_causal_mask)

        #################################################
        # MHSA
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
        # MHCA
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
        # PFFN
        #################################################
        residual = tgt_feature
        tgt_feature = apply_norm(self.norm_pffn, tgt_feature)
        tgt_feature = self.pffn(tgt_feature)
        tgt_feature = self.dropout(tgt_feature) + residual
        return tgt_feature


class PostNormTransformerDecoderLayer(nn.Module):
    """Post-normalization structure (Standard).

    For the details, please refer
    https://arxiv.org/pdf/2002.04745v1.pdf
    """
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, TransformerDecoderLayerSettings)
        assert settings.norm_first is False
        self.settings = settings

        #################################################
        # MHSA.
        #################################################
        self.self_attn = settings.mhsa_settings.build_layer()
        self.norm_sattn = create_norm(
            settings.norm_type_sattn, settings.dim_model, settings.norm_eps,
            settings.mhsa_settings.add_bias)

        #################################################
        # MHCA.
        #################################################
        self.cross_attn = settings.mhca_settings.build_layer()
        self.norm_cattn = create_norm(
            settings.norm_type_cattn, settings.dim_model, settings.norm_eps,
            settings.mhca_settings.add_bias)

        #################################################
        # PFFN.
        #################################################
        self.pffn = settings.pffn_settings.build_layer()
        self.norm_pffn = create_norm(
            settings.norm_type_pffn, settings.dim_model, settings.norm_eps,
            settings.pffn_settings.add_bias)

        self.dropout = nn.Dropout(p=settings.dropout)

        # To store attention weights.
        self.sattw = None
        self.cattw = None

    def forward(self,
                tgt_feature,
                enc_feature,
                tgt_causal_mask=None,
                enc_tgt_causal_mask=None,
                tgt_key_padding_mask=None,
                enc_key_padding_mask=None):

        # Create mask.
        tgt_san_mask, enc_tgt_mask = create_decoder_mask(
            tgt_feature, enc_feature,
            tgt_key_padding_mask, tgt_causal_mask,
            enc_key_padding_mask, enc_tgt_causal_mask)

        #################################################
        # MHSA
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
        # MHCA
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
        # PFFN
        #################################################
        residual = tgt_feature
        tgt_feature = self.pffn(tgt_feature)
        tgt_feature = self.dropout(tgt_feature) + residual
        tgt_feature = apply_norm(self.norm_pffn, tgt_feature)

        return tgt_feature


class TransformerDecoderSettings(ConfiguredModel):
    out_channels: int = 100
    num_layers: int = 1
    norm_type_tail: str = Field(default="layer", pattern=r"layer|batch")
    norm_eps: float = 1e-5
    add_bias: bool = True
    add_tailnorm: bool = True
    padding_idx: int = 0

    pe_settings: PositionalEncodingSettings = Field(
        default_factory=lambda: PositionalEncodingSettings())

    def build_layer(self, decoder_layer):
        return TransformerDecoder(self, decoder_layer)


class TransformerDecoder(nn.Module):
    def __init__(self,
                 settings,
                 decoder_layer):
        super().__init__()
        assert isinstance(settings, TransformerDecoderSettings)
        dim_model = settings.pe_settings.dim_model
        assert dim_model == decoder_layer.settings.dim_model
        self.settings = settings

        self.emb_layer = nn.Embedding(settings.out_channels,
                                      dim_model,
                                      padding_idx=settings.padding_idx)
        self.vocab_size = settings.out_channels

        self.pos_encoder = settings.pe_settings.build_layer()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _
            in range(settings.num_layers)])

        # This is applied only for pre-normalization structure because
        # post-normalization structure includes tail-normalization in encoder
        # layers.
        add_tailnorm0 = settings.add_tailnorm
        add_tailnorm1 = not isinstance(decoder_layer, PostNormTransformerDecoderLayer)
        if add_tailnorm0 and add_tailnorm1:
            self.norm_tail = create_norm(settings.norm_type_tail, dim_model,
                settings.norm_eps, settings.add_bias)
        else:
            self.norm_tail = Identity()

        self.head = nn.Linear(dim_model, settings.out_channels)

        self.reset_parameters(dim_model, settings.padding_idx)

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


class TransformerCSLRSettings(ConfiguredModel):
    in_channels: int = 64
    inter_channels: int = 64
    out_channels: int = 64
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    padding_idx: int = 0

    enlayer_settings: TransformerEncoderLayerSettings = Field(
        default_factory=lambda: TransformerEncoderLayerSettings())
    encoder_settings: TransformerEncoderSettings = Field(
        default_factory=lambda: TransformerEncoderSettings())

    delayer_settings: TransformerDecoderLayerSettings = Field(
        default_factory=lambda: TransformerDecoderLayerSettings())
    decoder_settings: TransformerDecoderSettings = Field(
        default_factory=lambda: TransformerDecoderSettings())

    def model_post_init(self, __context):
        # Adjust enlayer_settings.
        self.enlayer_settings.dim_model = self.inter_channels
        self.enlayer_settings.activation = self.activation
        # Adjust delayer_settings.
        self.delayer_settings.dim_model = self.inter_channels
        self.delayer_settings.activation = self.activation
        # Adjust decoder_settings.
        self.decoder_settings.padding_idx = self.padding_idx

        # Propagate.
        self.enlayer_settings.model_post_init(__context)
        self.encoder_settings.model_post_init(__context)
        self.delayer_settings.model_post_init(__context)
        self.decoder_settings.model_post_init(__context)

    def build_layer(self):
        return TransformerCSLR(self)


class TransformerCSLR(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, TransformerCSLRSettings)
        self.settings = settings

        # Feature extraction.
        self.linear = nn.Linear(settings.in_channels, settings.inter_channels)
        self.activation = select_reluwise_activation(settings.activation)

        # Transformer-Encoder.
        enlayer = settings.enlayer_settings.build_layer()
        self.tr_encoder = settings.encoder_settings.build_layer(enlayer)

        # Transformer-Decoder.
        delayer = settings.delayer_settings.build_layer()
        self.tr_decoder = settings.decoder_settings.build_layer(delayer)

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
                  max_seqlen=62):
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


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
