#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""conformer: Conformer layers.
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

# Third party's modules
import torch
from torch import nn

# Local modules
from .misc import (
    GPoolRecognitionHead,
    Identity,
    apply_norm,
    create_norm)
from .transformer import (
    MultiheadAttention,
    PositionwiseFeedForward,
    TransformerEncoder)

from ..utils import (
    make_san_mask,
    select_reluwise_activation)


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


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
