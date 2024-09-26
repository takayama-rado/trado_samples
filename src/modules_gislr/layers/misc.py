#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""misc: Miscellaneous layers.
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
from inspect import signature

# Third party's modules
import numpy as np


import torch
from torch import nn

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


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


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
