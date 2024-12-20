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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field)

from torch import nn
from torch.nn import functional as F

# Local modules
from modules_gislr.layers import maskednorm as mn

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
    elif norm_type == "batch2d":
        norm = nn.BatchNorm2d(dim_model, eps=eps)
    elif norm_type == "masked_batch":
        ntype1d, _ = mn.select_norm_type([norm_type])
        norm = mn.create_norm_layer(ntype1d, dim_model)
    elif norm_type == "masked_batch2d":
        _, ntype2d = mn.select_norm_type([norm_type])
        norm = mn.create_norm_layer(ntype2d, dim_model)
    return norm


def apply_norm(norm_layer, feature, mask=None, channel_first=False):
    shape = feature.shape
    if len(shape) > 5:
        raise NotImplementedError("Unsupported feature shape:{shape}")
    if isinstance(norm_layer, nn.LayerNorm):
        if channel_first:
            if len(shape) == 3:
                # `[N, C, *] -> [N, *, C] -> [N, C, *]`
                feature = feature.permute([0, 2, 1]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 2, 1]).contiguous()
            elif len(shape) == 4:
                # `[N, C, *, *] -> [N, *, *, C] -> [N, C, *, *]`
                feature = feature.permute([0, 2, 3, 1]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 3, 1, 2]).contiguous()
            elif len(shape) == 5:
                # `[N, C, *, *, *] -> [N, *, *, *, C] -> [N, C, *, *, *]`
                feature = feature.permute([0, 2, 3, 4, 1]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 4, 1, 2, 3]).contiguous()
        else:
            feature = norm_layer(feature)
    elif isinstance(norm_layer, nn.BatchNorm1d):
        if channel_first:
            feature = norm_layer(feature)
        else:
            if len(shape) == 3:
                # `[N, *, C] -> [N, C, *] -> [N, *, C]`
                feature = feature.permute([0, 2, 1]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 2, 1]).contiguous()
            elif len(shape) == 4:
                # `[N, *, *, C] -> [N, C, *, *] -> [N, *, *, C]`
                feature = feature.permute([0, 3, 1, 2]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 2, 3, 1]).contiguous()
            elif len(shape) == 5:
                # `[N, *, *, C] -> [N, C, *, *] -> [N, *, *, C]`
                feature = feature.permute([0, 4, 1, 2, 3]).contiguous()
                feature = norm_layer(feature)
                feature = feature.permute([0, 2, 3, 4, 1]).contiguous()
    elif isinstance(norm_layer, mn.MaskedSwitchNorm1d):
        feature = mn.apply_normalization(norm_layer, feature, mask=mask,
            channel_last=not channel_first)
    elif isinstance(norm_layer, mn.MaskedSwitchNorm2d):
        feature = mn.apply_normalization(norm_layer, feature, mask=mask,
            channel_last=not channel_first)
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
        return torch.zeros([1], dtype=feature.dtype, device=feature.device)


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


class ConfiguredModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalAttentionSettings(ConfiguredModel):
    in_channels: int = 64
    attention_type: str = Field(default="sigmoid", pattern=r"none|sigmoid|softmax")
    post_scale: bool = False

    def build_layer(self):
        if self.attention_type == "none":
            att = Identity()
        else:
            att = TemporalAttention(self)
        return att


class TemporalAttention(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, TemporalAttentionSettings)
        self.settings = settings

        self.linear = nn.Linear(settings.in_channels, 1)
        self.attention_type = settings.attention_type

        if settings.attention_type == "sigmoid":
            self.scale_layer = nn.Sigmoid()
        elif settings.attention_type == "softmax":
            self.scale_layer = nn.Softmax(dim=1)

        self.neg_inf = None
        self.post_scale = settings.post_scale

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


class GPoolRecognitionHeadSettings(ConfiguredModel):
    in_channels: int = 64
    out_channels: int = 64
    dropout: float = 0.0
    pooling_mode: str = Field(default="avg", pattern=r"avg|max")

    def build_layer(self):
        return GPoolRecognitionHead(self)


class GPoolRecognitionHead(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, GPoolRecognitionHeadSettings)
        self.settings = settings
        self.pooling_mode = settings.pooling_mode

        self.dropout = nn.Dropout(p=0)

        self.head = nn.Linear(settings.in_channels, settings.out_channels)
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.head.weight,
                        mean=0.,
                        std=math.sqrt(1. / self.settings.out_channels))

    def forward(self, feature, feature_pad_mask=None):
        # Averaging over temporal axis.
        # `[N, C, T] -> [N, C, 1] -> [N, C]`
        if feature_pad_mask is not None:
            tlength = feature_pad_mask.sum(dim=-1)
            feature = feature * feature_pad_mask.unsqueeze(1)
            if self.pooling_mode == "avg":
                feature = feature.sum(dim=-1) / tlength.unsqueeze(-1)
            if self.pooling_mode == "max":
                feature = F.max_pool1d(feature, kernel_size=feature.shape[-1])
        else:
            if self.pooling_mode == "avg":
                feature = F.avg_pool1d(feature, kernel_size=feature.shape[-1])
            elif self.pooling_mode == "max":
                feature = F.max_pool1d(feature, kernel_size=feature.shape[-1])
        feature = feature.reshape(feature.shape[0], -1)

        # Dropout.
        feature = self.dropout(feature)

        # Predict.
        feature = self.head(feature)
        return feature


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
