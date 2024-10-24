#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""feature_extraction: Feature extraction layer.
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

from typing import (
    Tuple)

# Third party's modules
import torch

from pydantic import (
    Field)

from torch import nn
from torch.nn import functional as F

# Local modules
from .misc import (
    ConfiguredModel,
    Identity,
    Zero,
    apply_norm,
    create_norm)

from ..utils import (
    select_reluwise_activation)

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


def create_fext_module(settings):
    if settings is None:
        fext_module = nn.ModuleList(
            [Identity()])
    else:
        fext_module = nn.ModuleList(
            [setting.build_layer() for setting in settings])
    return fext_module


class LinearFeatureExtractorSettings(ConfiguredModel):
    fext_type: str = "linear"
    in_channels: int = 64
    out_channels: int = 64
    norm_type: str = Field(default="batch", pattern=r"batch|batch2d|layer")
    norm_eps: float = 1e-5
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    add_bias: bool = True
    dropout: float = 0.1
    channel_first: bool = True
    add_residual: bool = True

    def build_layer(self):
        return LinearFeatureExtractor(self)


class LinearFeatureExtractor(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, LinearFeatureExtractorSettings)
        self.settings = settings
        self.channel_first = settings.channel_first

        self.linear = nn.Linear(
            in_features=settings.in_channels,
            out_features=settings.out_channels,
            bias=settings.add_bias)

        self.norm = create_norm(
            settings.norm_type, settings.out_channels, settings.norm_eps,
            settings.add_bias)

        self.activation = select_reluwise_activation(settings.activation)

        self.dropout = nn.Dropout(p=settings.dropout)

        if settings.add_residual and settings.in_channels == settings.out_channels:
            self.residual = Identity()
        else:
            self.residual = Zero()

    def forward(self,
                feature):
        shape = feature.shape
        if self.channel_first:
            # `[N, C, T] -> [N, T, C]`
            if len(shape) == 3:
                feature = feature.permute([0, 2, 1])
            # `[N, C, T, *] -> [N, T, C, *] -> [N, T, C']`
            elif len(shape) == 4:
                feature = feature.permute([0, 2, 1, 3])
                feature = feature.reshape([shape[0], shape[2], -1])
            # `[N, C, T, *, *] -> [N, T, C, *, *] -> [N, T, C']`
            elif len(shape) == 5:
                feature = feature.permute([0, 2, 1, 3, 4])
                feature = feature.reshape([shape[0], shape[2], -1])
            else:
                raise NotImplementedError(f"Unsupported feature shape:{shape}.")
        res = self.residual(feature)
        feature = self.linear(feature)
        feature = apply_norm(self.norm, feature, channel_first=False)
        feature = self.activation(feature)
        feature = self.dropout(feature)
        feature = feature + res
        if self.channel_first:
            # `[N, T, C] -> [N, C, T]`
            feature = feature.permute([0, 2, 1])
        return feature


class EfficientChannelAttention(nn.Module):
    def __init__(self,
                 spatial_channels=1,
                 kernel_size=3):
        super().__init__()
        if spatial_channels == 1:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif spatial_channels == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            message = f"ECA for {spatial_channels} spatial channels is not implemented."
            raise NotImplementedError(message)

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
            padding=(kernel_size-1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.spatial_channels = spatial_channels

    def _forward_sc1(self,
                     feature):
        # `[N, C, T] -> [N, C, 1] -> [N, 1, C]`
        att = self.avg_pool(feature)
        att = att.permute([0, 2, 1])
        att = self.conv(att)
        # `[N, 1, C] -> [N, C, 1]`
        att = att.permute([0, 2, 1])
        return att

    def _forward_sc2(self,
                     feature):
        # `[N, C, T, J] -> [N, C, 1, 1] -> [N, 1, C]`
        att = self.avg_pool(feature)
        att = att.squeeze(-1)
        att = att.permute([0, 2, 1])
        att = self.conv(att)
        # `[N, 1, C] -> [N, C, 1, 1]`
        att = att.permute([0, 2, 1])
        att = att.unsqueeze(-1)
        return att

    def forward(self,
                feature):
        if self.spatial_channels == 1:
            att = self._forward_sc1(feature)
        elif self.spatial_channels == 2:
            att = self._forward_sc2(feature)
        att = self.sigmoid(att)
        return feature * att


class CNN1DFeatureExtractorSettings(ConfiguredModel):
    fext_type: str = "cnn1d"
    in_channels: int = 64
    out_channels: int = 64
    # Conv1D settings.
    kernel_size: int = Field(default=3, ge=3)
    stride: int = Field(default=1, ge=1)
    padding_mode: str = "zeros"
    attention_type: str = Field(default="none",
        pattern=r"none|glu|eca")
    conv_type: str = Field(default="separable",
        pattern=r"separable|standard|predepth")
    norm_type: str = Field(default="batch",
        pattern=r"layer|batch")
    norm_eps: float = 1e-5
    activation: str = Field(default="swish",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    causal: bool | None = False
    add_residual: bool = True
    add_bias: bool = True
    dropout: float = 0.1

    def model_post_init(self, __context):
        message = f"kernel_size:{self.kernel_size} must be the odd number."
        assert (self.kernel_size - 1) % 2 == 0, message

    def build_layer(self):
        return CNN1DFeatureExtractor(self)


class CNN1DFeatureExtractor(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, CNN1DFeatureExtractorSettings)
        self.settings = settings
        self.causal = settings.causal
        self.channel_expand = 2 if settings.attention_type == "glu" else 1

        if settings.causal:
            self.padding = (settings.kernel_size - 1)
        else:
            self.padding = (settings.kernel_size - 1) // 2

        self.conv_module = self._build_conv_module(settings, self.padding)
        self.norm = create_norm(settings.norm_type, settings.out_channels, settings.norm_eps)
        self.activation = select_reluwise_activation(settings.activation)
        self.dropout = nn.Dropout(p=settings.dropout)

        if settings.add_residual:
            if settings.in_channels == settings.out_channels:
                if settings.stride == 1:
                    self.residual = Identity()
                else:
                    self.residual = nn.MaxPool1d(
                        kernel_size=settings.kernel_size,
                        stride=settings.stride,
                        padding=self.padding)
            else:
                self.residual = nn.Conv1d(
                    in_channels=settings.in_channels,
                    out_channels=settings.out_channels,
                    kernel_size=1,
                    stride=settings.stride)
        else:
            self.residual = Zero()

    def _build_separable_conv(self, settings, attention, padding):
        dict_modules = collections.OrderedDict([
            # Point-wise.
            ("pconv",
             nn.Conv1d(
                in_channels=settings.in_channels,
                out_channels=settings.out_channels * self.channel_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=settings.add_bias)),
            ("attention", attention),
            # Depth-wise
            ("dconv",
             nn.Conv1d(
                in_channels=settings.out_channels,
                out_channels=settings.out_channels,
                kernel_size=settings.kernel_size,
                stride=settings.stride,
                padding=padding,
                groups=settings.out_channels,
                padding_mode=settings.padding_mode,
                bias=settings.add_bias))])

        conv_module = nn.Sequential(dict_modules)
        return conv_module

    def _build_predepth_conv(self, settings, attention, padding):
        dict_modules = collections.OrderedDict([
            # Depth-wise
            ("dconv",
             nn.Conv1d(
                in_channels=settings.in_channels,
                out_channels=settings.out_channels,
                kernel_size=settings.kernel_size,
                stride=settings.stride,
                padding=padding,
                groups=settings.out_channels,
                padding_mode=settings.padding_mode,
                bias=settings.add_bias)),
            # Point-wise.
            ("pconv",
             nn.Conv1d(
                in_channels=settings.out_channels,
                out_channels=settings.out_channels * self.channel_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=settings.add_bias)),
            ("attention", attention)])
        conv_module = nn.Sequential(dict_modules)
        return conv_module

    def _build_standard_conv(self, settings, attention, padding):
        dict_modules = collections.OrderedDict([
            ("conv",
             nn.Conv1d(
                 in_channels=settings.in_channels,
                 out_channels=settings.out_channels * self.channel_expand,
                 kernel_size=settings.kernel_size,
                 stride=settings.stride,
                 padding=self.padding,
                 bias=settings.add_bias)),
            ("attention", attention)])
        conv_module = nn.Sequential(dict_modules)
        return conv_module

    def _build_conv_module(self, settings, padding):
        if settings.attention_type == "none":
            attention = Identity()
        elif settings.attention_type == "glu":
            attention = nn.GLU(dim=1)
        elif settings.attention_type == "eca":
            if "_" in settings.attention_type:
                kernel_size = int(settings.attention_type.split("_")[-1])
            else:
                kernel_size = 3
            attention = EfficientChannelAttention(
                spatial_channels=1, kernel_size=kernel_size)

        if settings.conv_type == "separable":
            conv_module = self._build_separable_conv(settings, attention, padding)
        elif settings.conv_type == "predepth":
            conv_module = self._build_predepth_conv(settings, attention, padding)
        elif settings.conv_type == "standard":
            conv_module = self._build_standard_conv(settings, attention, padding)
        return conv_module

    def forward(self,
                feature):
        shape = feature.shape
        if len(shape) == 4:
            # `[N, C, T, *] -> [N, C, *, T] -> [N, C', T]`
            feature = feature.permute([0, 1, 3, 2])
            feature = feature.reshape([shape[0], -1, shape[2]])
        elif len(shape) == 5:
            # `[N, C, T, *, *] -> [N, C, *, *, T] -> [N, C', T]`
            feature = feature.permute([0, 1, 3, 4, 2])
            feature = feature.reshape([shape[0], -1, shape[2]])

        res = self.residual(feature)
        feature = self.conv_module(feature)
        feature = apply_norm(self.norm, feature, channel_first=True)
        feature = self.activation(feature)
        feature = self.dropout(feature)
        feature = feature + res
        return feature


class CNN2DFeatureExtractorSettings(ConfiguredModel):
    fext_type: str = "cnn2d"
    in_channels: int = 64
    out_channels: int = 64
    # Conv2D settings.
    kernel_size: int | Tuple[int, int] = 3
    stride: int | Tuple[int, int] = 1
    padding: int | Tuple[int, int] = 0
    padding_mode: str = "zeros"
    dilation: int | Tuple[int, int] = 1
    groups: int = 1
    causal: bool | None = False
    add_residual: bool = True

    norm_type: str = Field(default="batch", pattern=r"batch|batch2d|layer")
    norm_eps: float = 1e-5
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    add_bias: bool = True
    dropout: float = 0.1

    def model_post_init(self, __context):
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)

        if self.causal is not None:
            if self.causal is True:
                padding_t = self.kernel_size[0] - 1
                padding_j = (self.kernel_size[1] - 1) // 2
            else:
                padding_t = (self.kernel_size[0] - 1) // 2
                padding_j = (self.kernel_size[1] - 1) // 2
            self.padding = (padding_t, padding_j)

    def build_layer(self):
        return CNN2DFeatureExtractor(self)


class CNN2DFeatureExtractor(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, CNN2DFeatureExtractorSettings)
        self.settings = settings

        self.conv = nn.Conv2d(
            in_channels=settings.in_channels,
            out_channels=settings.out_channels,
            kernel_size=settings.kernel_size,
            stride=settings.stride,
            padding=settings.padding,
            dilation=settings.dilation,
            groups=settings.groups,
            bias=settings.add_bias,
            padding_mode=settings.padding_mode)

        self.norm = create_norm(
            settings.norm_type, settings.out_channels, settings.norm_eps,
            settings.add_bias)

        self.activation = select_reluwise_activation(settings.activation)

        self.dropout = nn.Dropout(p=settings.dropout)

        if settings.add_residual:
            if settings.in_channels == settings.out_channels:
                if settings.stride[0] == 1:
                    self.residual = Identity()
                else:
                    self.residual = nn.MaxPool2d(
                        kernel_size=settings.kernel_size,
                        stride=settings.stride,
                        padding=settings.padding)
            else:
                self.residual = nn.Conv2d(
                    in_channels=settings.in_channels,
                    out_channels=settings.out_channels,
                    kernel_size=(1, 1),
                    stride=settings.stride)
        else:
            self.residual = Zero()

    def forward(self,
                feature):
        res = self.residual(feature)
        feature = self.conv(feature)

        if self.settings.causal is True:
            feature = feature[:, :, :-self.settings.padding[0]]

        feature = apply_norm(self.norm, feature, channel_first=True)

        feature = self.activation(feature)
        feature = self.dropout(feature)
        feature = feature + res
        return feature


class STSeparableCNN2DFeatureExtractorSettings(ConfiguredModel):
    fext_type: str = "stcnn2d"
    in_channels: int = 64
    inter_channels: int = 64
    out_channels: int = 64

    spatial_conv_settings: CNN2DFeatureExtractorSettings = Field(
        default_factory=lambda: CNN2DFeatureExtractorSettings())

    temporal_conv_settings: CNN2DFeatureExtractorSettings = Field(
        default_factory=lambda: CNN2DFeatureExtractorSettings())

    def _init_spatial(self):
        sc = self.spatial_conv_settings
        sc.in_channels = self.in_channels
        sc.out_channels = self.inter_channels
        if isinstance(sc.kernel_size, int):
            sc.kernel_size = (1, sc.kernel_size)
        else:
            # Force to spatial only.
            sc.kernel_size = (1, sc.kernel_size[-1])
        if isinstance(sc.stride, int):
            sc.stride = (1, sc.stride)
        else:
            sc.stride = (1, sc.stride[-1])
        if isinstance(sc.dilation, int):
            sc.dilation = (1, sc.dilation[-1])
        else:
            sc.dilation = (1, sc.dilation[-1])
        # Force uncausal.
        sc.causal = False
        sc.padding = (0, (sc.kernel_size[1] - 1) // 2)

    def _init_temporal(self):
        tc = self.temporal_conv_settings
        tc.in_channels = self.inter_channels
        tc.out_channels = self.out_channels
        if isinstance(tc.kernel_size, int):
            tc.kernel_size = (tc.kernel_size, 1)
        else:
            # Force to temporal only.
            tc.kernel_size = (tc.kernel_size[0], 1)
        if isinstance(tc.stride, int):
            tc.stride = (tc.stride, 1)
        else:
            tc.stride = (tc.stride[0], 1)
        if isinstance(tc.dilation, int):
            tc.dilation = (tc.dilation, 1)
        else:
            tc.dilation = (tc.dilation[0], 1)
        if tc.causal is not None:
            if tc.causal is True:
                padding = tc.kernel_size[0] - 1
            else:
                padding = (tc.kernel_size[0] - 1) // 2
            tc.padding = (padding, 0)

    def model_post_init(self, __context):
        self._init_spatial()
        self._init_temporal()

        self.spatial_conv_settings.model_post_init(__context)
        self.temporal_conv_settings.model_post_init(__context)

    def build_layer(self):
        return STSeparableCNN2DFeatureExtractor(self)


class STSeparableCNN2DFeatureExtractor(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, STSeparableCNN2DFeatureExtractorSettings)
        self.settings = settings

        # Spatial conv.
        self.sconv = settings.spatial_conv_settings.build_layer()

        # Temporal conv.
        self.tconv = settings.temporal_conv_settings.build_layer()

    def forward(self,
                feature):
        feature = self.sconv(feature)
        feature = self.tconv(feature)
        return feature


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
