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

import math

# Third party's modules
import torch
from torch import nn
from torch.nn import functional as F

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"


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


class TransformerEnISLR(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 num_head=1,
                 num_layers=1,
                 dim_feedforward=256,
                 batch_first=True,
                 norm_first=True,
                 dropoutrate=0.1):
        super().__init__()

        self._build_feature_extractor(in_channels, inter_channels)

        self._build_tr_encoder(inter_channels, num_head, num_layers,
                               dim_feedforward, dropoutrate,
                               batch_first, norm_first)

        self._build_recognition_head(inter_channels, out_channels)

    def _build_feature_extractor(self, in_channels, inter_channels):
        self.linear = nn.Linear(in_channels, inter_channels)
        self.activation = nn.ReLU()

    def _build_tr_encoder(self,
                          inter_channels,
                          nhead,
                          num_layers,
                          dim_feedforward,
                          dropoutrate,
                          batch_first,
                          norm_first):
        self.pe = PositionalEncoding(dim_model=inter_channels,
                                     dropout=dropoutrate)
        enlayer = nn.TransformerEncoderLayer(
            d_model=inter_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropoutrate,
            batch_first=batch_first,
            norm_first=norm_first)
        self.tr_encoder = nn.TransformerEncoder(
            encoder_layer=enlayer,
            num_layers=num_layers)

    def _build_recognition_head(self,
                                inter_channels,
                                out_channels):
        self.head = GPoolRecognitionHead(inter_channels, out_channels)

    def _invert_mask(self, mask):
        _mask = torch.logical_not(mask) if mask is not None else None
        return _mask

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

        # Apply PE.
        feature = self.pe(feature)
        if torch.isnan(feature).any():
            raise ValueError()

        # Note:
        # Different from the other layers, the attention of pytorch's
        # Transformer attends to `False` position indicated by the input masks.
        # Therefor, we invert the flags of the input masks here.
        _feature_causal_mask = self._invert_mask(feature_causal_mask)
        _feature_pad_mask = self._invert_mask(feature_pad_mask)

        feature = self.tr_encoder(src=feature,
            mask=_feature_causal_mask,
            src_key_padding_mask=_feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()

        # `[N, T, C] -> [N, C, T]`
        logit = self.head(feature.permute([0, 2, 1]))
        if torch.isnan(feature).any():
            raise ValueError()
        return logit


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
