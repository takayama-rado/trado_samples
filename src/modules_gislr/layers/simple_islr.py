#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""simple_islr: Simple ISLR model.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


# Third party's modules
from pydantic import (
    Field)

from torch import nn

# Local modules
from .misc import (
    ConfiguredModel,
    GPoolRecognitionHeadSettings)

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class SimpleISLRSettings(ConfiguredModel):
    in_channels: int
    inter_channels: int
    out_channels: int
    head_settings: GPoolRecognitionHeadSettings = Field(
        default_factory=lambda: GPoolRecognitionHeadSettings())

    def model_post_init(self, __context):
        self.head_settings.in_channels = self.inter_channels
        self.head_settings.out_channels = self.out_channels

        # Propagate.
        self.head_settings.model_post_init(__context)

    def build_layer(self):
        return SimpleISLR(self)


class SimpleISLR(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, SimpleISLRSettings)
        self.settings = settings

        self.linear = nn.Linear(settings.in_channels, settings.inter_channels)
        self.activation = nn.ReLU()

        self.head = settings.head_settings.build_layer()

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


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
