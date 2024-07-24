#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""activation: Implement extra activation functions.
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

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class FuncTanhExp(torch.autograd.Function):  # pylint: disable=W0223
    """Implementation of TanhExp activation.
    """
    # It is difficult to handle arguments-differ correctly.
    # https://github.com/PyCQA/pylint/issues/3812
    # pylint: disable=arguments-differ
    @staticmethod
    def forward(ctx, inputs):
        """Perform foward computation."""
        inputs = torch.clamp(inputs, max=88)
        ctx.save_for_backward(inputs)
        output = inputs * torch.tanh(torch.exp(inputs))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Perform backward computation."""
        inputs, = ctx.saved_tensors
        results = grad_output * (torch.tanh(torch.exp(inputs))
                                 + inputs * (1 - torch.tanh(torch.exp(inputs))**2)
                                 * torch.exp(inputs))
        return results


class TanhExp(nn.Module):
    """Applies TanhExp function.

    # Note
      For the details, please see https://arxiv.org/abs/2003.09855 .
    """
    def __init__(self):
        # Keep for the consistent style.
        # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Perform foward computation."""
        return FuncTanhExp.apply(inputs)


class GELUAcc(nn.Module):
    """Accurate approximation of GELU function.

    https://github.com/hendrycks/GELUs

    """
    FACTOR = math.sqrt(2 / math.pi)

    def __init__(self):
        # Keep for the consistent style.
        # pylint: disable=useless-super-delegation
        super().__init__()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Perform foward computation."""
        return 0.5 * feature * (1 + torch.tanh(self.FACTOR * (
            feature + 0.044715 * torch.pow(feature, 3))))


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
