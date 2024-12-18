#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""perturbation: Train perturbation.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

# Third party's modules
import torch

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., target=None):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if target is not None:
                    if name == target:
                        self.backup[name] = param.data.clone()
                        norm = torch.norm(param.grad)
                        if norm != 0:
                            r_at = epsilon * param.grad / norm
                            param.data.add_(r_at)
                        break
                else:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0:
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)
                    break

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
            self.backup = {}


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
