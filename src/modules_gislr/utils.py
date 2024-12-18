#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""utils: Utility functions.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random

# Third party's modules
import numpy as np

import torch

# Local modules
from .activation import (
    GELUAcc,
    TanhExp)

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


def make_san_mask(pad_mask,
                  causal_mask):
    """Make self-attention mask.

    # Args:
      - pad_mask: The padding mask. `[N, T]`
      - causal_mask: `[N, T (query), T (key)]`
        The mask for future context. For example, if T = 3, causal_mask
        should be:
        [[1, 0, 0,
         [1, 1, 0,
         [1, 1, 1]]
    # Returns:
      - san_mask: `[N, T (query), T (key)]`
    """
    # xx_mask: `[N, qlen, klen]`
    san_mask = pad_mask.unsqueeze(1).repeat([1, pad_mask.shape[-1], 1])
    if causal_mask is not None:
        san_mask = san_mask & causal_mask
    return san_mask


def make_causal_mask(ref_mask,
                     lookahead=0):
    """Make causal mask.
    # Args:
      - ref_mask: `[N, T (query)]`
        The reference mask to make causal mask.
      - lookahead: lookahead frame
    # Returns:
      - ref_mask: `[T (query), T (key)]`
    """
    causal_mask = ref_mask.new_ones([ref_mask.size(1), ref_mask.size(1)],
                                    dtype=ref_mask.dtype)
    causal_mask = torch.tril(causal_mask,
                             diagonal=lookahead,
                             out=causal_mask).unsqueeze(0)
    return causal_mask


def select_reluwise_activation(activation):
    if activation == "relu":
        layer = torch.nn.ReLU()
    elif activation == "gelu":
        layer = torch.nn.GELU()
    elif activation in ["swish", "silu"]:
        layer = torch.nn.SiLU()
    elif activation == "mish":
        layer = torch.nn.Mish()
    elif activation == "geluacc":
        layer = GELUAcc()
    elif activation == "tanhexp":
        layer = TanhExp()
    else:
        raise NotImplementedError(f"Activation for {activation} is not implemented.")
    return layer


def save_checkpoint(ckpt_path, model,
                    epoch=0,
                    optimizer=None,
                    scheduler=None):
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {}
    checkpoint = {
        "model": model_to_save.state_dict(),
        "epoch": epoch,
        "random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        "cuda_random": torch.cuda.get_rng_state(),
        "cuda_random_all": torch.cuda.get_rng_state_all(),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint, ckpt_path)


def load_checkpoint(ckpt_path, model,
                    optimizer=None,
                    scheduler=None):
    checkpoint = torch.load(ckpt_path, weights_only=False)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    epoch = checkpoint["epoch"]
    random.setstate(checkpoint["random"])
    np.random.set_state(checkpoint["np_random"])
    torch.set_rng_state(checkpoint["torch"])
    torch.random.set_rng_state(checkpoint["torch_random"])
    try:
        torch.cuda.set_rng_state(checkpoint["cuda_random"])
        torch.cuda.torch.cuda.set_rng_state_all(checkpoint["cuda_random_all"])
    except:
        pass

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return model, epoch, optimizer, scheduler


def plot_pred_times(outpath, train_times, val_times, test_times):
    _train_times = train_times[:, 1]
    _val_times = val_times[:, 1]
    _test_times = test_times[:, 1]

    plt.grid(axis="y", linestyle="dotted", color="k")

    points = [_train_times, _val_times, _test_times]
    plt.boxplot(points, labels=["Train", "Validation", "Test"])
    plt.ylabel("Prediction time[sec]")
    plt.xlabel("Process type")
    plt.savefig(outpath, bbox_inches="tight")


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
