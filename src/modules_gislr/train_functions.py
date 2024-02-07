#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""train_functions: Define functions for training.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
from inspect import signature

# Third party's modules
import torch

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


def train_loop(dataloader, model, loss_fn, optimizer, device, use_mask=True):
    num_batches = len(dataloader)
    train_loss = 0
    size = len(dataloader.dataset)

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Switch to training mode.
    model.train()
    # Main loop.
    print("Start training.")
    start = time.perf_counter()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        token = batch_sample["token"]
        feature = feature.to(device)
        token = token.to(device)

        # Predict.
        if use_mask:
            feature_pad_mask = batch_sample["feature_pad_mask"]
            feature_pad_mask = feature_pad_mask.to(device)
            pred = model(feature, feature_pad_mask=feature_pad_mask)
        else:
            pred = model(feature)
        # Compute loss.
        loss = loss_fn(pred, token.squeeze(-1))

        # Back propagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Print current loss per 100 steps.
        if batch_idx % 100 == 0:
            loss = loss.item()
            steps = batch_idx * len(feature)
            print(f"loss:{loss:>7f} [{steps:>5d}/{size:>5d}]")
    print(f"Done. Time:{time.perf_counter()-start}")
    # Average loss.
    train_loss /= num_batches
    print("Training performance: \n",
          f"Avg loss:{train_loss:>8f}\n")
    return train_loss


def val_loop(dataloader, model, loss_fn, device, use_mask=True):
    num_batches = len(dataloader)
    val_loss = 0

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start validation.")
    start = time.perf_counter()
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"]
            token = batch_sample["token"]
            feature = feature.to(device)
            token = token.to(device)

            # Predict.
            if use_mask:
                feature_pad_mask = batch_sample["feature_pad_mask"]
                feature_pad_mask = feature_pad_mask.to(device)
                pred = model(feature, feature_pad_mask=feature_pad_mask)
            else:
                pred = model(feature)
            val_loss += loss_fn(pred, token.squeeze(-1)).item()
    print(f"Done. Time:{time.perf_counter()-start}")

    # Average loss.
    val_loss /= num_batches
    print("Validation performance: \n",
          f"Avg loss:{val_loss:>8f}\n")
    return val_loss


def test_loop(dataloader, model, device, use_mask=False):
    size = len(dataloader.dataset)
    correct = 0

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Switch to evaluation mode.
    model.eval()
    # Main loop.
    print("Start evaluation.")
    start = time.perf_counter()
    with torch.no_grad():
        for batch_sample in dataloader:
            feature = batch_sample["feature"]
            token = batch_sample["token"]
            feature = feature.to(device)
            token = token.to(device)

            # Predict.
            if use_mask:
                feature_pad_mask = batch_sample["feature_pad_mask"]
                feature_pad_mask = feature_pad_mask.to(device)
                pred = model(feature, feature_pad_mask=feature_pad_mask)
            else:
                pred = model(feature)

            pred_ids = pred.argmax(dim=1).unsqueeze(-1)
            count = (pred_ids == token).sum().detach().cpu().numpy()
            correct += int(count)
    print(f"Done. Time:{time.perf_counter()-start}")

    acc = correct / size * 100
    print("Test performance: \n",
          f"Accuracy:{acc:>0.1f}%")
    return acc


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
