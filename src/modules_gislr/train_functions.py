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
import numpy as np

import torch

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


def train_loop(dataloader, model, loss_fn, optimizer, device, use_mask=True,
               return_pred_times=False):
    num_batches = len(dataloader)
    train_loss = 0
    size = len(dataloader.dataset)

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Collect prediction time.
    pred_times = []

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
        frames = feature.shape[-2]

        # Predict.
        pred_start = time.perf_counter()
        if use_mask:
            feature_pad_mask = batch_sample["feature_pad_mask"]
            feature_pad_mask = feature_pad_mask.to(device)
            pred = model(feature, feature_pad_mask=feature_pad_mask)
        else:
            pred = model(feature)
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

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
    pred_times = np.array(pred_times)
    retval = (train_loss, pred_times) if return_pred_times else train_loss
    return retval


def val_loop(dataloader, model, loss_fn, device, use_mask=True,
             return_pred_times=False):
    num_batches = len(dataloader)
    val_loss = 0

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Collect prediction time.
    pred_times = []

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
            frames = feature.shape[-2]

            # Predict.
            pred_start = time.perf_counter()
            if use_mask:
                feature_pad_mask = batch_sample["feature_pad_mask"]
                feature_pad_mask = feature_pad_mask.to(device)
                pred = model(feature, feature_pad_mask=feature_pad_mask)
            else:
                pred = model(feature)
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])

            val_loss += loss_fn(pred, token.squeeze(-1)).item()
    print(f"Done. Time:{time.perf_counter()-start}")

    # Average loss.
    val_loss /= num_batches
    print("Validation performance: \n",
          f"Avg loss:{val_loss:>8f}\n")
    pred_times = np.array(pred_times)
    retval = (val_loss, pred_times) if return_pred_times else val_loss
    return retval


def test_loop(dataloader, model, device, use_mask=False,
              return_pred_times=False):
    size = len(dataloader.dataset)
    correct = 0

    # Inspect model signature.
    sig = signature(model.forward)
    use_mask = True if "feature_pad_mask" in sig.parameters and use_mask is True else False

    # Collect prediction time.
    pred_times = []

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
            frames = feature.shape[-2]

            # Predict.
            pred_start = time.perf_counter()
            if use_mask:
                feature_pad_mask = batch_sample["feature_pad_mask"]
                feature_pad_mask = feature_pad_mask.to(device)
                pred = model(feature, feature_pad_mask=feature_pad_mask)
            else:
                pred = model(feature)
            pred_end = time.perf_counter()
            pred_times.append([frames, pred_end - pred_start])

            pred_ids = pred.argmax(dim=1).unsqueeze(-1)
            count = (pred_ids == token).sum().detach().cpu().numpy()
            correct += int(count)
    print(f"Done. Time:{time.perf_counter()-start}")

    acc = correct / size * 100
    print("Test performance: \n",
          f"Accuracy:{acc:>0.1f}%")
    pred_times = np.array(pred_times)
    retval = (acc, pred_times) if return_pred_times else acc
    return retval


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
