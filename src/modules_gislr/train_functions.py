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
import torch.nn.functional as F
from torch import nn

from nltk.metrics.distance import edit_distance

# Local modules
from .layers import (
    RNNCSLR,
    TransformerCSLR)

from .utils import (
    make_causal_mask)

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


def forward(model, feature, tokens, feature_pad_mask, tokens_pad_mask):
    if isinstance(model, RNNCSLR):
        preds = model(feature,
                      tokens,
                      feature_pad_mask=feature_pad_mask,
                      tokens_pad_mask=tokens_pad_mask)
    elif isinstance(model, TransformerCSLR):
        tokens_causal_mask = make_causal_mask(tokens_pad_mask)
        preds = model(src_feature=feature,
                      tgt_feature=tokens,
                      src_causal_mask=None,
                      src_padding_mask=feature_pad_mask,
                      tgt_causal_mask=tokens_causal_mask,
                      tgt_padding_mask=tokens_pad_mask)
    else:
        raise NotImplementedError(f"Unknown model type:{type(model)}.")
    return preds


def check_tokens_format(tokens, tokens_pad_mask, start_id, end_id):
    # Check token's format.
    end_indices0 = np.arange(len(tokens))
    end_indices1 = tokens_pad_mask.sum(dim=-1).detach().cpu().numpy() - 1
    message = "The start and/or end ids are not included in tokens. " \
        f"Please check data format. start_id:{start_id}, " \
        f"end_id:{end_id}, enc_indices:{end_indices1}, tokens:{tokens}"
    ref_tokens = tokens.detach().cpu().numpy()
    assert (ref_tokens[:, 0] == start_id).all(), message
    assert (ref_tokens[end_indices0, end_indices1] == end_id).all(), message


def train_loop_csir_s2s(dataloader,
                        model,
                        loss_fn,
                        optimizer,
                        device,
                        start_id,
                        end_id,
                        return_pred_times=False):
    num_batches = len(dataloader)
    train_loss = 0
    size = len(dataloader.dataset)

    # Collect prediction time.
    pred_times = []

    # Switch to training mode.
    model.train()
    # Main loop.
    print("Start training.")
    start = time.perf_counter()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        feature_pad_mask = batch_sample["feature_pad_mask"]
        tokens = batch_sample["token"]
        tokens_pad_mask = batch_sample["token_pad_mask"]

        check_tokens_format(tokens, tokens_pad_mask, start_id, end_id)

        feature = feature.to(device)
        feature_pad_mask = feature_pad_mask.to(device)
        tokens = tokens.to(device)
        tokens_pad_mask = tokens_pad_mask.to(device)

        frames = feature.shape[-2]

        # Predict.
        pred_start = time.perf_counter()
        preds = forward(model, feature, tokens,
                        feature_pad_mask, tokens_pad_mask)
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # Compute loss.
        # Preds do not include <start>, so skip that of tokens.
        loss = 0
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            for t_index in range(1, tokens.shape[-1]):
                pred = preds[:, t_index-1, :]
                token = tokens[:, t_index]
                loss += loss_fn(pred, token)
            loss /= tokens.shape[-1]
        # LabelSmoothingCrossEntropyLoss
        else:
            # `[N, T, C] -> [N, C, T]`
            preds = preds.permute([0, 2, 1])
            # Remove prediction after the last token.
            if preds.shape[-1] == tokens.shape[-1]:
                preds = preds[:, :, :-1]
            loss = loss_fn(preds, tokens[:, 1:])

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


def val_loop_csir_s2s(dataloader,
                      model,
                      loss_fn,
                      device,
                      start_id,
                      end_id,
                      return_pred_times=False):
    num_batches = len(dataloader)
    val_loss = 0

    # Collect prediction time.
    pred_times = []

    # Switch to training mode.
    model.train()
    # Main loop.
    print("Start validation.")
    start = time.perf_counter()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        feature_pad_mask = batch_sample["feature_pad_mask"]
        tokens = batch_sample["token"]
        tokens_pad_mask = batch_sample["token_pad_mask"]

        check_tokens_format(tokens, tokens_pad_mask, start_id, end_id)

        feature = feature.to(device)
        feature_pad_mask = feature_pad_mask.to(device)
        tokens = tokens.to(device)
        tokens_pad_mask = tokens_pad_mask.to(device)

        frames = feature.shape[-2]

        # Predict.
        pred_start = time.perf_counter()
        preds = forward(model, feature, tokens,
                        feature_pad_mask, tokens_pad_mask)
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # Compute loss.
        # Preds do not include <start>, so skip that of tokens.
        loss = 0
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            for t_index in range(1, tokens.shape[-1]):
                pred = preds[:, t_index-1, :]
                token = tokens[:, t_index]
                loss += loss_fn(pred, token)
            loss /= tokens.shape[-1]
        # LabelSmoothingCrossEntropyLoss
        else:
            # `[N, T, C] -> [N, C, T]`
            preds = preds.permute([0, 2, 1])
            # Remove prediction after the last token.
            if preds.shape[-1] == tokens.shape[-1]:
                preds = preds[:, :, :-1]
            loss = loss_fn(preds, tokens[:, 1:])

        val_loss += loss.item()
    print(f"Done. Time:{time.perf_counter()-start}")

    # Average loss.
    val_loss /= num_batches
    print("Validation performance: \n",
          f"Avg loss:{val_loss:>8f}\n")
    pred_times = np.array(pred_times)
    retval = (val_loss, pred_times) if return_pred_times else val_loss
    return retval


def inference(model, feature, start_id, end_id, max_seqlen):
    if isinstance(model, RNNCSLR):
        pred_ids, _ = model.inference(feature,
                                      start_id,
                                      end_id,
                                      max_seqlen=max_seqlen)
    elif isinstance(model, TransformerCSLR):
        pred_ids, _ = model.inference(feature,
                                      start_id,
                                      end_id,
                                      max_seqlen=max_seqlen)
    else:
        raise NotImplementedError(f"Unknown model type:{type(model)}.")
    return pred_ids


def test_loop_csir_s2s(dataloader,
                       model,
                       device,
                       start_id,
                       end_id,
                       max_seqlen=62,
                       use_normalized_wer=False,
                       return_pred_times=False):
    size = len(dataloader.dataset)
    total_wer = 0

    # Collect prediction time.
    pred_times = []

    # Switch to training mode.
    model.train()
    # Main loop.
    print("Start test.")
    start = time.perf_counter()
    for batch_idx, batch_sample in enumerate(dataloader):
        feature = batch_sample["feature"]
        tokens = batch_sample["token"]
        tokens_pad_mask = batch_sample["token_pad_mask"]

        check_tokens_format(tokens, tokens_pad_mask, start_id, end_id)

        feature = feature.to(device)
        tokens = tokens.to(device)
        tokens_pad_mask = tokens_pad_mask.to(device)

        frames = feature.shape[-2]

        # Predict.
        pred_start = time.perf_counter()
        pred_ids = inference(model, feature, start_id, end_id, max_seqlen)
        pred_end = time.perf_counter()
        pred_times.append([frames, pred_end - pred_start])

        # Compute WER.
        # <sos> and <eos> should be removed because they may boost performance.
        # print(tokens)
        # print(pred_ids)
        tokens = tokens[0, 1:-1]
        # pred_ids = pred_ids[0, 1:-1]
        pred_ids = [pid for pid in pred_ids[0] if pid not in [start_id, end_id]]
        if use_normalized_wer:
            ref_length = max(len(pred_ids), len(tokens))
        else:
            ref_length = len(tokens)
        wer = edit_distance(tokens, pred_ids)
        wer /= ref_length
        total_wer += wer
    print(f"Done. Time:{time.perf_counter()-start}")

    # Average WER.
    awer = total_wer / size * 100
    print("Test performance: \n",
          f"Avg WER:{awer:>0.1f}%\n")
    pred_times = np.array(pred_times)
    retval = (awer, pred_times) if return_pred_times else awer
    return retval


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    For the detail, please refer
    "Rethinking the Inception Architecture for Computer Vision"
    https://arxiv.org/abs/1512.00567
    """
    def __init__(self, weight=None, ignore_indices=None, reduction="none",
                 label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        if isinstance(ignore_indices, int):
            self.ignore_indices = [ignore_indices]
        else:
            self.ignore_indices = ignore_indices
        assert reduction in ["none",
                             "mean_batch_prior", "mean_temporal_prior",
                             "sum"]
        self.reduction = reduction
        assert label_smoothing >= 0.0
        assert label_smoothing <= 1.0
        self.label_smoothing = label_smoothing

    def _isnotin_ignore(self, target):
        # Please refer
        # https://github.com/pytorch/pytorch/issues/3025
        # pylint error of torch.tensor() should be solved in the future release.
        # https://github.com/pytorch/pytorch/issues/24807
        ignore = torch.tensor(self.ignore_indices, dtype=target.dtype,
                              device=target.device)
        isin = (target[..., None] == ignore).any(-1)
        return isin.bitwise_not()

    def _calc_loss(self, logit_t, target_t):
        logit_mask = torch.ones(logit_t.shape[-1],
                                dtype=logit_t.dtype,
                                device=logit_t.device)
        target_mask = torch.ones(target_t.shape,
                                 dtype=logit_t.dtype,
                                 device=logit_t.device)
        if self.ignore_indices is not None:
            logit_mask[self.ignore_indices] = 0
            target_mask = self._isnotin_ignore(target_t).float()
        if self.weight is None:
            weight = torch.ones(logit_t.shape[-1],
                                dtype=logit_t.dtype,
                                device=logit_t.device)
        else:
            weight = self.weight.to(dtype=logit_t.dtype, device=logit_t.device)
        # Calculate CE.
        logprobs = F.log_softmax(logit_t, dim=-1)
        logprobs_m = logprobs * weight * logit_mask
        nll_loss = -logprobs_m.gather(dim=-1, index=target_t.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs_m.sum(dim=-1) / logit_mask.sum()
        smooth_loss *= target_mask
        loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        return loss

    def forward(self, logit, target):
        """Perform forward computation.

        # Args:
          - logit: `[N, C]` or `[N, C, T]`
          - target: `[N]` or [N, T]
        """
        # Check format.
        if len(logit.shape) == 2:
            logit = logit.unsqueeze(-1)
        if len(target.shape) == 1:
            target = target.unsqueeze(-1)
        assert len(logit.shape) == 3, f"{logit.shape}"
        assert len(target.shape) == 2, f"{target.shape}"
        assert logit.shape[0] == target.shape[0], f"{logit.shape, target.shape}"
        assert logit.shape[-1] == target.shape[-1], f"{logit.shape, target.shape}"

        loss = 0
        for t in range(target.shape[-1]):
            _loss = self._calc_loss(logit[:, :, t], target[:, t])
            # Reduction should be conducted in a loop when reduction is
            # mean_batch_prior.
            if self.reduction == "mean_batch_prior":
                if self.ignore_indices is not None:
                    denom = len([t for t in target[:, t]
                                 if t not in self.ignore_indices])
                else:
                    denom = logit.shape[0]
                _loss /= max(denom, 1)
            loss += _loss

        # Reduction.
        if self.reduction == "sum":
            loss = loss.sum()
        # Temporal Normalization.
        if self.reduction == "mean_batch_prior":
            loss = loss.sum() / target.shape[-1]
        if self.reduction == "mean_temporal_prior":
            target_lengths = self._isnotin_ignore(target).sum(dim=-1)
            loss /= torch.clamp(target_lengths, min=1)
            loss = loss.mean()
        return loss


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
