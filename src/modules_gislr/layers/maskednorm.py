#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""maskednorm: Masked normalization layer.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import copy
import inspect
from functools import partial
from typing import (
    Any,
    Tuple)

# Third party's modules
import torch
from torch import nn
from torch.nn import functional as F

# Local modules
# from .misc import Identity

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


MaskedTypes = ["masked_switch",
               "masked_instance",
               "masked_layer",
               "masked_batch",
               "masked_positional"] # ,
               # "masked_power"]
StandardTypes = ["batch",
                 "layer",
                 "layer_g",
                 "group",
                 "instance"]
AvailableTypes = MaskedTypes + StandardTypes


def _pre_adjust_channel_axis(feature, channel_last):
    # Channel dimension should be second.
    if channel_last is True:
        if len(feature.shape) == 3:
            # B, T, C => B, C, T
            feature = feature.permute(0, 2, 1)
        elif len(feature.shape) == 4:
            # B, H, W, C => B, C, H, W
            feature = feature.premute(0, 3, 1, 2)
        elif len(feature.shape) == 5:
            # B, H, W, D, C => B, C, H, W, D
            feature = feature.permute(0, 4, 1, 2, 3)
    return feature


def _select_and_apply_norm(norm_layer, feature, mask):
    # Apply normalization.
    if isinstance(norm_layer, _MaskedSwitchNormNd):
        if mask is not None and len(mask.size()) < len(feature.size()):
            _mask = insert_axes_to_mask(mask, target_size=feature.size())
        else:
            _mask = mask
        feature = norm_layer(feature, _mask)
    elif isinstance(norm_layer, nn.LayerNorm):
        if len(feature.shape) == 3:
            # B, C, T => B, T, C => B, C, T
            feature = feature.permute(0, 2, 1)
            feature = norm_layer(feature)
            feature = feature.permute(0, 2, 1)
        elif len(feature.shape) == 4:
            # B, C, H, W => B, H, W, C => B, C, H, W
            feature = feature.permute(0, 2, 3, 1)
            feature = norm_layer(feature)
            feature = feature.permute(0, 3, 1, 2)
        elif len(feature.shape) == 5:
            # B, C, H, W, D => B, H, W, D, C => B, C, H, W, D
            feature = feature.permute(0, 2, 3, 4, 1)
            feature = norm_layer(feature)
            feature = feature.permute(0, 4, 1, 2, 3)
    else:
        feature = norm_layer(feature)
    return feature


def _post_adjust_channel_axis(feature, channel_last):
    # Back to original shape.
    if channel_last is True:
        if len(feature.shape) == 3:
            # B, C, T => B, T, C
            feature = feature.permute(0, 2, 1)
        if len(feature.shape) == 4:
            # B, C, H, W => B, H, W, C
            feature = feature.permute(0, 2, 3, 1)
        if len(feature.shape) == 5:
            # B, C, H, W, D => B, H, W, D, C
            feature = feature.permute(0, 2, 3, 4, 1)
    return feature


def apply_normalization(norm_layer, feature, mask=None, channel_last=False):
    """Apply normalization to input tensor.
    """
    # Channel dimension should be second.
    feature = _pre_adjust_channel_axis(feature, channel_last)
    # Apply normalization.
    feature = _select_and_apply_norm(norm_layer, feature, mask)
    # Back to original shape.
    feature = _post_adjust_channel_axis(feature, channel_last)
    return feature


def create_norm_layer(classobj, channel_dim):
    """Create normalization layer.
    """
    # if classobj is Identity:
    #     return classobj()
    spec = inspect.getfullargspec(classobj)
    if "num_features" in spec.args + spec.kwonlyargs:
        retval = classobj(num_features=channel_dim)
    elif "num_channels" in spec.args + spec.kwonlyargs:
        retval = classobj(num_channels=channel_dim)
    elif "normalized_shape" in spec.args + spec.kwonlyargs:
        retval = classobj(normalized_shape=channel_dim)
    else:
        _args = inspect.getfullargspec(classobj)
        raise ValueError(f"Unexpected interfaces. args:{_args}")
    return retval


def select_norm_type(typename: Tuple[str, ...],
                     raise_error=True) -> Tuple[Any, ...]:
    """Select normalization class according to the given type name.
    """
    assert isinstance(typename, (tuple, list)), \
        f"{type(typename)}:{typename}"
    check_masked = True
    for tname in typename:
        if tname not in MaskedTypes:
            check_masked = False
            break
    if check_masked is True:
        return (partial(MaskedSwitchNorm1d,
                        modes=typename,
                        filter_maskedval=True),
                partial(MaskedSwitchNorm2d,
                        modes=typename,
                        filter_maskedval=True))

    # Accept only single normalization layer if type is not masked.
    assert len(typename) == 1
    # if typename[0] == "switch":
    #     return (SwitchNorm1d, SwitchNorm2d)
    # if typename[0] == "masked":
    #     return (MaskedBatchNorm1d, MaskedBatchNorm2d)
    retval: Tuple[Any, ...] = ()
    if typename[0] == "batch":
        retval = (nn.BatchNorm1d, nn.BatchNorm2d)
    elif typename[0] == "layer":
        retval = (nn.LayerNorm, nn.LayerNorm)
    elif typename[0] == "layer_g":
        # For consistent interfaces, we substitute with GroupNorm(num_groups=1).
        retval = (partial(nn.GroupNorm, num_groups=1),
                  partial(nn.GroupNorm, num_groups=1))
    elif typename[0] == "group":
        retval = (nn.GroupNorm, nn.GroupNorm)
    elif typename[0] == "instance":
        retval = (nn.InstanceNorm1d, nn.InstanceNorm2d)
    # elif raise_error is False:
    #     retval = (Identity, Identity)
    else:
        raise ValueError(f"Unknown typename {typename}.")
    return retval


def insert_axes_to_mask(mask, target_size):
    """Insert the axis to mask to match target size.
    """
    _mask = mask
    _target = [t for i, t in enumerate(target_size) if i != 1]
    # If mask's temporary size is different from that of target,
    # we down sample mask.
    if _mask.size(1) != _target[1]:
        _mask = _mask.unsqueeze(1)
        dtype = _mask.dtype
        # Interpolate does not accept boolean type.
        _mask = F.interpolate(_mask.float(), size=_target[1], mode="nearest")
        if dtype == torch.bool:
            _mask = _mask.bool()
        _mask = _mask.squeeze()
    if len(mask.size()) == 2:
        if len(target_size) == 4:
            _mask = _mask.unsqueeze(-1)
        if len(target_size) == 5:
            _mask = _mask.unsqueeze(-1)
            _mask = _mask.unsqueeze(-1)
        _mask = _mask.expand(_target)
        _mask = _mask.unsqueeze(1)
    elif len(mask.size()) == 3:
        if len(target_size) == 5:
            _mask = _mask.unsqueeze(-1)
        _mask = _mask.expand(_target)
        _mask = _mask.unsqueeze(1)
    else:
        return _mask
    return _mask


class _MaskedSwitchNormNd(nn.Module):

    __constants__ = ["filter_maskedval", "mode", "last_gamma",
                     "track_running_stats", "momentum", "eps", "num_features"]
    num_features: int
    eps: float
    momentum: float
    track_running_stats: bool
    last_gamma: bool
    mode: list
    filter_maskedval: bool

    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True,
                 last_gamma=False, modes=("masked_switch"), filter_maskedval=False):
        super().__init__()
        for mode in modes:
            assert mode in AvailableTypes
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.last_gamma = last_gamma
        self.filter_maskedval = filter_maskedval

        if modes[0] == "masked_switch":
            self.mode = copy.deepcopy(AvailableTypes)
            self.mode.remove("masked_switch")
        else:
            self.mode = modes

        if len(self.mode) == 1:
            self.mean_weight = None
            self.var_weight = None
        else:
            self.mean_weight = nn.Parameter(torch.ones(len(self.mode)))
            self.var_weight = nn.Parameter(torch.ones(len(self.mode)))

        # pylint doesn't judge register_buffer() as initialization.
        # And "self.num_batches_tracked = None" disturbs register_buffer().
        # https://github.com/pytorch/pytorch/issues/1874
        # So, it is difficult to remove no-member error of num_batches_tracked
        # correctly.

    def _reset_parameters(self):
        if "masked_batch" in self.mode:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            # num_batches_tracked should be initialized in derived classes.
            # pylint: disable=no-member
            self.num_batches_tracked.zero_()
        if self.last_gamma:
            nn.init.zeros_(self.weight)
        else:
            nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def _check_input_dim(self, feature):  # pylint: disable=unused-argument
        message = "_MaskedSwitchNormNd can not be called directory. " \
          "Please use derived classes."
        raise NotImplementedError(message)

    def _init_in_mask(self, size, _mask):
        # Mask for Instance.
        # Calculate uniform distribution over feature dimensions.
        if "masked_instance" in self.mode:
            if len(size) == 3:
                num = torch.clamp(_mask.sum([2]), min=1)
                _mask_in = _mask / num[:, :, None]
            elif len(size) == 4:
                num = torch.clamp(_mask.sum([2, 3]), min=1)
                _mask_in = _mask / num[:, :, None, None]
            elif len(size) == 5:
                num = torch.clamp(_mask.sum([2, 3, 4]), min=1)
                _mask_in = _mask / num[:, :, None, None, None]
            _mask_in = _mask_in.expand(size)
        else:
            _mask_in = None
        return _mask_in

    def _init_ln_mask(self, size, _mask):
        # Mask for Layer.
        if "masked_layer" in self.mode:
            if len(size) == 3:
                num = torch.clamp(_mask.sum([2]) * size[1], min=1)
                _mask_ln = _mask / num[:, :, None]
            elif len(size) == 4:
                num = torch.clamp(_mask.sum([2, 3]) * size[1], min=1)
                _mask_ln = _mask / num[:, :, None, None]
            elif len(size) == 5:
                num = torch.clamp(_mask.sum([2, 3, 4]) * size[1], min=1)
                _mask_ln = _mask / num[:, :, None, None, None]
            _mask_ln = _mask_ln.expand(size)
        else:
            _mask_ln = None
        return _mask_ln

    def _init_pn_mask(self, size, _mask):
        if "masked_positional" in self.mode:
            _mask_pn = _mask / torch.clamp(size[1], min=1)
            _mask_pn = _mask_pn.expand(size)
        else:
            _mask_pn = None
        return _mask_pn

    def _init_bn_mask(self, size, _mask):
        if "masked_batch" in self.mode:
            num = torch.clamp(_mask.sum(), min=1)
            _mask_bn = _mask / num
            _mask_bn = _mask_bn.expand(size)
        else:
            _mask_bn = None
        return _mask_bn

    def _initialize_mask(self, feature, mask=None):
        size = feature.size()
        if mask is None:
            _mask = torch.ones(list(size[0:1]) + [1] + list(size[2:]),
                               dtype=feature.dtype, device=feature.device,
                               requires_grad=False)
        else:
            _mask = mask.to(feature.dtype)
        self._check_input_dim(_mask)

        # Mask for Instance.
        # Calculate uniform distribution over feature dimensions.
        _mask_in = self._init_in_mask(size, _mask)

        # Mask for Layer.
        _mask_ln = self._init_ln_mask(size, _mask)

        # Mask for Postional.
        _mask_pn = self._init_pn_mask(size, _mask)

        # Mask for Batch/Power.
        _mask_bn = self._init_bn_mask(size, _mask)

        _mask_one = _mask.expand(size)
        retval = {"masked_instance": _mask_in,
                  "masked_layer": _mask_ln,
                  "masked_positional": _mask_pn,
                  "masked_batch": _mask_bn,
                  "one": _mask_one}
        return retval

    def _apply_masked_instance(self, feature, _masks, stats, axes_in):
        size = feature.size()
        if "masked_instance" in self.mode:
            _mask_in = _masks["masked_instance"]
            mean_in = (_mask_in * feature).sum(axes_in, keepdim=True)
            var_in = (_mask_in * (feature - mean_in) ** 2).sum(axes_in, keepdim=True)
            var_in = torch.clip(var_in, min=0)
            assert (var_in >= 0).all(), f"{feature}, {_mask_in}, {mean_in}, {var_in}"
            mean_in = mean_in.expand(size)
            var_in = var_in.expand(size)
            stats["masked_instance"]["mean"] = mean_in
            stats["masked_instance"]["var"] = var_in

    def _apply_masked_layer(self, feature, _masks, stats, axes_ln):
        size = feature.size()
        if "masked_layer" in self.mode:
            _mask_ln = _masks["masked_layer"]
            mean_ln = (_mask_ln * feature).sum(axes_ln, keepdim=True)
            var_ln = (_mask_ln * (feature - mean_ln) ** 2).sum(axes_ln, keepdim=True)
            var_ln = torch.clip(var_ln, min=0)
            assert (var_ln >= 0).all(), f"{input}, {_mask_ln}, {mean_ln}, {var_ln}"
            mean_ln = mean_ln.expand(size)
            var_ln = var_ln.expand(size)
            stats["masked_layer"]["mean"] = mean_ln
            stats["masked_layer"]["var"] = var_ln

    def _apply_masked_positional(self, feature, _masks, stats, axes_pn):
        size = feature.size()
        if "masked_positional" in self.mode:
            _mask_pn = _masks["masked_positional"]
            mean_pn = (_mask_pn * feature).sum(axes_pn, keepdim=True)
            var_pn = (_mask_pn * (feature - mean_pn) ** 2).sum(axes_pn, keepdim=True)
            var_pn = torch.clip(var_pn, min=0)
            assert (var_pn >= 0).all(), f"{feature}, {_mask_pn}, {mean_pn}, {var_pn}"
            mean_pn = mean_pn.expand(size)
            var_pn = var_pn.expand(size)
            stats["masked_positional"]["mean"] = mean_pn
            stats["masked_positional"]["var"] = var_pn

    def _apply_masked_batch(self, feature, _masks, stats, axes_bn):
        size = feature.size()
        if "masked_batch" in self.mode:
            # if self.momentum is not None:
            #     exponential_average_factor = self.momentum
            # elif self.training and self.track_running_stats:
            #     # num_batches_tracked should be initialized in derived classes.
            #     if self.num_batches_tracked is not None:  # pylint: disable=no-member
            #         self.num_batches_tracked += 1  # pylint: disable=no-member
            #         # use cumulative moving average
            #         exponential_average_factor = 1.0 / float(self.num_batches_tracked)  # pylint: disable=no-member
            # else:
            #     exponential_average_factor = 0.0
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # num_batches_tracked should be initialized in derived classes.
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        exponential_average_factor = 1.0 / float(self.num_batch_tracked)
                    else:
                        exponential_average_factor = self.momentum

            _mask_bn = _masks["masked_batch"]
            mean_bn = (_mask_bn * feature).sum(axes_bn, keepdim=True)
            var_bn = (_mask_bn * (feature - mean_bn) ** 2).sum(axes_bn, keepdim=True)
            var_bn = torch.clip(var_bn, min=0)
            assert (var_bn >= 0).all(), f"{feature}, {_mask_bn}, {mean_bn}, {var_bn}"
            if self.training:
                with torch.no_grad():
                    # running_mean and running_var should be initialized in
                    # the derived classes by register_buffer().
                    # pylint: disable=attribute-defined-outside-init
                    self.running_mean = exponential_average_factor * mean_bn \
                        + (1 - exponential_average_factor) * self.running_mean
                    self.running_var = exponential_average_factor * var_bn \
                        + (1 - exponential_average_factor) * self.running_var
            elif self.track_running_stats:
                mean_bn = self.running_mean
                var_bn = self.running_var
            mean_bn = mean_bn.expand(size)
            var_bn = var_bn.expand(size)
            stats["masked_batch"]["mean"] = mean_bn
            stats["masked_batch"]["var"] = var_bn

    def forward(self, feature, mask=None):
        """Perform forward computation.

        # Args:
          - feature: `[N, C, *]`, * indicate 1-3 axes.
          - mask: `[N, 1, *]`

        # Returns:
          - feature: `[N, C, *]`
        """
        self._check_input_dim(feature)
        size = feature.size()
        _masks = self._initialize_mask(feature, mask)
        if len(size) == 3:
            axes_in = [2]
            axes_ln = [1, 2]
            axes_bn = [0, 2]
        elif len(size) == 4:
            axes_in = [2, 3]
            axes_ln = [1, 2, 3]
            axes_bn = [0, 2, 3]
        elif len(size) == 5:
            axes_in = [2, 3, 4]
            axes_ln = [1, 2, 3, 4]
            axes_bn = [0, 2, 3, 4]
        axes_pn = [1]

        stats = {"masked_instance": {"mean": None, "var": None},
                 "masked_layer": {"mean": None, "var": None},
                 "masked_positional": {"mean": None, "var": None},
                 "masked_batch": {"mean": None, "var": None}}

        # Masked instance.
        self._apply_masked_instance(feature, _masks, stats, axes_in)

        # Masked layer.
        self._apply_masked_layer(feature, _masks, stats, axes_ln)

        # Masked positional.
        self._apply_masked_positional(feature, _masks, stats, axes_pn)

        # Masked batch.
        self._apply_masked_batch(feature, _masks, stats, axes_bn)

        if len(self.mode) != 1:
            softmax = nn.Softmax(0)
            mean_weight = softmax(self.mean_weight)
            var_weight = softmax(self.var_weight)
        else:
            mean_weight = torch.ones(1)
            var_weight = torch.ones(1)

        mean = None
        var = None
        for i, _mode in enumerate(self.mode):
            _mean = mean_weight[i] * stats[_mode]["mean"]
            _var = var_weight[i] * stats[_mode]["var"]
            if mean is None:
                mean = _mean
                var = _var
            else:
                mean = mean + _mean
                var = var + _var

        outputs = (feature-mean) / (var+self.eps).sqrt()
        outputs = outputs.view(size)

        # Mask out.
        if self.filter_maskedval:
            outputs = outputs * _masks["one"]
        return outputs * self.weight + self.bias

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}, " \
          "track_running_stats={track_running_stats}, last_gamma={last_gamma}, " \
          "mode={mode}, filter_maskedval={filter_maskedval}".format(**self.__dict__)


class MaskedSwitchNorm1d(_MaskedSwitchNormNd):
    """Masked switchable normalization for 1-dimensional features.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True,
                 last_gamma=False, modes=("masked_switch"), filter_maskedval=False):
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         track_running_stats=track_running_stats,
                         last_gamma=last_gamma,
                         modes=modes,
                         filter_maskedval=filter_maskedval)
        if "masked_batch" in self.mode:
            # pylint error of torch.tensor() should be solved in the future release.
            # https://github.com/pytorch/pytorch/issues/24807
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1))
            self.register_buffer("running_var", torch.ones(1, num_features, 1))
            self.register_buffer("num_batches_tracked",
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self._reset_parameters()

    def _check_input_dim(self, feature):
        if feature.dim() != 3:
            raise ValueError(f"expected 3D input (got {feature.dim()}D input)")


class MaskedSwitchNorm2d(_MaskedSwitchNormNd):
    """Masked switchable normalization for 2-dimensional features.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=True,
                 last_gamma=False, modes=("masked_switch"), filter_maskedval=False):
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         track_running_stats=track_running_stats,
                         last_gamma=last_gamma,
                         modes=modes,
                         filter_maskedval=filter_maskedval)
        if "masked_batch" in self.mode:
            # pylint error of torch.tensor() should be solved in the future release.
            # https://github.com/pytorch/pytorch/issues/24807
            self.register_buffer("running_mean", torch.zeros(1, num_features, 1, 1))
            self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
            self.register_buffer("num_batches_tracked",
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self._reset_parameters()

    def _check_input_dim(self, feature):
        if feature.dim() != 4:
            raise ValueError(f"expected 4D input (got {feature.dim()}D input)")


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
