#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""transforms: Transform module.
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
import random
from typing import (
    Any,
    Dict,
    List)

# Third party's modules
import cv2

import numpy as np

import torch

from scipy import interpolate as ip

# Local modules
from .defines import(
    USE_FACE,
    USE_LHAND,
    USE_POSE,
    USE_RHAND,
    get_fullbody_landmarks)

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class ReplaceNan():
    """ Replace NaN value in the feature.
    """
    def __init__(self, replace_val=0.0) -> None:
        self.replace_val = replace_val

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        feature[np.isnan(feature)] = self.replace_val
        data["feature"] = feature
        return data


class SelectLandmarksAndFeature():
    """ Select joint and feature.
    """
    def __init__(self, landmarks, features=["x", "y", "z"]):
        self.landmarks = landmarks
        _features = []
        if "x" in features:
            _features.append(0)
        if "y" in features:
            _features.append(1)
        if "z" in features:
            _features.append(2)
        self.features = np.array(_features, dtype=np.int32)
        assert self.features.shape[0] > 0, f"{self.features}"

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        # `[C, T, J]`
        feature = feature[self.features]
        feature = feature[:, :, self.landmarks]
        data["feature"] = feature
        return data


class AlignAndNormalize():
    def __init__(self,
                 face_head=0,
                 face_num=76,
                 face_origin=[0, 2],
                 face_unit1=[7],
                 face_unit2=[42],
                 lhand_head=76,
                 lhand_num=21,
                 lhand_origin=[0, 2, 5, 9, 13, 17],
                 lhand_unit1=[0],
                 lhand_unit2=[2, 5, 9, 13, 17],
                 pose_head=76+21,
                 pose_num=12,
                 pose_origin=[0, 1],
                 pose_unit1=[0],
                 pose_unit2=[1],
                 rhand_head=76+21+12,
                 rhand_num=21,
                 rhand_origin=[0, 2, 3, 9, 13, 17],
                 rhand_unit1=[0],
                 rhand_unit2=[2, 5, 9, 13, 17],
                 wo_norm=False) -> None:
        self.face_head = face_head
        self.face_num = face_num
        self.face_origin = face_origin
        self.face_unit1 = face_unit1
        self.face_unit2 = face_unit2

        self.lhand_head = lhand_head
        self.lhand_num = lhand_num
        self.lhand_origin = lhand_origin
        self.lhand_unit1 = lhand_unit1
        self.lhand_unit2 = lhand_unit2

        self.pose_head = pose_head
        self.pose_num = pose_num
        self.pose_origin = pose_origin
        self.pose_unit1 = pose_unit1
        self.pose_unit2 = pose_unit2

        self.rhand_head = rhand_head
        self.rhand_num = rhand_num
        self.rhand_origin = rhand_origin
        self.rhand_unit1 = rhand_unit1
        self.rhand_unit2 = rhand_unit2

        self.wo_norm = wo_norm

    def _normalize(self, feature, tmask, origin_lm, unit_lm1, unit_lm2):
        _feature = feature * tmask
        # `[C, T, J] -> [C, T, 1]`
        origin = feature[:, :, origin_lm].mean(axis=-1, keepdims=True)
        unit1 = feature[:, :, unit_lm1].mean(axis=-1, keepdims=True)
        unit2 = feature[:, :, unit_lm2].mean(axis=-1, keepdims=True)
        unit = np.abs(unit1 - unit2).mean(axis=1, keepdims=True)
        unit = np.clip(unit, a_min=1.0, a_max=None)

        _feature = _feature - origin
        _feature = _feature / unit if self.wo_norm is False else _feature
        _feature = _feature * tmask
        return _feature

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        feature = data["feature"]
        tmask = feature == 0.0
        tmask = np.all(tmask, axis=(0, 2))
        tmask = np.logical_not(tmask.reshape([1, -1, 1]))

        if self.face_num > 0:
            face = feature[:, :, self.face_head: self.face_head+self.face_num]
            face = self._normalize(face, tmask, self.face_origin,
                                   self.face_unit1, self.face_unit2)
            feature[:, :, self.face_head: self.face_head+self.face_num] = face
        if self.lhand_num > 0:
            lhand = feature[:, :, self.lhand_head: self.lhand_head+self.lhand_num]
            lhand = self._normalize(lhand, tmask, self.lhand_origin,
                                    self.lhand_unit1, self.lhand_unit2)
            feature[:, :, self.lhand_head: self.lhand_head+self.lhand_num] = lhand
        if self.pose_num > 0:
            pose = feature[:, :, self.pose_head: self.pose_head+self.pose_num]
            pose = self._normalize(pose, tmask, self.pose_origin,
                                   self.pose_unit1, self.pose_unit2)
            feature[:, :, self.pose_head: self.pose_head+self.pose_num] = pose
        if self.rhand_num > 0:
            rhand = feature[:, :, self.rhand_head: self.rhand_head+self.rhand_num]
            rhand = self._normalize(rhand, tmask, self.rhand_origin,
                                    self.rhand_unit1, self.rhand_unit2)
            feature[:, :, self.rhand_head: self.rhand_head+self.rhand_num] = rhand
        data["feature"] = feature
        return data


class ToTensor():
    """ Convert data to torch.Tensor.
    """
    def __init__(self) -> None:
        pass

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        new_data = {}
        for key, val in data.items():
            if val is not None:
                if isinstance(val, list):
                    for i, subval in enumerate(val):
                        if subval.dtype in [float, np.float64]:
                            # pylint: disable=no-member
                            val[i] = torch.from_numpy(subval.astype(np.float32))
                        else:
                            val[i] = torch.from_numpy(subval)  # pylint: disable=no-member
                elif isinstance(val, np.ndarray):
                    if val.dtype in [float, np.float64]:
                        # pylint: disable=no-member
                        val = torch.from_numpy(val.astype(np.float32))
                    else:
                        val = torch.from_numpy(val)  # pylint: disable=no-member
            new_data[key] = val
        return new_data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


class RandomHorizontalFlip():
    """Horizontally flip the keypoints randomly.

    # Args:
      - apply_ratio: The ratio to apply augmentation.
      - num_joints: The number of input joints.
      - swap_pairs: The pairs of exchanged joint indices.
      - basewidth: The width of the image whose joints are extracted from.
        The swapped joints are calculated as bellow.
        swapped_x = -orig_x + basewidth
      - feature_dim: The expected dimension of each joint.
      - include_conf: If True, we assume each joint includes confidence value
        after the coordinates.
    """
    def __init__(self,
                 apply_ratio: float,
                 num_joints: int,
                 swap_pairs: List[List[int]],
                 basewidth: int,
                 feature_dim: int = 2,
                 include_conf: bool = True) -> None:
        self.apply_ratio = apply_ratio
        self.basewidth = basewidth
        self.feature_dim = feature_dim + 1 if include_conf else feature_dim

        indices = np.arange(0, num_joints, 1)
        for pair in swap_pairs:
            indices[pair[0]] = pair[1]
            indices[pair[1]] = pair[0]
        self.swap_indices = indices

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute flipping.
        """
        if random.random() > self.apply_ratio:
            return data

        feature = data["feature"]
        # Flip positions.
        temp = copy.deepcopy(feature)
        shape = temp.shape
        # `[C, T, J] -> [C, T*J]`
        temp = temp.reshape([self.feature_dim, -1])

        # Filter out when all elements == 0.
        # (it should be confidence == 0).
        row_indices = np.where((temp == 0).all(axis=0) == np.True_)[0]
        temp[0, :] = -temp[0, :] + self.basewidth
        temp[:, row_indices] = np.array([0] * self.feature_dim)[:, None]
        temp = temp.reshape(shape)
        temp = temp[:, :, self.swap_indices]
        assert temp.shape == feature.shape, f"{temp.shape}, {feature.shape}"

        data["feature"] = temp
        return data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


class RandomClip():
    """Clip sequence randomly.

    # Args:
      - apply_ratio: The ratio to apply augmentation.
      - clip_range: The range of clipping.
      - min_apply_size: The minimum temporal length to apply augmentation.
    """
    def __init__(self,
                 apply_ratio,
                 clip_range,
                 min_apply_size=10):
        self.apply_ratio = apply_ratio
        self.clip_range = clip_range
        self.min_apply_size = min_apply_size

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute clipping.
        """
        if random.random() > self.apply_ratio:
            return data

        feature = data["feature"]
        base_timelen = feature.shape[1]
        if base_timelen > self.min_apply_size:
            aug_tscale = np.random.random() * (self.clip_range[1] - self.clip_range[0]) + self.clip_range[0]
            aug_timelen = int(base_timelen * aug_tscale)
            if aug_timelen < base_timelen:
                offset = np.random.randint(0, base_timelen - aug_timelen)
                # `[C, T, J]`
                feature = feature[:, offset: aug_timelen + offset]

        data["feature"] = feature
        return data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


def matrix_interp(x, xs, ys):
    orig_shape = ys.shape

    # ========================================================================
    # Interpolation.
    # ========================================================================
    xs = xs.astype(ys.dtype)
    x = x.astype(ys.dtype)
    # Pad control points for extrapolation.
    xs = np.concatenate([[np.finfo(xs.dtype).min], xs, [np.finfo(xs.dtype).max]], axis=0)
    ys = np.concatenate([ys[:1], ys, ys[-1:]], axis=0)

    # Compute slopes, pad at the edges to flatten.
    sloops = (ys[1:] - ys[:-1]) / np.expand_dims((xs[1:] - xs[:-1]), axis=-1)
    sloops = np.pad(sloops[:-1], [(1, 1), (0, 0)])

    # Solve for intercepts.
    intercepts = ys - sloops * np.expand_dims(xs, axis=-1)

    # Search for the line parameters at each input data point.
    # Create a grid of the inputs and piece breakpoints for thresholding.
    # Rely on argmax stopping on the first true when there are duplicates,
    # which gives us an index into the parameter vectors.
    idx = np.argmax(np.expand_dims(xs, axis=-2) > np.expand_dims(x, axis=-1), axis=-1)
    sloop = sloops[idx]
    intercept = intercepts[idx]

    # Apply the linear mapping at each input data point.
    y = sloop * np.expand_dims(x, axis=-1) + intercept
    y = y.astype(ys.dtype)
    y = y.reshape([-1, orig_shape[1]])
    return y


class RandomTimeWarping():
    """Apply time warping randomly.

    # Args:
      - apply_ratio: The ratio to apply augmentation.
      - scale_range: The range of scale to warp sequence.
      - min_apply_size: The minimum temporal length to apply augmentation.
    """
    def __init__(self,
                 apply_ratio,
                 scale_range,
                 min_apply_size):
        self.apply_ratio = apply_ratio
        self.scale_range = scale_range
        self.min_apply_size = min_apply_size

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply time warping to data.
        """
        if random.random() > self.apply_ratio:
            return data

        feature = data["feature"]
        orig_shape = feature.shape
        tlen = orig_shape[1]
        if tlen > self.min_apply_size:
            # `[C, T, J] -> [T, C*J]`
            feature = feature.transpose([1, 0, 2]).reshape([tlen, -1])

            assert len(feature.shape) == 2, f"{feature.shape}"
            aug_scale = np.random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            aug_tlen = int((tlen-1) * aug_scale)

            x = np.linspace(0, tlen-1, num=aug_tlen)
            xs = np.arange(tlen)
            newfeature = matrix_interp(x, xs, feature)
            # `[T, C*J] -> [T, C, J] -> [C, T, J]`
            newfeature = newfeature.reshape([-1, orig_shape[0], orig_shape[2]])
            newfeature = newfeature.transpose([1, 0, 2])
        else:
            newfeature = feature

        data["feature"] = newfeature
        return data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


def get_affine_matrix_2d(center,
                         trans,
                         scale,
                         rot,
                         skew,
                         to_radians=True,
                         order=["center", "scale", "rot", "skew", "trans"],
                         dtype=np.float32):
    center = np.array(center)
    trans = np.array(trans)
    scale = np.array(scale)
    center_m = np.array([[1, 0, -center[0]],
                         [0, 1, -center[1]],
                         [0, 0, 1]])
    scale_m = np.array([[scale[0], 0, 0],
                        [0, scale[1], 0],
                        [0, 0, 1]])
    rot = np.radians(rot) if to_radians else rot
    _cos = np.cos(rot)
    _sin = np.sin(rot)
    rot_m = np.array([[_cos, -_sin, 0],
                      [_sin, _cos, 0],
                      [0, 0, 1]])
    _tan = np.tan(np.radians(skew)) if to_radians else np.tan(skew)
    skew_m = np.array([[1, _tan[0], 0],
                       [_tan[1], 1, 0],
                       [0, 0, 1]])
    move = center + trans
    trans_m = np.array([[1, 0, move[0]],
                        [0, 1, move[1]],
                        [0, 0, 1]])
    mat = np.identity(3, dtype=dtype)
    for name in order:
        if name == "center":
            mat = np.matmul(center_m, mat)
        if name == "scale":
            mat = np.matmul(scale_m, mat)
        if name == "rot":
            mat = np.matmul(rot_m, mat)
        if name == "skew":
            mat = np.matmul(skew_m, mat)
        if name == "trans":
            mat = np.matmul(trans_m, mat)
    return mat.astype(dtype)


def apply_affine(inputs, mat):
    xy = inputs[:, :, :2]
    xy = np.concatenate([xy, np.ones([xy.shape[0], xy.shape[1], 1])], axis=-1)
    xy = np.einsum("...j,ij", xy, mat)
    inputs[:, :, :2] = xy[:, :, :-1]
    return inputs


class RandomAffineTransform2D():
    def __init__(self,
                 apply_ratio,
                 center_joints,
                 trans_range,
                 scale_range,
                 rot_range,
                 skew_range,
                 random_seed=None,
                 order=["center", "scale", "rot", "skew", "trans"],
                 dtype=np.float32):
        self.apply_ratio = apply_ratio
        self.center_joints = center_joints
        self.trans_range = trans_range
        self.scale_range = scale_range
        self.rot_range = np.radians(rot_range)
        self.skew_range = np.radians(skew_range)
        self.order = order
        self.dtype = dtype
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng()

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        if self.rng.uniform() > self.apply_ratio:
            return data
        # `[C, T, J]`
        feature = data["feature"]

        # Calculate center position.
        temp = feature[:, :, self.center_joints]
        mask = np.sum(temp, axis=(0, 2)) != 0
        if np.all(mask == np.False_):
            return data

        # Use x and y only.
        # `[C, T, J] -> [C, J] -> [C]`
        center = temp[:, mask].mean(axis=1).mean(axis=1)[:2]

        trans = self.rng.uniform(self.trans_range[0], self.trans_range[1], 2)
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1], 2)
        rot = self.rng.uniform(self.rot_range[0], self.rot_range[1])
        skew = self.rng.uniform(self.skew_range[0], self.skew_range[1], 2)

        # Calculate matrix.
        mat = get_affine_matrix_2d(center, trans, scale, rot, skew,
            to_radians=False, order=self.order, dtype=self.dtype)

        # Apply transform.
        feature = apply_affine(feature, mat)
        data["feature"] = feature
        return data


class RandomNoise():
    """Add random noise to joints coordinates.

    # Args:
      - stds: The standard deviation of noise distribution.
      - target_joints: If defined, the noise is added only to target_joints.
      - feature_dim: The expected dimension of each joint.
      - include_conf: If True, we assume each joint includes confidence value
        after the coordinates.
    """
    def __init__(self,
                 apply_ratio,
                 stds,
                 target_joints=None,
                 feature_dim=3,
                 include_conf=True):
        self.apply_ratio = apply_ratio
        self.feature_dim = feature_dim
        self.include_conf = include_conf
        if isinstance(stds, (float, int)):
            self.stds = [float(stds)]
        else:
            self.stds = stds
        if target_joints is not None:
            assert len(self.stds) == len(target_joints)
        self.target_joints = target_joints

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:

        if random.random() > self.apply_ratio:
            return data

        feature = data["feature"]
        # Remove confidence if it's included.
        if self.include_conf:
            confs = feature[-1:, :, :]
            feature = feature[:-1, :, :]
        else:
            confs = None

        mask = np.all(feature != 0.0, axis=0)[None, :, :]
        if self.target_joints is None:
            feature += np.random.normal(size=feature.shape,
                                        scale=self.stds[0])
        else:
            for std, target in zip(self.stds, self.target_joints):
                feature[:, :, target] += np.random.normal(
                    size=feature[:, :, target].shape, scale=std)
        # Filter out noise to zero.
        feature *= mask
        # Back confidence.
        if confs is not None:
            feature = np.concatenate([feature, confs], axis=0)

        data["feature"] = feature
        return data

    def __str__(self):
        return f"{self.__class__.__name__}:{self.__dict__}"


class RandomDropJoints():
    def __init__(self,
                 apply_ratio,
                 drop_joints):
        self.apply_ratio = apply_ratio
        self.drop_joints = drop_joints

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.apply_ratio:
            return data

        # `[C, T, J]`
        feature = copy.deepcopy(data["feature"])
        feature[:, :, self.drop_joints] = 0.0
        # Avoid to drop all signals.
        if not (feature == 0.0).all():
            data["feature"] = feature
        return data


class RandomDropFaceOrPose():
    def __init__(self,
                 apply_ratio,
                 face_head=0,
                 face_num=len(USE_FACE),
                 pose_head=len(USE_FACE)+len(USE_LHAND),
                 pose_num=len(USE_POSE)):
        landmarks, _ = get_fullbody_landmarks()
        face_joints = landmarks[face_head: face_head + face_num]
        pose_joints = landmarks[pose_head: pose_head + pose_num]
        self.apply_ratio = apply_ratio

        self.drop_face = RandomDropJoints(
            apply_ratio=1.0,
            drop_joints=face_joints)
        self.drop_pose = RandomDropJoints(
            apply_ratio=1.0,
            drop_joints=pose_joints)

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:

        rval = random.random()
        if rval <= self.apply_ratio:
            if rval <= self.apply_ratio / 2:
                data = self.drop_face(data)
            else:
                data = self.drop_pose(data)
        return data


class RandomDropHand():
    def __init__(self,
                 apply_ratio,
                 lhand_head=len(USE_FACE),
                 lhand_num=len(USE_LHAND),
                 rhand_head=len(USE_FACE) + len(USE_LHAND) + len(USE_POSE),
                 rhand_num=len(USE_RHAND)):
        landmarks, _ = get_fullbody_landmarks()
        lhand_joints = landmarks[lhand_head: lhand_head + lhand_num]
        rhand_joints = landmarks[rhand_head: rhand_head + rhand_num]
        hand_joints = np.concatenate([lhand_joints, rhand_joints])
        self.apply_ratio = apply_ratio
        self.drop_hand = RandomDropJoints(
            apply_ratio=1.0,
            drop_joints=hand_joints)

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.apply_ratio:
            return data
        return self.drop_hand(data)


class RandomDropFingers():
    def __init__(self,
                 apply_ratio,
                 lhand_head=len(USE_FACE),
                 lhand_num=len(USE_LHAND),
                 rhand_head=len(USE_FACE) + len(USE_LHAND) + len(USE_POSE),
                 rhand_num=len(USE_RHAND),
                 drop_tsize=(0.1, 0.2),
                 num_drop_fingers=(2, 6),
                 num_drops=(2, 3)):
        self.lhand_head = lhand_head
        self.lhand_num = lhand_num
        self.rhand_head = rhand_head
        self.rhand_num = rhand_num
        self.apply_ratio = apply_ratio
        self.drop_tsize = drop_tsize
        self.num_drop_fingers = num_drop_fingers
        self.num_drops = num_drops

        lhand_indices = np.arange(lhand_head, lhand_head+lhand_num)
        rhand_indices = np.arange(rhand_head, rhand_head+rhand_num)
        self.hand_indices = np.concatenate([lhand_indices, rhand_indices], axis=0)

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.apply_ratio:
            return data

        feature = copy.deepcopy(data["feature"])
        tlength = feature.shape[1]
        num_drops = np.random.randint(low=self.num_drops[0], high=self.num_drops[1])
        num_drop_fingers = np.random.randint(
            low=self.num_drop_fingers[0],
            high=self.num_drop_fingers[1],
            size=num_drops)
        offset = 0
        for i, drop in enumerate(num_drop_fingers):
            drop_indices = np.random.random(drop) * (len(self.hand_indices) -1)
            drop_indices = np.unique(np.sort(np.round(drop_indices)))
            drop_indices = np.where(drop_indices < self.lhand_num,
                drop_indices + self.lhand_head,
                drop_indices + self.rhand_head - self.lhand_num)
            maxval = int(tlength / num_drops * (i+1))
            maxval = max(maxval, 1)
            start = int(np.round(np.random.randint(low=offset, high=maxval)))
            tsize = np.random.random() * (self.drop_tsize[1] - self.drop_tsize[0]) + self.drop_tsize[0]
            end = int(np.round(tsize * tlength + start))
            start = min(start, tlength-1)
            end = min(end, tlength)
            feature[:, start:end, drop_indices.astype(np.int32)] = 0.0

            offset = end
        # Avoid to drop all signals.
        if not (feature == 0.0).all():
            data["feature"] = feature
        return data


class RandomSpatialMasking():
    def __init__(self,
                 apply_ratio=1.0,
                 size=(0.2, 0.4)):
        self.apply_ratio = apply_ratio
        self.size = size

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.apply_ratio:
            return data

        feature = copy.deepcopy(data["feature"])
        minimums = np.min(feature, axis=(1, 2))
        maximums = np.max(feature, axis=(1, 2))

        min_x = minimums[0]
        min_y = minimums[1]
        max_x = maximums[0]
        max_y = maximums[1]

        if minimums.shape[0] == 3:
            min_z = minimums[2]
            max_z = minimums[2]
        else:
            min_z = None
            max_z = None

        mask_offset_x = np.random.random() * (max_x - min_x) + min_x
        mask_offset_y = np.random.random() * (max_y - min_y) + min_y

        mask_size = np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
        mask_size_x = (max_x - min_x) * mask_size
        mask_size_y = (max_y - min_y) * mask_size

        mask_x = (mask_offset_x <= feature[0, :, :])
        mask_x = mask_x * (feature[0, :, :] <= (mask_offset_x + mask_size_x))
        mask_y = (mask_offset_y <= feature[1, :, :])
        mask_y = mask_y * (feature[1, :, :] <= (mask_offset_y + mask_size_y))
        mask = mask_x & mask_y
        if min_z:
            mask_offset_z = np.random.random() * (max_z - min_z) + min_z
            mask_size_z = (max_z - min_z) * mask_size
            mask_z = (mask_offset_z <= feature[2, :, :])
            mask_z = mask_z * (feature[2, :, :] <= (mask_offset_z + mask_size_z))
            mask = mask & mask_z

        feature = np.where(mask[None, ...], 0.0, feature)
        # Avoid to drop all signals.
        if not (feature == 0.0).all():
            data["feature"] = feature
        return data


class RandomTemporalMasking():
    def __init__(self,
                 apply_ratio=1.0,
                 size=(0.1, 0.5)):
        self.apply_ratio = apply_ratio
        self.size = size

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.apply_ratio:
            return data

        feature = copy.deepcopy(data["feature"])

        # Calculate drop range.
        tlength = feature.shape[1]
        size = np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
        start = np.random.random()
        end = start + size
        start = int(tlength * start)
        end = min(int(tlength * end), tlength)

        # Masking.
        feature[:, start: end, :] = 0.0

        # Avoid to drop all signals.
        if not (feature == 0.0).all():
            data["feature"] = feature
        return data


class SelectiveResize():
    def __init__(self,
                 min_tlen=10,
                 max_tlen=None):
        assert min_tlen is not None or max_tlen is not None
        self.min_tlen = min_tlen
        self.max_tlen = max_tlen

    def __call__(self,
                 data: Dict[str, Any]) -> Dict[str, Any]:

        feature = data["feature"]
        orig_shape = feature.shape
        tlen = orig_shape[1]
        resize_mode = None
        if self.min_tlen and tlen < self.min_tlen:
            resize_mode = "enlarge"
        if self.max_tlen and tlen > self.max_tlen:
            resize_mode = "shrink"
        if resize_mode is None:
            return data

        # Apply warping.
        # `[C, T, J] -> [T, C*J]`
        feature = feature.transpose([1, 0, 2]).reshape([tlen, -1])

        assert len(feature.shape) == 2, f"{feature.shape}"
        aug_tlen = self.min_tlen if resize_mode == "enlarge" else self.max_tlen

        x = np.linspace(0, tlen-1, num=aug_tlen)
        xs = np.arange(tlen)
        newfeature = matrix_interp(x, xs, feature)
        # `[T, C*J] -> [T, C, J] -> [C, T, J]`
        newfeature = newfeature.reshape([-1, orig_shape[0], orig_shape[2]])
        newfeature = newfeature.transpose([1, 0, 2])

        data["feature"] = newfeature
        return data


Mappings = {
    "replace_nan": ReplaceNan,
    "select_landmarks_and_feature": SelectLandmarksAndFeature,
    "align_and_normalize": AlignAndNormalize,
    "to_tensor": ToTensor,
    "random_horizontal_flip": RandomHorizontalFlip,
    "random_clip": RandomClip,
    "random_time_warping": RandomTimeWarping,
    "random_affine_transform2d": RandomAffineTransform2D,
    "random_noise": RandomNoise,
    "random_drop_joints": RandomDropJoints,
    "random_drop_face_or_pose": RandomDropFaceOrPose,
    "random_drop_hand": RandomDropHand,
    "random_drop_fingers": RandomDropFingers,
    "random_spatial_masking": RandomSpatialMasking,
    "random_temporal_masking": RandomTemporalMasking,
    "selective_resize": SelectiveResize}


def get_transform(name, kwargs):
    cls_obj = Mappings[name]
    if len(kwargs) != 0:
        transform = cls_obj(**kwargs)
    else:
        transform = cls_obj()
    return transform


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
