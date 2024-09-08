#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""dataset: Dataset manipulation module.
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
import json
from dataclasses import (
    dataclass,
    field)
from functools import partial
from pathlib import (
    Path,
    PurePath)
from typing import (
    Any,
    Dict,
    List)

# Third party's modules
import h5py

import numpy as np

import torch
from torch.utils.data import (
    DataLoader,
    Dataset)

from torchvision.transforms import Compose

# Local modules
from .transforms import (
    ToTensor,
    get_transform)

# Execution settings
VERSION = u"%(prog)s dev"


class HDF5Dataset(Dataset):
    def __init__(self,
                 hdf5files,
                 load_into_ram=False,
                 convert_to_channel_first=False,
                 pre_transforms=None,
                 transforms=None):
        self.convert_to_channel_first = convert_to_channel_first
        self.pre_transforms = pre_transforms
        self.load_into_ram = load_into_ram
        data_info = []
        # Load file pointers.
        for fin in hdf5files:
            swap = 1 if "_swap" in fin.name else 0
            # filename should be [pid].hdf5 or [pid]_swap.hdf5
            pid = int(fin.stem.split("_")[0])
            with h5py.File(fin.resolve(), "r") as fread:
                keys = list(fread.keys())
                for key in keys:
                    if load_into_ram:
                        data = {"feature": fread[key]["feature"][:],
                                "token": fread[key]["token"][:]}
                        if self.convert_to_channel_first:
                            feature = data["feature"]
                            # `[T, J, C] -> [C, T, J]`
                            feature = np.transpose(feature, [2, 0, 1])
                            data["feature"] = feature
                        if self.pre_transforms:
                            data = self.pre_transforms(data)
                    else:
                        data = None
                    data_info.append({
                        "file": fin,
                        "data_key": key,
                        "swap": swap,
                        "pid": pid,
                        "data": data})
        self.data_info = data_info

        # Check and assign transforms.
        self.transforms = self._check_transforms(transforms)

    def _check_transforms(self, transforms):
        # Check transforms.
        if transforms:
            if isinstance(transforms, Compose):
                _transforms = transforms.transforms
            else:
                _transforms = transforms
            check_totensor = False
            for trans in _transforms:
                if isinstance(trans, ToTensor):
                    check_totensor = True
                    break
            message = "Dataset should return torch.Tensor but transforms does " \
                + "not include ToTensor class."
            assert check_totensor, message

        if transforms is None:
            transforms = Compose([ToTensor()])
        elif not isinstance(transforms, Compose):
            transforms = Compose(transforms)
        return transforms

    def __getitem__(self, index):
        info = self.data_info[index]
        if info["data"]:
            data = info["data"]
        else:
            with h5py.File(info["file"], "r") as fread:
                data = {"feature": fread[info["data_key"]]["feature"][:],
                        "token": fread[info["data_key"]]["token"][:]}
        _data = copy.deepcopy(data)
        if self.load_into_ram is False:
            if self.convert_to_channel_first:
                feature = _data["feature"]
                # `[T, J, C] -> [C, T, J]`
                feature = np.transpose(feature, [2, 0, 1])
                _data["feature"] = feature
            if self.pre_transforms:
                _data = self.pre_transforms(_data)
        _data = self.transforms(_data)
        return _data

    def __len__(self):
        return len(self.data_info)


def merge(sequences, merged_shape, padding_val=0):
    merged = torch.full(tuple(merged_shape),
                        padding_val,
                        dtype=sequences[0].dtype)
    if len(merged_shape) == 2:
        for i, seq in enumerate(sequences):
            merged[i,
                   :seq.shape[0]] = seq
    if len(merged_shape) == 3:
        for i, seq in enumerate(sequences):
            merged[i,
                   :seq.shape[0],
                   :seq.shape[1]] = seq
    if len(merged_shape) == 4:
        for i, seq in enumerate(sequences):
            merged[i,
                   :seq.shape[0],
                   :seq.shape[1],
                   :seq.shape[2]] = seq
    if len(merged_shape) == 5:
        for i, seq in enumerate(sequences):
            merged[i,
                   :seq.shape[0],
                   :seq.shape[1],
                   :seq.shape[2],
                   :seq.shape[3]] = seq
    return merged


def merge_padded_batch(batch,
                       feature_shape,
                       token_shape,
                       feature_padding_val=0,
                       token_padding_val=0):
    feature_batch = [sample["feature"] for sample in batch]
    token_batch = [sample["token"] for sample in batch]

    # ==========================================================
    # Merge feature.
    # ==========================================================
    # `[B, C, T, J]`
    merged_shape = [len(batch), *feature_shape]
    # Use maximum frame length in a batch as padded length.
    if merged_shape[2] == -1:
        tlen = max([feature.shape[1] for feature in feature_batch])
        merged_shape[2] = tlen
    merged_feature = merge(feature_batch, merged_shape, padding_val=feature_padding_val)

    # ==========================================================
    # Merge tocken.
    # ==========================================================
    # `[B, L]`
    merged_shape = [len(batch), *token_shape]
    # Use maximum token length in a batch as padded length.
    if merged_shape[1] == -1:
        tlen = max([token.shape[0] for token in token_batch])
        merged_shape[1] = tlen
    merged_token = merge(token_batch, merged_shape, padding_val=token_padding_val)

    # Generate padding mask.
    # Pad: 0, Signal: 1
    # The frames which all channels and landmarks are equals to padding value
    # should be padded.
    feature_pad_mask = merged_feature == feature_padding_val
    feature_pad_mask = torch.all(feature_pad_mask, dim=1)
    feature_pad_mask = torch.all(feature_pad_mask, dim=-1)
    feature_pad_mask = torch.logical_not(feature_pad_mask)
    token_pad_mask = torch.logical_not(merged_token == token_padding_val)

    retval = {
        "feature": merged_feature,
        "token": merged_token,
        "feature_pad_mask": feature_pad_mask,
        "token_pad_mask": token_pad_mask}
    return retval


@dataclass
class DataLoaderSettings():
    batch_size: int = 1
    # Dataset and dataloader.
    load_into_ram: bool = True
    shuffle: bool = True
    val_pid: int = 16069
    test_pid: int = 16069
    include_swap: bool = False
    convert_to_channel_first: bool = False
    dataset_dir: PurePath = field(default_factory=Path(""))
    key2token: Dict[str, int] = field(default_factory=lambda: {})
    token2key: Dict[int, str] = field(default_factory=lambda: {})
    vocabulary: int = 0
    use_landmarks: List[int] = field(default_factory=lambda: [])
    use_features: List[str] = field(default_factory=lambda: ["x", "y", "z"])
    # Transforms.
    pre_transforms_settings: Dict[str, Any] = field(default_factory=lambda: {})
    train_transforms_settings: Dict[str, Any] = field(default_factory=lambda: {})
    val_transforms_settings: Dict[str, Any] = field(default_factory=lambda: {})
    test_transforms_settings: Dict[str, Any] = field(default_factory=lambda: {})
    # Task.
    task_type: str = "islr"
    num_workers: int = 2

    def __post_init__(self):
        # Load files.
        files = list(self.dataset_dir.iterdir())
        dictionary = [fin for fin in files if ".json" in fin.name][0]
        hdf5_files = [fin for fin in files if ".json" not in fin.name]
        if not self.include_swap:
            hdf5_files = [fin for fin in hdf5_files if "_swap" not in fin.name]

        train_hdf5files = [fin for fin in hdf5_files if str(self.val_pid) not in fin.name]
        train_hdf5files = [fin for fin in train_hdf5files if str(self.test_pid) not in fin.name]
        val_hdf5files = [fin for fin in hdf5_files if str(self.val_pid) in fin.name]
        test_hdf5files = [fin for fin in hdf5_files if str(self.test_pid) in fin.name]

        # Load dictionary.
        with open(dictionary, "r") as fread:
            self.key2token = json.load(fread)
            self.token2key = {value: key for key, value in key2token.items()}
        self.vocaburary = len(self.key2token)
        if self.task_type == "cslr_s2s":
            self.key2token["<sos>"] = self.vocaburary
            self.key2token["<eos>"] = self.vocaburary + 1
            self.key2token["<pad>"] = self.vocaburary + 2
            self.token2key[self.vocaburary] = "<sos>"
            self.token2key[self.vocaburary + 1] = "<eos>"
            self.token2key[self.vocaburary + 2] = "<pad>"
        elif self.task_type == "cslr_ctc":
            self.key2token["<pad>"] = self.vocaburary
            self.token2key[self.vocaburary] = "<pad>"
        # Reset.
        self.vocaburary = len(self.key2token)

        # Load transforms.
        pre_transforms = []
        for name, kwargs in self.pre_transforms_settings.items():
            pre_transforms.append(get_transform(name, kwargs))
        pre_transforms = Compose(pre_transforms)

        train_transforms = []
        for name, kwargs in self.train_transforms_settings.items():
            train_transforms.append(get_transform(name, kwargs))
        train_transforms = Compose(train_transforms)

        val_transforms = []
        for name, kwargs in self.val_transforms_settings.items():
            val_transforms.append(get_transform(name, kwargs))
        val_transforms = Compose(val_transforms)

        test_transforms = []
        for name, kwargs in self.test_transforms_settings.items():
            test_transforms.append(get_transform(name, kwargs))
        test_transforms = Compose(test_transforms)

        # Create dataset.
        train_dataset = HDF5Dataset(train_hdf5files,
            convert_to_channel_first=self.convert_to_channel_first,
            pre_transforms=pre_transforms,
            transforms=train_transforms,
            load_into_ram=self.load_into_ram)
        val_dataset = HDF5Dataset(val_hdf5files,
            convert_to_channel_first=self.convert_to_channel_first,
            pre_transforms=pre_transforms,
            transforms=val_transforms,
            load_into_ram=self.load_into_ram)
        test_dataset = HDF5Dataset(test_hdf5files,
            convert_to_channel_first=self.convert_to_channel_first,
            pre_transforms=pre_transforms,
            transforms=test_transforms,
            load_into_ram=self.load_into_ram)

        # Create dataloaders.
        feature_shape = (len(self.use_features), -1, len(self.use_landmarks))
        token_shape = (-1,)
        if self.task_type == "islr":
            token_padding_val = self.vocaburary + 1
        elif self.task_type in ["cslr_s2s", "cslr_ctc"]:
            token_padding_val = self.key2token["<pad>"]
        merge_fn = partial(merge_padded_batch,
                           feature_shape=feature_shape,
                           token_shape=token_shape,
                           feature_padding_val=0.0,
                           token_padding_val=token_padding_val)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, collate_fn=merge_fn,
            shuffle=self.shuffle, num_workers=self.num_workers)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, collate_fn=merge_fn,
            shuffle=False)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=1, collate_fn=merge_fn,
            shuffle=False)

        # Set estimated feature size.
        self.in_channels = len(self.use_features) * len(self.use_landmarks)
        self.out_channels = self.vocabulary


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
