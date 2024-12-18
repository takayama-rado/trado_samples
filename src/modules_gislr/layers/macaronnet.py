#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""macaronnet: Macaron Net layers.
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
import numpy as np

import torch

from pydantic import (
    Field)

from torch import nn

# Local modules
from .misc import (
    ConfiguredModel,
    GPoolRecognitionHeadSettings,
    apply_norm,
    create_norm)
from .transformer import (
    MultiheadAttentionSettings,
    PositionwiseFeedForwardSettings,
    TransformerDecoderSettings,
    TransformerEncoderSettings,
    build_pooling_layer,
    create_decoder_mask,
    create_encoder_mask)

from ..utils import (
    make_causal_mask,
    select_reluwise_activation)

# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


class MacaronNetEncoderLayerSettings(ConfiguredModel):
    dim_model: int = 64
    dim_pffn: int = 256
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    norm_type_sattn: str = Field(default="layer", pattern=r"layer|batch")
    norm_type_pffn: str = Field(default="layer", pattern=r"layer|batch")
    norm_eps: float = 1e-5
    dropout: float = 0.1
    fc_factor: float = 0.5
    shared_pffn: bool = False

    mhsa_settings: MultiheadAttentionSettings = Field(
        default_factory=lambda: MultiheadAttentionSettings())
    pffn_settings: PositionwiseFeedForwardSettings = Field(
        default_factory=lambda: PositionwiseFeedForwardSettings())

    def model_post_init(self, __context):
        # Adjust mhsa_settings.
        self.mhsa_settings.key_dim = self.dim_model
        self.mhsa_settings.query_dim = self.dim_model
        self.mhsa_settings.att_dim = self.dim_model
        self.mhsa_settings.out_dim = self.dim_model
        # Adjust pffn_settings.
        self.pffn_settings.dim_model = self.dim_model
        self.pffn_settings.dim_pffn = self.dim_pffn
        self.pffn_settings.activation = self.activation

        # Propagate.
        self.mhsa_settings.model_post_init(__context)
        self.pffn_settings.model_post_init(__context)

    def build_layer(self):
        return MacaronNetEncoderLayer(self)


class MacaronNetEncoderLayer(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, MacaronNetEncoderLayerSettings)
        self.settings = settings
        self.fc_factor = settings.fc_factor

        #################################################
        # First half PFFN.
        #################################################
        self.norm_pffn1 = create_norm(settings.norm_type_pffn,
            settings.dim_model, settings.norm_eps,
            settings.pffn_settings.add_bias)
        self.pffn1 = settings.pffn_settings.build_layer()

        #################################################
        # MHA.
        #################################################
        self.norm_sattn = create_norm(settings.norm_type_sattn,
            settings.dim_model, settings.norm_eps,
            settings.mhsa_settings.add_bias)
        self.self_attn = settings.mhsa_settings.build_layer()

        #################################################
        # Second half PFFN.
        #################################################
        self.norm_pffn2 = create_norm(settings.norm_type_pffn,
            settings.dim_model, settings.norm_eps,
            settings.pffn_settings.add_bias)
        if settings.shared_pffn:
            self.pffn2 = self.pffn1
        else:
            self.pffn2 = settings.pffn_settings.build_layer()

        self.dropout = nn.Dropout(p=settings.dropout)

        # To store attention weights.
        self.attw = None

    def forward(self,
                feature,
                causal_mask=None,
                src_key_padding_mask=None):
        bsize, qlen = feature.shape[:2]
        san_mask = create_encoder_mask(src_key_padding_mask, causal_mask)

        #################################################
        # First half PFFN.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_pffn1, feature)
        feature = self.pffn1(feature)
        feature = self.fc_factor * self.dropout(feature) + residual

        #################################################
        # MHSA.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_sattn, feature)
        feature, self.attw = self.self_attn(
            key=feature,
            value=feature,
            query=feature,
            mask=san_mask)
        feature = self.dropout(feature) + residual

        #################################################
        # Second half PFFN.
        #################################################
        # `[N, qlen, dim_model]`
        residual = feature
        feature = apply_norm(self.norm_pffn2, feature)
        feature = self.pffn2(feature)
        feature = self.fc_factor * self.dropout(feature) + residual

        return feature


class MacaronNetEnISLRSettings(ConfiguredModel):
    in_channels: int = 64
    inter_channels: int = 64
    out_channels: int = 64
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    pooling_type: str = Field(default="none", pattern=r"none|average|max")

    enlayer_settings: MacaronNetEncoderLayerSettings = Field(
        default_factory=lambda: MacaronNetEncoderLayerSettings())
    encoder_settings: TransformerEncoderSettings = Field(
        default_factory=lambda: TransformerEncoderSettings())

    head_settings: GPoolRecognitionHeadSettings = Field(
        default_factory=lambda: GPoolRecognitionHeadSettings())

    def model_post_init(self, __context):
        # Adjust enlayer_settings.
        self.enlayer_settings.dim_model = self.inter_channels
        self.enlayer_settings.activation = self.activation

        # Adjust head_settings.
        self.head_settings.in_channels = self.inter_channels
        self.head_settings.out_channels = self.out_channels

        # Propagate.
        self.enlayer_settings.model_post_init(__context)
        self.encoder_settings.model_post_init(__context)
        self.head_settings.model_post_init(__context)

    def build_layer(self):
        return MacaronNetEnISLR(self)


class MacaronNetEnISLR(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, MacaronNetEnISLRSettings)
        self.settings = settings

        # Feature extraction.
        self.linear = nn.Linear(settings.in_channels, settings.inter_channels)
        self.activation = select_reluwise_activation(settings.activation)

        self.pooling = build_pooling_layer(settings.pooling_type)

        # Transformer-Encoder.
        enlayer = settings.enlayer_settings.build_layer()
        self.tr_encoder = settings.encoder_settings.build_layer(enlayer)

        self.head = settings.head_settings.build_layer()

    def forward(self,
                feature,
                feature_causal_mask=None,
                feature_pad_mask=None):
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = feature.shape
        feature = feature.permute([0, 2, 1, 3])
        feature = feature.reshape(N, T, -1)

        feature = self.linear(feature)
        if torch.isnan(feature).any():
            raise ValueError()
        feature = self.activation(feature)
        if torch.isnan(feature).any():
            raise ValueError()

        # Apply pooling.
        feature = self.pooling(feature)
        if feature_pad_mask is not None:
            # Cast to apply pooling.
            feature_pad_mask = feature_pad_mask.to(feature.dtype)
            feature_pad_mask = self.pooling(feature_pad_mask.unsqueeze(-1)).squeeze(-1)
            # Binarization.
            # This removes averaged signals with padding.
            feature_pad_mask = feature_pad_mask > 0.5
            if feature_causal_mask is not None:
                feature_causal_mask = make_causal_mask(feature_pad_mask)

        feature = self.tr_encoder(
            feature=feature,
            causal_mask=feature_causal_mask,
            src_key_padding_mask=feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()

        # `[N, T, C] -> [N, C, T]`
        logit = self.head(feature.permute([0, 2, 1]), feature_pad_mask)
        if torch.isnan(feature).any():
            raise ValueError()
        return logit


class MacaronNetDecoderLayerSettings(ConfiguredModel):
    dim_model: int = 64
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    norm_type_sattn: str = Field(default="layer", pattern=r"layer|batch")
    norm_type_cattn: str = Field(default="layer", pattern=r"layer|batch")
    norm_type_pffn: str = Field(default="layer", pattern=r"layer|batch")
    norm_eps: float = 1e-5
    dropout: float = 0.1
    fc_factor: float = 0.5
    shared_pffn: bool = False

    mhsa_settings: MultiheadAttentionSettings = Field(
        default_factory=lambda: MultiheadAttentionSettings())
    mhca_settings: MultiheadAttentionSettings = Field(
        default_factory=lambda: MultiheadAttentionSettings())
    pffn_settings: PositionwiseFeedForwardSettings = Field(
        default_factory=lambda: PositionwiseFeedForwardSettings())

    def model_post_init(self, __context):
        # Adjust mhsa_settings.
        self.mhsa_settings.key_dim = self.dim_model
        self.mhsa_settings.query_dim = self.dim_model
        self.mhsa_settings.att_dim = self.dim_model
        self.mhsa_settings.out_dim = self.dim_model
        # Adjust mhca_settings.
        self.mhca_settings.key_dim = self.dim_model
        self.mhca_settings.query_dim = self.dim_model
        self.mhca_settings.att_dim = self.dim_model
        self.mhca_settings.out_dim = self.dim_model
        # Adjust pffn_settings.
        self.pffn_settings.dim_model = self.dim_model
        self.pffn_settings.activation = self.activation

        # Propagate.
        self.mhsa_settings.model_post_init(__context)
        self.mhca_settings.model_post_init(__context)
        self.pffn_settings.model_post_init(__context)

    def build_layer(self):
        return MacaronNetDecoderLayer(self)


class MacaronNetDecoderLayer(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, MacaronNetDecoderLayerSettings)
        self.settings = settings
        self.fc_factor = settings.fc_factor

        #################################################
        # First half PFFN.
        #################################################
        self.norm_pffn1 = create_norm(
            settings.norm_type_pffn, settings.dim_model, settings.norm_eps,
            settings.pffn_settings.add_bias)
        self.pffn1 = settings.pffn_settings.build_layer()

        #################################################
        # MHSA.
        #################################################
        self.norm_sattn = create_norm(
            settings.norm_type_sattn, settings.dim_model, settings.norm_eps,
            settings.mhsa_settings.add_bias)
        self.self_attn = settings.mhsa_settings.build_layer()

        #################################################
        # MHCA.
        #################################################
        self.norm_cattn = create_norm(
            settings.norm_type_cattn, settings.dim_model, settings.norm_eps,
            settings.mhca_settings.add_bias)
        self.cross_attn = settings.mhca_settings.build_layer()

        #################################################
        # Second half PFFN.
        #################################################
        self.norm_pffn2 = create_norm(
            settings.norm_type_pffn, settings.dim_model, settings.norm_eps,
            settings.pffn_settings.add_bias)
        if settings.shared_pffn:
            self.pffn2 = self.pffn1
        else:
            self.pffn2 = settings.pffn_settings.build_layer()

        self.dropout = nn.Dropout(p=settings.dropout)

        # To store attention weights.
        self.sattw = None
        self.cattw = None

    def forward(self,
                tgt_feature,
                enc_feature,
                tgt_causal_mask=None,
                enc_tgt_causal_mask=None,
                tgt_key_padding_mask=None,
                enc_key_padding_mask=None):

        # Create mask.
        tgt_san_mask, enc_tgt_mask = create_decoder_mask(
            tgt_feature, enc_feature,
            tgt_key_padding_mask, tgt_causal_mask,
            enc_key_padding_mask, enc_tgt_causal_mask)

        #################################################
        # First half PFFN
        #################################################
        residual = tgt_feature
        tgt_feature = apply_norm(self.norm_pffn1, tgt_feature)
        tgt_feature = self.pffn1(tgt_feature)
        tgt_feature = self.fc_factor * self.dropout(tgt_feature) + residual

        #################################################
        # MHSA
        #################################################
        residual = tgt_feature
        tgt_feature = apply_norm(self.norm_sattn, tgt_feature)
        tgt_feature, self.sattw = self.self_attn(
            key=tgt_feature,
            value=tgt_feature,
            query=tgt_feature,
            mask=tgt_san_mask)
        tgt_feature = self.dropout(tgt_feature) + residual

        #################################################
        # MHCA
        #################################################
        residual = tgt_feature
        tgt_feature = apply_norm(self.norm_cattn, tgt_feature)
        tgt_feature, self.cattw = self.cross_attn(
            key=enc_feature,
            value=enc_feature,
            query=tgt_feature,
            mask=enc_tgt_mask)
        tgt_feature = self.dropout(tgt_feature) + residual

        #################################################
        # Second half PFFN
        #################################################
        residual = tgt_feature
        tgt_feature = apply_norm(self.norm_pffn2, tgt_feature)
        tgt_feature = self.pffn2(tgt_feature)
        tgt_feature = self.dropout(tgt_feature) + residual
        return tgt_feature


class MacaronNetCSLRSettings(ConfiguredModel):
    in_channels: int = 64
    inter_channels: int = 64
    out_channels: int = 64
    activation: str = Field(default="relu",
        pattern=r"relu|gelu|swish|silu|mish|geluacc|tanhexp")
    padding_idx: int = 0

    enlayer_settings: MacaronNetEncoderLayerSettings = Field(
        default_factory=lambda: MacaronNetEncoderLayerSettings())
    encoder_settings: TransformerEncoderSettings = Field(
        default_factory=lambda: TransformerEncoderSettings())

    delayer_settings: MacaronNetDecoderLayerSettings = Field(
        default_factory=lambda: MacaronNetDecoderLayerSettings())
    decoder_settings: TransformerDecoderSettings = Field(
        default_factory=lambda: TransformerDecoderSettings())

    def model_post_init(self, __context):
        # Adjust enlayer_settings.
        self.enlayer_settings.dim_model = self.inter_channels
        self.enlayer_settings.activation = self.activation
        # Adjust delayer_settings.
        self.delayer_settings.dim_model = self.inter_channels
        self.delayer_settings.activation = self.activation
        # Adjust decoder_settings.
        self.decoder_settings.padding_idx = self.padding_idx

        # Propagate.
        self.enlayer_settings.model_post_init(__context)
        self.encoder_settings.model_post_init(__context)
        self.delayer_settings.model_post_init(__context)
        self.decoder_settings.model_post_init(__context)

    def build_layer(self):
        return MacaronNetCSLR(self)


class MacaronNetCSLR(nn.Module):
    def __init__(self,
                 settings):
        super().__init__()
        assert isinstance(settings, MacaronNetCSLRSettings)
        self.settings = settings

        # Feature extraction.
        self.linear = nn.Linear(settings.in_channels, settings.inter_channels)
        self.activation = select_reluwise_activation(settings.activation)

        # MacaronNet-Encoder.
        enlayer = settings.enlayer_settings.build_layer()
        self.tr_encoder = settings.encoder_settings.build_layer(enlayer)

        # MacaronNet-Decoder.
        delayer = settings.delayer_settings.build_layer()
        self.tr_decoder = settings.decoder_settings.build_layer(delayer)

    def forward(self,
                src_feature,
                tgt_feature,
                src_causal_mask,
                src_padding_mask,
                tgt_causal_mask,
                tgt_padding_mask):
        """Forward computation for train.
        """
        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = src_feature.shape
        src_feature = src_feature.permute([0, 2, 1, 3])
        src_feature = src_feature.reshape(N, T, -1)

        src_feature = self.linear(src_feature)

        enc_feature = self.tr_encoder(
            feature=src_feature,
            causal_mask=src_causal_mask,
            src_key_padding_mask=src_padding_mask)

        preds = self.tr_decoder(tgt_feature=tgt_feature,
                                enc_feature=enc_feature,
                                tgt_causal_mask=tgt_causal_mask,
                                enc_tgt_causal_mask=None,
                                tgt_key_padding_mask=tgt_padding_mask,
                                enc_key_padding_mask=src_padding_mask)
        # `[N, T, C]`
        return preds

    def inference(self,
                  src_feature,
                  start_id,
                  end_id,
                  src_padding_mask=None,
                  max_seqlen=62):
        """Forward computation for test.
        """

        # Feature extraction.
        # `[N, C, T, J] -> [N, T, C, J] -> [N, T, C*J] -> [N, T, C']`
        N, C, T, J = src_feature.shape
        src_feature = src_feature.permute([0, 2, 1, 3])
        src_feature = src_feature.reshape(N, T, -1)

        src_feature = self.linear(src_feature)

        enc_feature = self.tr_encoder(
            feature=src_feature,
            causal_mask=None,
            src_key_padding_mask=src_padding_mask)

        # Apply decoder.
        dec_inputs = torch.tensor([start_id]).to(src_feature.device)
        # `[N, T]`
        dec_inputs = dec_inputs.reshape([1, 1])
        preds = None
        pred_ids = [start_id]
        for _ in range(max_seqlen):
            pred = self.tr_decoder(
                tgt_feature=dec_inputs,
                enc_feature=enc_feature,
                tgt_causal_mask=None,
                enc_tgt_causal_mask=None,
                tgt_key_padding_mask=None,
                enc_key_padding_mask=src_padding_mask)
            # Extract last prediction.
            pred = pred[:, -1:, :]
            # `[N, T, C]`
            if preds is None:
                preds = pred
            else:
                # Concatenate last elements.
                preds = torch.cat([preds, pred], dim=1)

            pid = torch.argmax(pred, dim=-1)
            dec_inputs = torch.cat([dec_inputs, pid], dim=-1)

            pid = pid.reshape([1]).detach().cpu().numpy()[0]
            pred_ids.append(int(pid))
            if int(pid) == end_id:
                break

        # `[N, T]`
        pred_ids = np.array([pred_ids])
        return pred_ids, preds


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
