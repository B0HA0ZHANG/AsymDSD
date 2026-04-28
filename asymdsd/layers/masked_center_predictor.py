import math
from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn

from ..components import FactoryConfig
from .activation import ActivationLayer
from .multilayer_perceptron import MLP
from .normalization import NormalizationLayer
from .tokenization import TrainableToken
from .transformer import TransformerDecoder, TransformerDecoderConfig


@dataclass
class MaskedCenterPredictorConfig(FactoryConfig):
    embed_dim: int | None = None
    num_heads: int = 6
    num_layers: int = 2
    hidden_ratio: float = 4.0
    norm_layer: NormalizationLayer = nn.LayerNorm
    act_layer: ActivationLayer = nn.GELU
    dropout_p: float = 0.0
    drop_path_p: float = 0.0
    uniform_drop_path: bool = False
    efficient_drop_path: bool = True
    add_pos_enc_every_layer: bool = True
    layer_scale_init: float | None = None
    bias: bool = True
    allow_grad_ckpt: bool = False
    self_attention: bool = True
    concat_tgt_memory: bool = False
    query_fourier_bands: int = 8
    query_hidden_dim: int = 128
    output_hidden_dim: int = 256
    predict_residual: bool = True

    @property
    def CLS(self) -> type["MaskedCenterPredictor"]:
        return MaskedCenterPredictor


class MaskedCenterPredictor(nn.Module):
    def __init__(self, config: MaskedCenterPredictorConfig) -> None:
        super().__init__()
        self.cfg = deepcopy(config)
        if self.cfg.embed_dim is None:
            raise ValueError("MaskedCenterPredictorConfig.embed_dim must be set.")

        self.query_fourier_bands = max(0, self.cfg.query_fourier_bands)
        query_in_dim = 1 + 2 * self.query_fourier_bands

        decoder_cfg = TransformerDecoderConfig(
            embed_dim=self.cfg.embed_dim,
            num_heads=self.cfg.num_heads,
            num_layers=self.cfg.num_layers,
            hidden_ratio=self.cfg.hidden_ratio,
            norm_layer=self.cfg.norm_layer,
            act_layer=self.cfg.act_layer,
            dropout_p=self.cfg.dropout_p,
            drop_path_p=self.cfg.drop_path_p,
            uniform_drop_path=self.cfg.uniform_drop_path,
            efficient_drop_path=self.cfg.efficient_drop_path,
            add_pos_enc_every_layer=self.cfg.add_pos_enc_every_layer,
            layer_scale_init=self.cfg.layer_scale_init,
            bias=self.cfg.bias,
            allow_grad_ckpt=self.cfg.allow_grad_ckpt,
            self_attention=self.cfg.self_attention,
            concat_tgt_memory=self.cfg.concat_tgt_memory,
        )

        self.decoder = TransformerDecoder(decoder_cfg)
        self.query_token = TrainableToken(self.cfg.embed_dim)
        self.query_embedding = MLP(
            in_dim=query_in_dim,
            hidden_dim=self.cfg.query_hidden_dim,
            out_dim=self.cfg.embed_dim,
            norm_layer=self.cfg.norm_layer,
            act_layer=self.cfg.act_layer,
            dropout_p=self.cfg.dropout_p,
            bias=self.cfg.bias,
        )
        self.output_head = MLP(
            in_dim=self.cfg.embed_dim,
            hidden_dim=self.cfg.output_hidden_dim,
            out_dim=3,
            act_layer=self.cfg.act_layer,
            dropout_p=self.cfg.dropout_p,
            bias=self.cfg.bias,
        )

    def enable_gradient_checkpointing(self) -> None:
        self.decoder.enable_gradient_checkpointing()

    def _fourier_encode(
        self,
        masked_patch_indices: torch.Tensor,
        num_patches: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        denom = max(num_patches - 1, 1)
        x = masked_patch_indices.to(dtype=dtype) / denom
        features = [x.unsqueeze(-1)]

        if self.query_fourier_bands > 0:
            freqs = torch.arange(
                self.query_fourier_bands,
                device=masked_patch_indices.device,
                dtype=dtype,
            )
            freqs = (2.0**freqs) * (2.0 * math.pi)
            angles = x.unsqueeze(-1) * freqs
            features.extend((angles.sin(), angles.cos()))

        return torch.cat(features, dim=-1)

    def forward(
        self,
        visible_features: torch.Tensor,
        visible_centers: torch.Tensor,
        masked_patch_indices: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        query_pos = self.query_embedding(
            self._fourier_encode(
                masked_patch_indices,
                num_patches,
                dtype=visible_features.dtype,
            )
        )
        query_tokens = self.query_token.expand(
            masked_patch_indices.shape[0],
            masked_patch_indices.shape[1],
            -1,
        )

        decoded = self.decoder(
            query_tokens,
            query_pos,
            memory=visible_features,
            memory_centers=visible_centers,
        ).x

        pred_centers = torch.tanh(self.output_head(decoded))
        if self.cfg.predict_residual:
            pred_centers = (
                visible_centers.mean(dim=1, keepdim=True) + pred_centers
            ).clamp(-1.0, 1.0)

        return pred_centers
