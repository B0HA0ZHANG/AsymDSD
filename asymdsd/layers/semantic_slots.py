from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from ..components import FactoryConfig
from .activation import ActivationLayer
from .normalization import NormalizationLayer
from .transformer import TransformerDecoder, TransformerDecoderConfig


class SemanticSlotOutput(NamedTuple):
    x: torch.Tensor
    attn_weights: list[torch.Tensor] | None = None


@dataclass
class SemanticSlotConfig(FactoryConfig):
    num_slots: int = 16
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 2
    hidden_ratio: float = 4.0
    norm_layer: NormalizationLayer = nn.LayerNorm
    act_layer: ActivationLayer = nn.GELU
    dropout_p: float = 0.0
    drop_path_p: float = 0.0
    uniform_drop_path: bool = False
    efficient_drop_path: bool = True
    add_pos_enc_every_layer: bool = False
    layer_scale_init: float | None = None
    bias: bool = True
    allow_grad_ckpt: bool = False
    self_attention: bool = True
    init_std: float = 0.02

    @property
    def CLS(self):
        return SemanticSlotBottleneck

    def instantiate(self) -> "SemanticSlotBottleneck":
        return self.CLS(self)


class SemanticSlotBottleneck(nn.Module):
    def __init__(self, config: SemanticSlotConfig) -> None:
        super().__init__()
        if config.num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {config.num_slots}.")

        self.config = config
        self.slots = nn.Parameter(torch.empty(1, config.num_slots, config.embed_dim))
        nn.init.trunc_normal_(self.slots, std=config.init_std)

        decoder_config = TransformerDecoderConfig(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            hidden_ratio=config.hidden_ratio,
            norm_layer=config.norm_layer,
            act_layer=config.act_layer,
            dropout_p=config.dropout_p,
            drop_path_p=config.drop_path_p,
            uniform_drop_path=config.uniform_drop_path,
            efficient_drop_path=config.efficient_drop_path,
            add_pos_enc_every_layer=config.add_pos_enc_every_layer,
            layer_scale_init=config.layer_scale_init,
            bias=config.bias,
            allow_grad_ckpt=config.allow_grad_ckpt,
            relative_3d_bias=None,
            self_attention=config.self_attention,
            concat_tgt_memory=False,
        )
        self.decoder = TransformerDecoder(decoder_config)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> SemanticSlotOutput:
        slots = self.slots.expand(tokens.size(0), -1, -1)
        slot_pos = torch.zeros_like(slots)

        out = self.decoder(
            slots,
            slot_pos,
            memory=tokens,
            return_attention=return_attention,
        )
        return SemanticSlotOutput(x=out.x, attn_weights=out.attn_weights)

    def enable_gradient_checkpointing(self) -> None:
        self.decoder.enable_gradient_checkpointing()


class SemanticSlotPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        norm_layer: NormalizationLayer = nn.LayerNorm,
        act_layer: ActivationLayer = nn.GELU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        self.net = nn.Sequential(
            norm_layer(embed_dim),
            nn.Linear(embed_dim, hidden_dim, bias=bias),
            act_layer(),
            nn.Linear(hidden_dim, embed_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def semantic_slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    slots = F.normalize(slots, dim=-1, eps=1e-6)
    sim = torch.bmm(slots, slots.transpose(1, 2))
    eye = torch.eye(sim.size(-1), dtype=torch.bool, device=sim.device)
    off_diag = sim[:, ~eye]
    return off_diag.square().mean()
