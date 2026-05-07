from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import NamedTuple, Self

import torch
from jsonargparse import lazy_instance
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..components import FactoryConfig
from ..components.common_types import OptionalTensor
from ..components.utils import init_lazy_defaults
from .activation import ActivationLayer
from .multilayer_perceptron import MLPVarLen
from .normalization import NormalizationLayer
from .patchify import MultiPatches


class Tokens(NamedTuple):
    embeddings: torch.Tensor
    pos_embeddings: torch.Tensor
    patches: OptionalTensor = None
    centers: OptionalTensor = None


class TrainableToken(nn.Parameter):
    # Simply a wrapper around nn.Parameter
    def __new__(cls, embed_dim: int = 384, num_tokens: int = 1) -> Self:
        return super().__new__(cls, torch.empty((1, num_tokens, embed_dim)))  # type: ignore

    def __init__(self, embed_dim: int = 384, num_tokens: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.embed_dim, self.num_tokens)
            result.data = self.data.clone(memory_format=torch.preserve_format)
            memo[id(self)] = result
            return result


@dataclass
class PositionEmbeddingConfig(FactoryConfig):
    in_features: int = 3
    embed_dim: int = 384
    act_layer: ActivationLayer = nn.GELU
    normalize: bool = False

    @property
    def CLS(self) -> type["PositionEmbedding"]:
        return PositionEmbedding

    def instantiate(self) -> "PositionEmbedding":
        return PositionEmbedding(
            in_features=self.in_features,
            embed_dim=self.embed_dim,
            act_layer=self.act_layer,
            normalize=self.normalize,
        )


@dataclass
class PointEmbeddingConfig(FactoryConfig, ABC):
    in_features: int = 3
    embed_dim: int = 384
    allow_grad_ckpt: bool = True


# Not using dataclass due to jsonargparse's inabbility to deal with different configs
@dataclass
class MemEfficientPointMaxEmbeddingConfig(PointEmbeddingConfig):
    hidden_dims: tuple[int, int, int] = (128, 256, 512)
    act_layer: ActivationLayer = nn.GELU
    norm_layer: NormalizationLayer = nn.LayerNorm
    dropout_p: float = 0.0
    bias: bool = True
    process_num_chunks: int = 1

    @property
    def CLS(self) -> type["MemEfficientPointMaxEmbedding"]:
        return MemEfficientPointMaxEmbedding


@dataclass
class VarMemEfficientPointMaxEmbeddingConfig(PointEmbeddingConfig):
    hidden_dims: list[list[int]] = field(
        default_factory=lambda: [[256, 512, 1024], [2048]]
    )
    act_layer: ActivationLayer = nn.GELU
    norm_layer: NormalizationLayer = nn.LayerNorm
    dropout_p: float = 0.0
    bias: bool = True
    process_num_chunks: int = 1

    def __post_init__(self):
        self.hidden_dims = deepcopy(self.hidden_dims)

    @property
    def CLS(self) -> type["VarMemEfficientPointMaxEmbedding"]:
        return VarMemEfficientPointMaxEmbedding


DEFAULT_POS_EMBED_CFG = lazy_instance(PositionEmbeddingConfig)
DEFAULT_MEPME_CFG = lazy_instance(MemEfficientPointMaxEmbeddingConfig)


@dataclass
class PatchEmbeddingConfig(FactoryConfig):
    position_embedding: PositionEmbeddingConfig = field(
        default_factory=lambda: DEFAULT_POS_EMBED_CFG
    )
    point_embedding: PointEmbeddingConfig = field(
        default_factory=lambda: DEFAULT_MEPME_CFG
    )
    normalize_patches: bool = False

    @property
    def CLS(self) -> type["PatchEmbedding"]:
        return PatchEmbedding

    def instantiate(self) -> "PatchEmbedding":
        return PatchEmbedding(
            position_embedding=self.position_embedding,
            point_embedding=self.point_embedding,
            normalize_patches=self.normalize_patches,
        )


class PointEmbedding(ABC, nn.Module):
    def __init__(self, config: PointEmbeddingConfig) -> None:
        super().__init__()
        self._config = config
        self._gradient_checkpointing = False

    @property
    def config(self) -> PointEmbeddingConfig:
        return deepcopy(self._config)

    def enable_gradient_checkpointing(self) -> None:
        if self._config.allow_grad_ckpt:
            self._gradient_checkpointing = True


class PointMaxEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int,  # >=3
        embed_dim: int,
        hidden_dims: tuple[int, int, int] = (128, 256, 512),
        act_layer: ActivationLayer = nn.GELU,
        norm_layer: NormalizationLayer = nn.LayerNorm,
    ) -> None:
        # TODO: Consider adding shifting layernorm and adding dropout
        # TODO: Consider adding parameters for bias and dims
        super().__init__()
        self.block_1 = nn.Sequential(
            # Without bias deterimental for performance
            nn.Linear(in_features, hidden_dims[0], bias=False),
            # Could also use BatchNorm or LayerNorm only on the last dim
            # Computes element wise affine for every element in normalized_shape
            norm_layer(hidden_dims[0]),  # TODO: Consider moving after act layer
            act_layer(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
        )

        self.block_2 = nn.Sequential(
            # Without bias deterimental for performance
            nn.Linear(2 * hidden_dims[1], hidden_dims[2], bias=False),
            norm_layer(hidden_dims[2]),
            act_layer(),
            nn.Linear(hidden_dims[2], embed_dim),
        )

        # Performs the above using MLP
        # self.block_1 = MLP(
        #     in_features, 128, 256, norm_layer=nn.LayerNorm, act_layer=act_layer
        # )
        # self.block_2 = MLP(
        #     512, 512, embed_dim, norm_layer=nn.LayerNorm, act_layer=act_layer
        # )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.block_1(patches)  # (B, P, K, 256)

        # Put into module
        global_max: torch.Tensor = torch.amax(x, dim=-2, keepdim=True)  # (B, P, 1, 256)

        x = torch.concat([x, global_max.expand(-1, -1, x.shape[-2], -1)], dim=-1)
        # (B, P, K, 512)

        x = self.block_2(x)  # (B, P, K, embed_dim)

        global_max = torch.amax(x, dim=-2)  # (B, P, embed_dim)

        return global_max


class MemEfficientPointMaxEmbedding(PointEmbedding):
    def __init__(
        self,
        config: MemEfficientPointMaxEmbeddingConfig,
    ) -> None:
        # TODO: Consider adding shifting layernorm and adding dropout
        # TODO: Consider adding parameters for bias and dims
        super().__init__(config)
        self.cfg = cfg = deepcopy(config)

        self.block_1 = nn.Sequential(
            # Without bias deterimental for performance
            nn.Linear(cfg.in_features, cfg.hidden_dims[0], bias=False),
            # Could also use BatchNorm or LayerNorm only on the last dim
            # Computes element wise affine for every element in normalized_shape
            cfg.norm_layer(cfg.hidden_dims[0]),  # TODO: Consider moving after act layer
            cfg.act_layer(),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[1], bias=cfg.bias),
        )

        self.block_2 = nn.Sequential(
            # Without bias deterimental for performance
            nn.Linear(2 * cfg.hidden_dims[1], cfg.hidden_dims[2], bias=False),
            cfg.norm_layer(cfg.hidden_dims[2]),
            cfg.act_layer(),
            nn.Linear(cfg.hidden_dims[2], cfg.embed_dim, bias=cfg.bias),
        )

        self._gradient_checkpointing = False

    def embed(self, patches: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.block_1(patches)  # (B, K, 256)

        global_max: torch.Tensor = torch.amax(x, dim=-2, keepdim=True)  # (B, 1, 256)

        x = torch.concat([x, global_max.expand(-1, x.shape[-2], -1)], dim=-1)
        # x = torch.concat([x, global_max.expand(-1, -1, x.shape[-2], -1)], dim=-1)
        # (B, K, 512)

        x = self.block_2(x)  # (B, K, embed_dim)

        global_max = torch.amax(x, dim=-2)  # (B, embed_dim)

        return global_max

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, P, K, _ = patches.shape
        # TODO: Might not be required
        x = patches.view(B * P, K, -1)  # Should be contiguous

        # Does not need to chunk on K dim (1) can also do batchwise
        chunks = torch.chunk(x, self.cfg.process_num_chunks, dim=0)

        embeddings = []

        for chunk in chunks:
            if self._gradient_checkpointing:
                x = checkpoint(self.embed, chunk, use_reentrant=False)
            else:
                x = self.embed(chunk)

            embeddings.append(x)

        embeddings = torch.cat(embeddings, dim=0).view(B, P, -1)

        return embeddings


class VarMemEfficientPointMaxEmbedding(PointEmbedding):
    def __init__(
        self,
        config: VarMemEfficientPointMaxEmbeddingConfig,
    ) -> None:
        # TODO: Consider adding shifting layernorm and adding dropout
        # TODO: Consider adding parameters for bias and dims
        super().__init__(config)
        self.cfg = cfg = deepcopy(config)

        # Add input features to hidden dims
        cfg.hidden_dims[0] = [cfg.in_features, *cfg.hidden_dims[0]]

        # Each block has as input size double that of the output of the previous block
        for i in range(1, len(cfg.hidden_dims)):
            cfg.hidden_dims[i] = [2 * cfg.hidden_dims[i - 1][-1], *cfg.hidden_dims[i]]
        cfg.hidden_dims[-1] = [*cfg.hidden_dims[-1], cfg.embed_dim]

        self.blocks = [
            MLPVarLen(
                *hidden_dims,
                norm_layer=cfg.norm_layer,
                act_layer=cfg.act_layer,
                dropout_p=cfg.dropout_p,
            )
            for hidden_dims in cfg.hidden_dims
        ]
        self.blocks = nn.ModuleList(self.blocks)

        self._gradient_checkpointing = False

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.blocks) - 1):
            x = self.blocks[i](x)
            global_max: torch.Tensor = torch.amax(x, dim=-2, keepdim=True)
            x = torch.concat([x, global_max.expand(-1, x.shape[-2], -1)], dim=-1)

        x = self.blocks[-1](x)
        global_max = torch.amax(x, dim=-2)

        return global_max

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, P, K, _ = patches.shape
        # TODO: Might not be required
        x = patches.view(B * P, K, -1)  # Should be contiguous

        # Does not need to chunk on K dim (1) can also do batchwise
        chunks = torch.chunk(x, self.cfg.process_num_chunks, dim=0)

        embeddings = []

        for chunk in chunks:
            if self._gradient_checkpointing:
                x = checkpoint(self.embed, chunk, use_reentrant=False)
            else:
                x = self.embed(chunk)

            embeddings.append(x)

        embeddings = torch.cat(embeddings, dim=0).view(B, P, -1)

        return embeddings


PatchEmbeddingLayer = (
    type[PointMaxEmbedding]
    | type[MemEfficientPointMaxEmbedding]
    | type[VarMemEfficientPointMaxEmbedding]
)


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        embed_dim: int = 384,
        act_layer: ActivationLayer = nn.GELU,
        normalize: bool = False,
    ) -> None:
        self.in_features = in_features
        self.embed_dim = embed_dim

        super().__init__()
        self.position_embedding = nn.Sequential(
            nn.Linear(in_features, 128),
            act_layer(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim) if normalize else nn.Identity(),
        )

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        return self.position_embedding(centers)


class PatchEmbedding(nn.Module):
    @init_lazy_defaults
    def __init__(
        self,
        position_embedding: PositionEmbedding
        | PositionEmbeddingConfig = DEFAULT_POS_EMBED_CFG,
        point_embedding: PointEmbeddingConfig | PointEmbedding = DEFAULT_MEPME_CFG,
        normalize_patches: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(point_embedding, PointEmbeddingConfig):
            point_embedding = point_embedding.instantiate()  # type: ignore

        self.point_embedding: PointEmbedding = point_embedding  # type: ignore

        embed_dim = self.point_embedding.config.embed_dim

        if isinstance(position_embedding, PositionEmbeddingConfig):
            position_embedding.embed_dim = embed_dim
            position_embedding = position_embedding.instantiate()

        self.position_embedding: PositionEmbedding = position_embedding

        if self.position_embedding.embed_dim != embed_dim:
            raise ValueError(
                "Position embedding must have the same dimension as the point embedding. "
                f"Got {self.position_embedding.embed_dim} and {embed_dim}."
            )

        # TODO: Consider using data statistics for normalization
        def _normalize_fn(x):
            return nn.functional.normalize(
                x,
                dim=(-1, -2, -3),  # type: ignore
            )

        self.normalize = _normalize_fn if normalize_patches else nn.Identity()

    def forward(
        self,
        multi_patches: MultiPatches,
        return_patches: bool = False,
    ) -> Tokens:
        # TODO: Only works for single level patches
        patches, patches_idx, centers = multi_patches
        centers = centers[-1]

        ret_patches = patches + centers.unsqueeze(2) if return_patches else None

        patches = self.normalize(patches)

        token_embeddings = self.point_embedding(patches)
        position_embeddings = self.position_embedding(centers)

        return Tokens(token_embeddings, position_embeddings, ret_patches, centers)
