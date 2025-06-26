from jsonargparse import lazy_instance

from ..components import *
from ..data.multi_crop import MultiCropConfig
from ..layers import (
    ClassificationHeadConfig,
    MultiPointPatchify,
    PointPatchify,
    ProjectionHeadConfig,
    TransformerEncoderConfig,
)
from ..layers.tokenization import (
    PatchEmbedding,
    PatchEmbeddingConfig,
)

# For proper handling of immutable defaults


DEFAULT_SUBSAMPLING_TRANSFORM = lazy_instance(FarthestPointSubSamplePC)
DEFAULT_AUG_TRANSFORM = lazy_instance(RandomRotateAxisPC)
DEFAULT_NORM_TRANSFORM = lazy_instance(NormalizeUnitSpherePC)

DEFAULT_MULTI_CROP_CONFIG = lazy_instance(MultiCropConfig)
DEFAULT_MASKING_GENERATOR = lazy_instance(InverseBlockPatchMasking)

DEFAULT_TRANSFORMER_ENC_CONFIG = lazy_instance(TransformerEncoderConfig)
DEFAULT_TRANSFORMER_PROJ_CONFIG = lazy_instance(TransformerEncoderConfig, num_layers=4)
DEFAULT_PROJECTION_HEAD_CONFIG = lazy_instance(ProjectionHeadConfig)
DEFAULT_CLASSIFICATION_HEAD_CONFIG = lazy_instance(ClassificationHeadConfig)

DEFAULT_EMA_DECAY = lazy_instance(
    CosineAnnealingWarmupSchedule,
    base_value=0.995,
    final_value=1.0,
)
DEFAULT_TEACHER_TEMP = lazy_instance(
    LinearWarmupSchedule,
    start_value=0.04,
    final_value=0.07,
)
DEFAULT_OPTIMIZER = lazy_instance(AdamWSpec)

DEFAULT_PATCHIFY = lazy_instance(PointPatchify)
DEFAULT_MULTI_PATCHIFY = lazy_instance(MultiPointPatchify)

DEFAULT_PATCH_EMBEDDING = lazy_instance(PatchEmbedding)
DEFAULT_PATCH_EMBEDDING_CFG = lazy_instance(PatchEmbeddingConfig)
