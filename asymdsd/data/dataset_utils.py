from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from .pc_transforms import PCTransform, RandomizedPCTransform
from .transforms import Compose


def get_dataset_key(
    dataset: Dict[Any, Any], key_priority_list: List[Any]
) -> Any | None:
    for key in key_priority_list:
        if key in dataset:
            return key
    return None


def compose_transform(
    transform: PCTransform | Sequence[PCTransform] | None,
    seed: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if transform is None:
        return lambda x: x

    if isinstance(transform, Sequence):
        for a_transform in transform:
            if isinstance(a_transform, RandomizedPCTransform):
                a_transform.set_seed(seed)
        transform = Compose(transform)  # type: ignore
    elif isinstance(transform, RandomizedPCTransform):
        transform.set_seed(seed)

    return transform  # type: ignore
