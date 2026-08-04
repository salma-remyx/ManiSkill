# Defines shared type annotations for backend-agnostic simulation APIs.

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import warp as wp

DeviceLike: TypeAlias = str | wp.Device
ArrayLike: TypeAlias = (
    wp.array[Any, Any]
    | np.ndarray[Any, Any]
    | Sequence[float]
    | Sequence[Sequence[float]]
)
