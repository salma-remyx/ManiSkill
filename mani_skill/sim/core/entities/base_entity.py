from typing import Sequence


class BaseEntity:
    name: str
    """The name of the entity."""
    scene_idxs: Sequence[int]
    """The indices of the scenes that the entity is in."""
