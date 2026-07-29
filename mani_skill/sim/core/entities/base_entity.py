import torch


class BaseEntity:
    name: str
    """The name of the entity."""
    scene_idxs: torch.Tensor
    """The indices of the scenes that the entity is in."""
