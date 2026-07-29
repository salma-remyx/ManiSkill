from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mani_skill.utils.structs.base import BaseStruct
from mani_skill.utils.structs.link import Link

if TYPE_CHECKING:
    from mani_skill.utils.structs.articulation import Articulation


@dataclass(kw_only=True)
class ArticulationJoint(BaseStruct):
    name: str
    """The name of the joint."""
    index: torch.Tensor
    """index of this joint among all joints"""
    active_index: torch.Tensor | None
    """index of this joint amongst the active joints"""

    articulation: Articulation | None = None
    child_link: Link | None = None
    parent_link: Link | None = None
