from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mani_skill.utils.structs.articulation_joint import ArticulationJoint

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim
    from mani_skill.sim.newton.structs.articulation import NewtonArticulation
    from mani_skill.sim.newton.structs.link import NewtonLink


@dataclass(kw_only=True)
class NewtonArticulationJoint(ArticulationJoint):
    articulation: NewtonArticulation | None = None
    child_link: NewtonLink | None = None
    parent_link: NewtonLink | None = None

    @classmethod
    def create(
        cls,
        sim: NewtonSim,
        scene_idxs: torch.Tensor | list[int],
        name: str,
        joint_index: torch.Tensor,
        active_joint_index: torch.Tensor | None = None,
        articulation: NewtonArticulation | None = None,
    ):
        return cls(
            name=name,
            index=joint_index,
            active_index=active_joint_index,
            sim=sim,
            _scene_idxs=scene_idxs,
            articulation=articulation,
        )
