from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import warp as wp

from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


@dataclass(kw_only=True)
class NewtonArticulation(Articulation):
    model: newton.ModelBuilder
    """The Newton model builder containing this articulation."""
    sim: NewtonSim
    """The Newton simulation containing this articulation."""
    _initial_pose: Pose
    """The pose used when the articulation was added to the scene."""

    @classmethod
    def create_from_model(
        cls,
        model: newton.ModelBuilder,
        sim: NewtonSim,
        name: str,
        initial_pose: Pose,
        xform: wp.transform,
    ):
        sim._scene_mb.add_builder(model, xform=xform, label_prefix=name or None)
        return cls(
            model=model,
            sim=sim,
            name=name,
            _initial_pose=initial_pose,
        )
