from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import torch

from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


@dataclass(kw_only=True)
class NewtonActor(Actor):
    model: newton.ModelBuilder
    """The Newton ModelBuilder for the actor."""
    sim: NewtonSim
    """The NewtonSim object this actor is in."""
    _initial_pose: Pose
    """The initial pose of the actor."""

    @classmethod
    def create_from_model(
        cls, model: newton.ModelBuilder, sim: NewtonSim, name: str, initial_pose: Pose
    ):
        sim._scene_mb.add_builder(model, label_prefix=name)
        return cls(
            model=model,
            sim=sim,
            name=name,
            _initial_pose=initial_pose,
        )

    @property
    def linear_velocity(self) -> torch.Tensor:
        pass

    @linear_velocity.setter
    def linear_velocity(self, velocity: torch.Tensor):
        pass

    @property
    def angular_velocity(self) -> torch.Tensor:
        pass

    @angular_velocity.setter
    def angular_velocity(self, velocity: torch.Tensor):
        pass
