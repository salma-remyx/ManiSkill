from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton

from mani_skill.utils.structs.actor import Actor

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


@dataclass(kw_only=True)
class NewtonActor(Actor):
    model: newton.ModelBuilder
    """The Newton ModelBuilder for the actor."""
    sim: NewtonSim
    """The NewtonSim object this actor is in."""

    @classmethod
    def create_from_model(cls, model: newton.ModelBuilder, sim: NewtonSim, name: str):
        sim._scene_mb.add_builder(model, label_prefix=name)
        return cls(
            model=model,
            sim=sim,
            name=name,
        )
