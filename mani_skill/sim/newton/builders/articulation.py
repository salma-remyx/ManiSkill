from typing import TYPE_CHECKING

import newton

from mani_skill.sim.builders.articulation import BaseArticulationBuilder

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


class NewtonArticulationBuilder(BaseArticulationBuilder):
    sim: NewtonSim

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this articulation in. If None, the articulation will
    be built in all scenes."""

    _mb: newton.ModelBuilder
    """The model builder for the articulation."""

    def __init__(self):
        super().__init__()
        self._mb = newton.ModelBuilder()
