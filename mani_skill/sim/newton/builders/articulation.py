# Defines the Newton-specific articulation builder scaffold.

import newton

from mani_skill.sim.core.builders.articulation import BaseArticulationBuilder


class NewtonArticulationBuilder(BaseArticulationBuilder):
    """Newton-specific articulation builder."""

    def __init__(self, scene_mb: newton.ModelBuilder) -> None:
        self.scene_mb = scene_mb
