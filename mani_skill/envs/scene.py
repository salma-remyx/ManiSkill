# Coordinates backend-independent scene construction across simulation backends.

from mani_skill.sim.core.base_sim import BaseSim
from mani_skill.sim.core.builders.actor import ActorBuilder
from mani_skill.sim.core.builders.articulation import ArticulationBuilder


class ManiSkillScene:
    physics_sim: BaseSim
    render_sim: BaseSim

    def __init__(
        self,
        *,
        physics_sim: BaseSim,
        render_sim: BaseSim,
    ) -> None:
        self.physics_sim = physics_sim
        self.render_sim = render_sim

    def close(self) -> None:
        """
        Closes the scene.
        """
        self.physics_sim.close()
        self.render_sim.close()

    def create_actor_builder(self) -> ActorBuilder:
        """Create a backend-agnostic actor builder for this scene."""
        return ActorBuilder(scene=self)

    def create_articulation_builder(self) -> ArticulationBuilder:
        """Create a backend-agnostic articulation builder for this scene."""
        return ArticulationBuilder(scene=self)
