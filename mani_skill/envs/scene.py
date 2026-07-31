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

    def create_actor_builder(self) -> ActorBuilder:
        physics_actor_builder = self.physics_sim.create_actor_builder()
        if self.physics_sim == self.render_sim:
            render_actor_builder = physics_actor_builder
        else:
            render_actor_builder = self.render_sim.create_actor_builder()

        return ActorBuilder(physics_actor_builder, render_actor_builder)

    def create_articulation_builder(self) -> ArticulationBuilder:
        physics_builder = self.physics_sim.create_articulation_builder()
        if self.physics_sim == self.render_sim:
            render_builder = physics_builder
        else:
            render_builder = self.render_sim.create_articulation_builder()
        return ArticulationBuilder(physics_builder, render_builder)
