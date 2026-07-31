# Defines the Newton simulation backend and its construction entry points.

from mani_skill.sim.core.base_sim import BaseSim, BaseSimConfig
from mani_skill.sim.newton.builders import (
    NewtonActorBuilder,
    NewtonArticulationBuilder,
)


class NewtonSimConfig(BaseSimConfig):
    """Configure the Newton simulation backend."""


class NewtonSim(BaseSim):
    """Provide the Newton implementation scaffold for the core simulator API."""

    id: str = "newton"
    physics_device: str
    render_device: str

    def __init__(
        self,
        num_envs: int = 1,
        cfg: NewtonSimConfig | None = None,
        physics_device: str | None = None,
        render_device: str | None = None,
    ) -> None:
        super().__init__(
            num_envs,
            cfg or NewtonSimConfig(),
            physics_device,
            render_device,
        )

    def create_actor_builder(self) -> NewtonActorBuilder:
        """Create an empty Newton actor builder."""
        return NewtonActorBuilder()

    def create_articulation_builder(self) -> NewtonArticulationBuilder:
        """Create an empty Newton articulation builder."""
        return NewtonArticulationBuilder()

    def create_articulation_builder_from_urdf(
        self, urdf_path: str
    ) -> NewtonArticulationBuilder:
        """Create an empty Newton articulation builder for a URDF path.

        Args:
            urdf_path: Path reserved for the future Newton URDF loader.
        """
        return NewtonArticulationBuilder()

    def compile_render_scene(self) -> None:
        """Leave render-scene compilation empty until Newton rendering is implemented."""

    def can_render(self) -> bool:
        """Return whether the Newton rendering scaffold can render."""
        return False

    def compile_physical_scene(self) -> None:
        """Leave physical-scene compilation empty until Newton physics is implemented."""

    def physics_step(self) -> None:
        """Leave physics stepping empty until Newton physics is implemented."""

    def can_physics(self) -> bool:
        """Return whether the Newton physics scaffold can simulate."""
        return False
