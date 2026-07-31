# Defines the Newton-specific actor builder scaffold.

import newton

from mani_skill.sim.core.builders.actor import BaseActorBuilder


class NewtonActorBuilder(BaseActorBuilder):
    """Newton-specific actor builder."""

    def __init__(self, scene_mb: newton.ModelBuilder) -> None:
        self.scene_mb = scene_mb
        self.body = scene_mb.add_body(label="body")

    def add_box_collision(
        self,
        name: str,
        half_size: tuple[float, float, float],
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        self.scene_mb.add_shape_box(
            body=self.body,
            hx=half_size[0],
            hy=half_size[1],
            hz=half_size[2],
            label=name,
        )

    def add_box_visual(
        self,
        name: str,
        half_size: tuple[float, float, float],
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        self.scene_mb.add_shape_box(
            body=self.body,
            hx=half_size[0],
            hy=half_size[1],
            hz=half_size[2],
            label=name,
        )
