class BaseActorBuilder:
    def add_box_collision(
        self,
        name: str,
        half_size: tuple[float, float, float],
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        raise NotImplementedError

    def add_box_visual(
        self,
        name: str,
        half_size: tuple[float, float, float],
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        raise NotImplementedError


class ActorBuilder(BaseActorBuilder):
    """A multi-backend actor builder."""

    def __init__(
        self,
        physics_actor_builder: BaseActorBuilder,
        render_actor_builder: BaseActorBuilder,
    ) -> None:
        self.physics_actor_builder = physics_actor_builder
        self.render_actor_builder = render_actor_builder

    def add_box_collision(
        self,
        name: str,
        half_size: tuple[float, float, float],
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        self.physics_actor_builder.add_box_collision(
            name, half_size, position, orientation
        )

    def add_box_visual(
        self,
        name: str,
        half_size: tuple[float, float, float],
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        self.render_actor_builder.add_box_visual(name, half_size, position, orientation)
