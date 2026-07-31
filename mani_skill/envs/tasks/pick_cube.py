from mani_skill.envs.base_env import BaseEnv
from mani_skill.envs.registry import register_env


@register_env("PickCube-v1")
class PickCube(BaseEnv):
    pass

    def _load_scene(self) -> None:
        super()._load_scene()

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            name="cube",
            half_size=(0.05, 0.05, 0.05),
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        builder.add_box_visual(
            name="cube",
            half_size=(0.05, 0.05, 0.05),
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
