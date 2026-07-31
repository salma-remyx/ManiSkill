from mani_skill.envs.base_env import BaseEnv
from mani_skill.envs.registry import register_env


@register_env("PickCube-v1")
class PickCube(BaseEnv):
    pass

    def _load_scene(self) -> None:
        super()._load_scene()

        self._physics_sim.create_actor_builder()
