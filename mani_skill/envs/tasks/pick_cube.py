from mani_skill.envs.base_env import BaseEnv
from mani_skill.envs.registry import register_env


@register_env("PickCube-v1")
class PickCube(BaseEnv):
    pass
