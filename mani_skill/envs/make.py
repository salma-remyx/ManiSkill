# Provides a typed ManiSkill wrapper around Gymnasium style environment construction.

from mani_skill.envs import BaseEnv
from mani_skill.envs.registry import REGISTERED_ENVS


def make(
    env_id: str,
    *,
    obs_mode: str | None = None,
    reward_mode: str | None = None,
    render_mode: str | None = None,
    num_envs: int = 1,
    physics_backend: str = "auto",
    render_backend: str = "auto",
) -> BaseEnv:
    """Create a registered ManiSkill environment.

    Args:
        env_id: ID of the registered environment to create.
        obs_mode: The mode of observation to use.
        reward_mode: The mode of reward to use.
        render_mode: The mode of rendering to use.
        num_envs: The number of environments to run.
        physics_backend: The backend to use for physics simulation.
        render_backend: The backend to use for rendering.

    Returns:
        The constructed Gymnasium environment.
    """
    return REGISTERED_ENVS[env_id](
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        render_mode=render_mode,
        num_envs=num_envs,
        physics_backend=physics_backend,
        render_backend=render_backend,
    )
