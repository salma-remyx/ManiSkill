# Defines the base class and common construction options for ManiSkill envs, based on Gymnasium

from typing import Any, Sequence

from mani_skill.sim.core.base_sim import BaseSimConfig


class BaseEnv:
    def __init__(
        self,
        *,
        obs_mode: str | None = None,
        reward_mode: str | None = None,
        render_mode: str | None = None,
        num_envs: int = 1,
        physics_backend: str = "auto",
        render_backend: str = "auto",
    ):
        """Initialize a ManiSkill environment.

        Args:
            obs_mode: The mode of observation to use.
            reward_mode: The mode of reward to use.
            render_mode: The mode of rendering to use.
            num_envs: The number of environments to run.
            physics_backend: The backend to use for physics simulation.
            render_backend: The backend to use for rendering.
        """
        super().__init__()
        self._init_kwargs = {
            "obs_mode": obs_mode,
            "reward_mode": reward_mode,
            "render_mode": render_mode,
            "num_envs": num_envs,
            "physics_backend": physics_backend,
            "render_backend": render_backend,
        }

    @property
    def _default_sim_config(self) -> BaseSimConfig:
        """Return the default simulation configuration for this environment."""
        return BaseSimConfig()

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any]:
        return None, None, None, None, None

    def reset(
        self,
        *,
        seed: int | Sequence[int] | None = None,
        reconfigure: bool = False,
        env_idx: Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """
        Resets the environment to an initial state. Initial state is defined by the task's
        initialize_episode function.

        Args:
            seed: The seed for the environment. If a single integer is provided, it will be used
                to seed all parallel environment RNG sequences. If a sequence/list of integers is
                provided, each parallel environment's RNG sequence will be seeded with a different
                integer. Default is None, in which we use the seed 2026 to seed all environment's
                upon initial creation of the environment.
            reconfigure: Whether to reconfigure the environment. Reconfiguration essentially means
                the simulation is deleted, and a new one is created permitting scene recompilation.
                Useful if you want to randomize assets.
            env_idx: The indices of the environment to reset. If None, all environments are reset.
            options: Additional options to pass to the environment's reset function. Used by custom
                environments to pass in additional information.
        """
        return None, None

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
