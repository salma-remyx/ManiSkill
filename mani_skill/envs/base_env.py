# Defines the base class and common construction options for ManiSkill envs, based on Gymnasium

from typing import Any, Sequence, cast

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sim.core.base_sim import BaseSim, BaseSimConfig
from mani_skill.sim.core.utils.backend import BackendInfo, parse_sim_and_render_backend


class BaseEnv:
    obs_mode: str
    reward_mode: str
    render_mode: str | None
    num_envs: int = 1
    backend: BackendInfo
    _physics_sim: BaseSim
    _render_sim: BaseSim
    scene: ManiSkillScene
    sim_config: BaseSimConfig

    def __init__(
        self,
        *,
        obs_mode: str = "state",
        reward_mode: str = "normalized_dense",
        render_mode: str | None = None,
        num_envs: int = 1,
        physics_backend: str = "newton.mj_cpu",
        render_backend: str = "newton.warp",
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
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.num_envs = num_envs
        self.backend = parse_sim_and_render_backend(physics_backend, render_backend)
        # TODO (stao): handle sim config overrides
        self.sim_config = self._default_sim_config

    @property
    def _default_sim_config(self) -> BaseSimConfig:
        """Return the default simulation configuration for this environment."""
        return BaseSimConfig()

    # ------------------------------------------------------------
    # Gymnasium style methods for env life-cycle
    # ------------------------------------------------------------

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
        if reconfigure:
            self._reconfigure()
        return None, None

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    # ------------------------------------------------------------
    # Environment management methods
    # ------------------------------------------------------------

    def _reconfigure(self) -> None:
        # TODO (stao): add a sim backend registry?

        if self.backend.physics_backend_package == "newton":
            from mani_skill.sim.newton.sim import NewtonSim, NewtonSimConfig

            self._physics_sim = NewtonSim(
                num_envs=self.num_envs,
                cfg=cast(NewtonSimConfig, self.sim_config),
                physics_device=self.backend.physics_device_id,
                render_device=self.backend.render_device_id,
            )
        else:
            raise ValueError(
                f"Unrecognized physics backend package: {self.backend.physics_backend_package}"
            )

        if self.backend.render_backend_package == "newton":
            from mani_skill.sim.newton.sim import NewtonSim, NewtonSimConfig

            if (
                self.backend.render_backend_package
                == self.backend.physics_backend_package
            ):
                self._render_sim = self._physics_sim
            else:
                from mani_skill.sim.newton.sim import NewtonSim

                self._render_sim = NewtonSim(
                    num_envs=self.num_envs,
                    cfg=cast(NewtonSimConfig, self.sim_config),
                    physics_device=self.backend.physics_device_id,
                    render_device=self.backend.render_device_id,
                )
        else:
            raise ValueError(
                f"Unrecognized render backend package: {self.backend.render_backend_package}"
            )
        self.scene = ManiSkillScene(
            physics_sim=self._physics_sim,
            render_sim=self._render_sim,
        )
        self._load_scene()

    # ------------------------------------------------------------
    # Task-specific methods
    # ------------------------------------------------------------
    def _load_scene(self) -> None:
        pass

    def _initialize_episode(self) -> None:
        pass
