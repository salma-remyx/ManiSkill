# Defines the shared configuration and interface for simulation backends.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from mani_skill.sim.core.entities.actor import Actor
from mani_skill.sim.core.entities.articulation import Articulation

if TYPE_CHECKING:
    from mani_skill.sim.core.builders.actor import ActorBuilder

RENDER_MODES = Literal["human", "rgb_array"]
"""The modes of rendering to use that are available."""


@dataclass(frozen=True)
class BaseSimConfig:
    """
    Base configuration dataclass for the simulation backends.
    """

    spacing: float = 2.0
    """Controls the spacing between parallel environments when simulating on GPU in meters.
    Increase this value if you expect objects in one parallel environment to impact objects
    within this spacing distance."""
    sim_freq: int = 240
    """simulation frequency (Hz)."""
    control_freq: int = 60
    """control frequency (Hz). Every control step (e.g. env.step) contains
    (sim_freq / control_freq) physics steps."""


class BaseSim(ABC):
    """
    Base class for all simulation backends.

    A simulation backend consists of primarily a physics engine and a renderer. It is possible
    for a simulation backend to only have one or the other as well activated.

    Args:
        num_envs: The number of environments to simulate.
        cfg: The configuration for the simulation backend.
        sim_device_torch: The torch device that physics engine returns data on. If none,
            this sim object is not performing any physics simulation.
        render_device_torch: The torch device that the renderer returns data on. If none,
            this sim object is not performing any rendering.
    """

    id: str
    """The id of the simulation backend."""
    physics_device: Any
    """The torch device that the physics engine returns data on."""
    render_device: Any
    """The torch device that the renderer returns data on."""
    cfg: BaseSimConfig
    """The configuration for the simulation backend."""
    render_mode: RENDER_MODES | None
    """The mode of rendering to use."""
    num_envs: int
    """The number of environments to simulate."""
    batch_sim_enabled: bool
    """Whether the simulation backend is batched. Usually this refers to a parallelized simulation
    like a GPU simulation."""
    actors: dict[str, Any]
    """The dictionary of backend-specific actors in the simulation backend."""
    articulations: dict[str, Articulation]
    """The dictionary of articulations in the simulation backend."""
    _gpu_sim_initialized: bool
    """whether the GPU simulation has been initialized"""
    _physics_steps: int
    """The number of physics steps taken."""
    _viewer: Any | None
    """The viewer for the scene."""

    def __init__(
        self,
        num_envs: int = 1,
        cfg: BaseSimConfig | None = None,
        render_mode: RENDER_MODES | None = None,
        physics_device: Any = None,
        render_device: Any = None,
    ):
        if physics_device is None:
            physics_device = "cpu"
        if render_device is None:
            render_device = "cpu"
        self.num_envs = num_envs
        self.cfg = cfg or BaseSimConfig()
        self.render_mode = render_mode
        self.physics_device = physics_device
        self.render_device = render_device
        self.batch_sim_enabled = num_envs > 1
        self.actors = dict()
        self.articulations = dict()
        self._gpu_sim_initialized = False
        self._physics_steps = 0
        self._viewer = None

    # ---------------------------------------------------------------------------- #
    # Shared derived properties
    # ---------------------------------------------------------------------------- #
    @cached_property
    def timestep(self) -> float:
        """The physics timestep (dt) of the simulation in seconds."""
        return 1.0 / self.cfg.sim_freq

    @property
    def sim_time(self) -> float:
        """The total simulation time passed in seconds. Equal to physics_steps * timestep."""
        return self._physics_steps * self.timestep

    def remove_actor(self, actor: Actor) -> None:
        """
        Removes an actor from the simulation scene.
        """
        raise NotImplementedError()

    def remove_articulation(self, articulation: Articulation) -> None:
        """
        Removes an articulation from the simulation scene.
        """
        raise NotImplementedError()

    def close(self) -> None:
        """
        Closes the simulation backend.
        """
        raise NotImplementedError()

    def add_actor_builder(
        self,
        builder: ActorBuilder,
        *,
        build_physics: bool = True,
        build_render: bool = True,
    ) -> Any:
        """Compile an actor description into this simulation backend.

        Args:
            builder: Backend-neutral actor description to compile.
            build_physics: Whether to compile collision and physical data.
            build_render: Whether to compile visual data.

        Returns:
            The backend-specific actor bound to the compiled data.
        """
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Code for working with cameras and sensors
    # ---------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------- #
    # Code for lighting
    # ---------------------------------------------------------------------------- #
    @property
    def ambient_light(self) -> tuple[float, float, float]:
        """
        The ambient light of the simulation scene.
        """
        raise NotImplementedError()

    @ambient_light.setter
    def ambient_light(self, color: tuple[float, float, float]) -> None:
        """
        Sets the ambient light of the simulation scene.
        """
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Code for compiling simulator scene for rendering
    # ---------------------------------------------------------------------------- #
    @abstractmethod
    def compile_render_scene(self):
        """
        Compiles the simulation scene for rendering.
        """

    # ---------------------------------------------------------------------------- #
    # Rendering code
    # ---------------------------------------------------------------------------- #
    @abstractmethod
    def can_render(self) -> bool:
        """
        Whether the simulation backend can render.
        """

    def update_sensors(
        self,
        update_sensors: bool = True,
        update_human_render_cameras: bool = True,
        update_viewer_cameras: bool = True,
    ) -> None:
        """
        Updates all sensors such as cameras. ManiSkill further groups sensors into three categories:

        1. Sensors for actual observations to be fed into policies.
        2. Human render cameras for (high-quality) video capture for qualitative review.
        3. Viewer cameras for any GUI based applications.

        Args:
            update_sensors: Whether to update the sensors for actual observations to be fed into
            policies.
            update_human_render_cameras: Whether to update the human render cameras for
            (high-quality) video capture for qualitative review.
            update_viewer_cameras: Whether to update the viewer cameras for any GUI based
            applications.
        """
        raise NotImplementedError()

    def render(self) -> None:
        if self.render_mode == "human":
            self.render_human()
        elif self.render_mode == "rgb_array":
            self.render_rgb_array()

    def render_rgb_array(self) -> Any:
        """
        Renders the simulation scene and returns an RGB array for
        qualitative/visualization purposes.
        """
        raise NotImplementedError()

    def render_human(self) -> Any:
        """
        Renders the simulation scene for human viewing. Usually this should open a GUI.
        """
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Code for compiling simulator scene for physical simulation
    # ---------------------------------------------------------------------------- #
    @abstractmethod
    def compile_physical_scene(self):
        """
        Compiles the simulation scene for physical simulation. Usually necessary to have an
        explicit compilation stage for simulators with GPU parallelization, but some
        simulation backends permit larger changes to the physical scene at runtime.
        """

    # ---------------------------------------------------------------------------- #
    # Physical simulation code
    # ---------------------------------------------------------------------------- #

    @abstractmethod
    def control_step(self) -> None:
        """
        Steps the simulation forward by one control step, which contains (sim_freq / control_freq)
        physics steps.
        """

    @abstractmethod
    def physics_step(self):
        """
        Runs a single physics step at dt = 1 / self.cfg.sim_freq seconds.
        """

    @abstractmethod
    def can_physics(self) -> bool:
        """
        Whether the simulation backend can run physical simulation.
        """
