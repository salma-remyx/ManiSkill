# Defines the Newton simulation backend and its construction entry points.

from dataclasses import dataclass
from typing import Literal

import newton
import warp as wp
from newton.viewer import ViewerBase

from mani_skill.sim.core.base_sim import RENDER_MODES, BaseSim, BaseSimConfig
from mani_skill.sim.core.builders.actor import ActorBuilder
from mani_skill.sim.newton.builders.actor import (
    add_collision_records,
    add_visual_records,
    pose_record_to_transform,
)
from mani_skill.sim.newton.entities.actor import NewtonActor
from mani_skill.utils.logging import logger

VIEWER_BACKENDS = Literal["gl", "rtx", "viser"]


@dataclass(frozen=True)
class NewtonSimConfig(BaseSimConfig):
    """Configure the Newton simulation backend."""

    viewer_backend: VIEWER_BACKENDS = "rtx"


class NewtonSim(BaseSim):
    """The Newton simulation backend."""

    id: str = "newton"
    physics_device: str | None
    render_device: str | None
    actors: dict[str, NewtonActor]
    _viewer: ViewerBase | None
    """The viewer for the Newton scene."""

    def __init__(
        self,
        num_envs: int = 1,
        cfg: NewtonSimConfig | None = None,
        render_mode: RENDER_MODES | None = None,
        physics_device: str | None = None,
        render_device: str | None = None,
    ) -> None:
        super().__init__(
            num_envs,
            cfg or NewtonSimConfig(),
            render_mode,
            physics_device,
            render_device,
        )

        self.scene_mb = newton.ModelBuilder()
        # NOTE (stao): following dummy root is necessary for Newton to work with Mujoco
        root_body = self.scene_mb.add_link(label="__root__")
        root_joint = self.scene_mb.add_joint_fixed(
            -1, root_body, label="__root_joint__"
        )
        self.scene_mb.add_articulation([root_joint])
        self._compiled = False
        self._control_step_graph = None

    def close(self) -> None:
        """
        Closes the Newton simulation backend.
        """
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None
        del self.model
        del self.state
        self._compiled = False
        self._control_step_graph = None

    def add_actor_builder(
        self,
        builder: ActorBuilder,
        *,
        build_physics: bool = True,
        build_render: bool = True,
    ) -> NewtonActor:
        """Compile an actor description into the Newton scene template.

        Args:
            builder: Backend-neutral actor description to compile.
            build_physics: Whether to compile collision and physical data.
            build_render: Whether to compile visual data.

        Returns:
            Newton actor bound to the resulting body.
        """
        actor_model_builder = newton.ModelBuilder()
        # TODO (stao): handle static objects, may need to call add_shape directly since
        # add_shape_box and other builder functions do not have a static option
        if builder.body_type != "static" and build_physics:
            body_index = actor_model_builder.add_body(
                xform=pose_record_to_transform(builder.initial_pose),
                label=builder.name or None,
                is_kinematic=builder.body_type != "dynamic",
            )
        else:
            body_index = -1
        if build_physics:
            add_collision_records(actor_model_builder, body_index, builder)
        if build_render:
            add_visual_records(actor_model_builder, body_index, builder)

        actor = NewtonActor(self, actor_model_builder, builder, body_index)
        self.actors[builder.name] = actor
        return actor

    def _compile_scene(self, device: str | None) -> None:
        """Replicate and finalize the scene template on one device.

        Args:
            device: Warp device on which to create the Newton model and state.
        """
        if self._compiled:
            return

        for actor in self.actors.values():
            if actor.builder.scene_idxs is None:
                self.scene_mb.add_builder(actor.mb)
        final_model_builder = newton.ModelBuilder()

        # replicate the scene model builder num_envs times
        final_model_builder.replicate(
            self.scene_mb,
            world_count=self.num_envs,
            spacing=(self.cfg.spacing, self.cfg.spacing, 0.0),
        )
        for actor in self.actors.values():
            if actor.builder.scene_idxs is not None:
                if tuple(actor.builder.scene_idxs) == (0,):
                    final_model_builder.add_builder(actor.mb)
                else:
                    raise NotImplementedError(
                        "Currently the newton package/backend does not "
                        "support heterogeneous worlds the way SAPIEN does. Currently only support "
                        "homogeneous worlds and (ground) planes in scene 0."
                    )
        self.model = final_model_builder.finalize(device=device)
        if self.render_mode == "human":
            self._setup_viewer()

        bodies_per_world = len(self.scene_mb.body_q)
        for actor in self.actors.values():
            actor.set_body_indices(
                wp.array(
                    [
                        actor.body_index + world_index * bodies_per_world
                        for world_index in range(self.num_envs)
                    ],
                    dtype=wp.int32,
                    device=device,
                )
            )

        self.state = self.model.state()
        if self.can_physics():
            contact_max = 1024
            self.model.rigid_contact_max = contact_max
            self._collision_pipeline = newton.CollisionPipeline(
                self.model,
                reduce_contacts=True,
                rigid_contact_max=contact_max,
                broad_phase="nxn",
            )
            self._solver = newton.solvers.SolverMuJoCo(
                self.model,
                solver="newton",
                integrator="implicitfast",
                iterations=30,
                ls_iterations=10,
                nconmax=contact_max,
                njmax=contact_max * 2,
                cone="elliptic",
                impratio=50.0,
                use_mujoco_cpu=False,
                use_mujoco_contacts=False,
            )
            self._next_state = self.model.state()
            self._control = self.model.control()
            self._contacts = self.model.contacts()
        newton.eval_fk(
            self.model,
            self.model.joint_q,  # pyright: ignore[reportArgumentType]
            self.model.joint_qd,  # pyright: ignore[reportArgumentType]
            self.state,
        )
        if self.physics_device is not None:  # equivalent to self.can_physics()
            if self.physics_device[:4] == "cuda":
                with wp.ScopedCapture() as capture:
                    self.control_step()
                self._control_step_graph = capture.graph

        self._compiled = True

    def compile_render_scene(self) -> None:
        """Finalize the Newton scene on the rendering device."""
        self._compile_scene(self.render_device)

    def can_render(self) -> bool:
        """Return whether the Newton rendering scaffold can render."""
        return self.render_device is not None

    def _setup_viewer(self) -> None:
        assert isinstance(self.cfg, NewtonSimConfig)
        if self.cfg.viewer_backend == "gl":
            from newton.viewer import ViewerGL

            self._viewer = ViewerGL()
        elif self.cfg.viewer_backend == "rtx":
            from newton.viewer import ViewerRTX

            self._viewer = ViewerRTX()
        elif self.cfg.viewer_backend == "viser":
            from newton.viewer import ViewerViser

            self._viewer = ViewerViser()
        else:
            raise ValueError(f"Invalid viewer backend: {self.cfg.viewer_backend}")
        self._viewer.set_model(self.model)

    def render_human(self) -> ViewerBase:
        """Render the Newton scene for human viewing."""
        if self._viewer is None:
            if self.render_mode is None:
                logger.warning(
                    "Render mode is not set, might not support applying forces interactively."
                )
            self._setup_viewer()
        assert self._viewer is not None

        self._viewer.begin_frame(self.sim_time)
        self._viewer.log_state(self.state)
        self._viewer.end_frame()
        return self._viewer

    def compile_physical_scene(self) -> None:
        """Finalize the Newton scene on the physics device."""
        self._compile_scene(self.physics_device)

    def control_step(self) -> None:
        """Step the simulation forward by one control step, which contains (sim_freq / control_freq)
        physics steps."""
        if self._control_step_graph is not None:
            wp.capture_launch(self._control_step_graph)
        else:
            self._control_step()
        self._physics_steps += self.cfg.sim_freq // self.cfg.control_freq

    def _control_step(self):
        self._collision_pipeline.collide(self.state, self._contacts)
        for _ in range(self.cfg.sim_freq // self.cfg.control_freq):
            self.physics_step()

    def physics_step(self):
        self.state.clear_forces()
        if self._viewer is not None:
            self._viewer.apply_forces(self.state)
        self._solver.step(  # pyright: ignore[reportUnknownMemberType]
            state_in=self.state,
            state_out=self._next_state,
            control=self._control,
            contacts=self._contacts,
            dt=self.timestep,
        )

        self.state, self._next_state = self._next_state, self.state

    def can_physics(self) -> bool:
        """Return whether the Newton physics scaffold can simulate."""
        return self.physics_device is not None
