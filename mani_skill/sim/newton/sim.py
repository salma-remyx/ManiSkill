from dataclasses import dataclass

import newton
import torch
import warp as wp

from mani_skill.envs.utils.system.backend import parse_backend_device_id
from mani_skill.sim.base_sim import BaseSim, BaseSimConfig
from mani_skill.sim.newton.structs.actor import NewtonActor
from mani_skill.sim.newton.structs.articulation import NewtonArticulation


@dataclass(frozen=True)
class NewtonSimConfig(BaseSimConfig):
    pass


class NewtonSim(BaseSim):
    id: str = "newton"
    cfg: NewtonSimConfig
    actors: dict[str, NewtonActor]
    articulations: dict[str, NewtonArticulation]
    sim_warp_device: wp.Device
    """The warp device that the simulation is running on."""
    render_warp_device: wp.Device
    """The warp device that the rendering is running on."""
    _model: newton.Model
    """The Newton model of the scene."""
    _state_0: newton.State
    """The state of the scene at the current time step."""
    _state_1: newton.State
    """The state of the scene at the next time step."""
    _control: newton.Control
    """The control of the scene."""
    _contacts: newton.Contacts
    """The contacts buffer"""
    _viewer: newton.viewer.ViewerBase | None = None
    """The viewer for the scene."""
    _physics_step_graph: wp.Graph | None = None
    """The CUDA graph of the physics step."""

    def __init__(
        self,
        num_envs: int = 1,
        cfg: NewtonSimConfig | None = None,
        sim_backend: str | None = "newton.mujoco_cpu",
        render_backend: str | None = "newton.warp",
    ):
        if cfg is None:
            cfg = NewtonSimConfig()
        # Determine devices simulation and/or rendering are running on
        sim_device_torch = torch.device("cpu")
        render_device_torch = torch.device("cpu")
        sim_warp_device = None
        render_warp_device = None
        if sim_backend is not None:
            _, sim_backend, sim_device_id = parse_backend_device_id(
                sim_backend, sim_backend=True
            )
            if sim_backend == "mujoco_cpu":
                sim_device_torch = torch.device("cpu")
                sim_warp_device = wp.get_device("cpu")
            elif sim_backend == "mujoco_warp":
                device_str = (
                    f"cuda:{sim_device_id}" if sim_device_id is not None else "cuda"
                )
                sim_device_torch = torch.device(device_str)
                sim_warp_device = wp.get_device(device_str)
            else:
                raise ValueError(f"Invalid simulation backend: {sim_backend}")
            self.sim_warp_device = sim_warp_device
        if render_backend is not None:
            _, render_backend, render_device_id = parse_backend_device_id(
                render_backend, sim_backend=False
            )
            if render_backend == "warp":
                device_str = (
                    f"cuda:{render_device_id}"
                    if render_device_id is not None
                    else "cuda"
                )
                render_device_torch = torch.device(device_str)
                render_warp_device = wp.get_device(device_str)
            else:
                raise ValueError(f"Invalid rendering backend: {render_backend}")
            self.render_warp_device = render_warp_device

        super().__init__(num_envs, cfg, sim_device_torch, render_device_torch)

        self._scene_mb = newton.ModelBuilder()
        root_body = self._scene_mb.add_link(label="__root__")
        root_joint = self._scene_mb.add_joint_fixed(
            -1, root_body, label="__root_joint__"
        )
        self._scene_mb.add_articulation([root_joint])

    def create_actor_builder(self):
        from mani_skill.sim.newton.builders.actor import NewtonActorBuilder

        builder = NewtonActorBuilder()
        builder.sim = self
        return builder

    def create_articulation_builder(self):
        from mani_skill.sim.newton.builders.articulation import (
            NewtonArticulationBuilder,
        )

        builder = NewtonArticulationBuilder()
        builder.sim = self
        return builder

    def create_urdf_loader(self):
        from mani_skill.sim.newton.loaders.urdf import NewtonURDFLoader

        loader = NewtonURDFLoader()
        loader.sim = self
        return loader

    def compile_render_scene(self):
        pass

    def can_render(self):
        pass

    def can_physics(self):
        pass

    def update_render(self):
        pass

    def compile_physical_scene(self):
        self._model = self._scene_mb.finalize(self.sim_warp_device)

        contact_max = 16384
        self._model.rigid_contact_max = contact_max
        self.collision_pipeline = newton.CollisionPipeline(
            self._model,
            reduce_contacts=True,
            rigid_contact_max=contact_max,
            broad_phase="nxn",
        )
        self._solver = newton.solvers.SolverMuJoCo(
            self._model,
            solver="newton",
            integrator="implicitfast",
            iterations=15,
            ls_iterations=100,
            nconmax=contact_max,
            njmax=contact_max * 2,
            cone="elliptic",
            impratio=50.0,
            use_mujoco_contacts=False,
        )

        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()
        newton.eval_fk(
            self._model,
            self._model.joint_q,
            self._model.joint_qd,
            self._state_0,  # type: ignore
        )

        if self.sim_warp_device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.physics_step()
            self._physics_step_graph = capture.graph

    def physics_step(self):
        if self._physics_steps % (self.cfg.sim_freq // self.cfg.control_freq) == 0:
            self.collision_pipeline.collide(self._state_0, self._contacts)
        if self._physics_step_graph is not None:
            wp.capture_launch(self._physics_step_graph)
        else:
            self._physics_step()
        self._physics_steps += 1

    def _physics_step(self):
        self._state_0.clear_forces()
        if self._viewer is not None:
            self._viewer.apply_forces(self._state_0)
        self._solver.step(
            state_in=self._state_0,
            state_out=self._state_1,
            control=self._control,
            contacts=self._contacts,
            dt=self.timestep,
        )

        self._state_0, self._state_1 = self._state_1, self._state_0

    def _gpu_apply_all(self):
        pass

    def _gpu_fetch_all(self):
        pass

    def _gpu_update_articulation_kinematics(self):
        pass
