from dataclasses import dataclass

import torch

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
        if sim_backend is not None:
            package_name, sim_backend, sim_device_id = parse_backend_device_id(
                sim_backend, sim_backend=True
            )
            if sim_backend == "mujoco_cpu":
                sim_device_torch = torch.device("cpu")
            elif sim_backend == "mujoco_warp":
                device_str = (
                    f"cuda:{sim_device_id}" if sim_device_id is not None else "cuda"
                )
                sim_device_torch = torch.device(device_str)
            else:
                raise ValueError(f"Invalid simulation backend: {sim_backend}")
        if render_backend is not None:
            package_name, render_backend, render_device_id = parse_backend_device_id(
                render_backend, sim_backend=False
            )
            if render_backend == "warp":
                device_str = (
                    f"cuda:{render_device_id}"
                    if render_device_id is not None
                    else "cuda"
                )
                render_device_torch = torch.device(device_str)
            else:
                raise ValueError(f"Invalid rendering backend: {render_backend}")
        super().__init__(num_envs, cfg, sim_device_torch, render_device_torch)

    def create_actor_builder(self):
        pass

    def create_articulation_builder(self):
        pass

    def create_urdf_loader(self):
        pass

    def compile_render_scene(self):
        pass

    def can_render(self):
        pass

    def can_physics(self):
        pass

    def update_render(self):
        pass

    def compile_physical_scene(self):
        pass

    def physics_step(self):
        pass
