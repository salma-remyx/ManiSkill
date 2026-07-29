import torch

from mani_skill.sim.core.base_sim import BaseSim, BaseSimConfig


class NewtonSimConfig(BaseSimConfig):
    pass


class NewtonSim(BaseSim):
    def __init__(
        self,
        num_envs: int = 1,
        cfg: NewtonSimConfig | None = None,
        sim_device_torch: torch.device | None = None,
        render_device_torch: torch.device | None = None,
    ):
        super().__init__(num_envs, cfg, sim_device_torch, render_device_torch)
