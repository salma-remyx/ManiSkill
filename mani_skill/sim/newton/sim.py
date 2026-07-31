from mani_skill.sim.core.base_sim import BaseSim, BaseSimConfig


class NewtonSimConfig(BaseSimConfig):
    pass


class NewtonSim(BaseSim):
    id: str = "newton"
    physics_device: str
    render_device: str

    def __init__(
        self,
        num_envs: int = 1,
        cfg: NewtonSimConfig | None = None,
        physics_device: str | None = None,
        render_device: str | None = None,
    ):
        super().__init__(num_envs, cfg, physics_device, render_device)
