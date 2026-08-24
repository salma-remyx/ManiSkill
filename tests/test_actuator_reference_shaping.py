import numpy as np
import pytest
import torch

from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.actuator_reference_shaping import (
    ActuatorReferenceShaper,
    ReferenceModelConfig,
)


def _shaper(**kwargs):
    cfg = ReferenceModelConfig.from_pd_gains(
        stiffness=1e3, damping=1e2, control_dt=1 / 30
    )
    kwargs.setdefault("initial_qpos", torch.zeros(6))
    return ActuatorReferenceShaper(num_joints=6, config=cfg, **kwargs)


def test_reference_model_from_pd_gains():
    # the SO100 arm is configured with stiffness=1e3, damping=1e2
    cfg = ReferenceModelConfig.from_pd_gains(1e3, 1e2, control_dt=1 / 30)
    assert cfg.natural_frequency == pytest.approx(31.62, abs=0.01)
    assert cfg.damping_ratio == pytest.approx(1.58, abs=0.01)
    assert cfg.control_dt == pytest.approx(1 / 30)


def test_reference_dynamics_matches_matrix_exponential():
    """the shaped reference must follow the trained-on second-order dynamics"""
    rng = np.random.default_rng(0)
    for zeta in [1.58, 1.0, 0.5, 2.3]:
        wn = 31.6
        dt = 1 / 30
        A = torch.tensor([[0.0, 1.0], [-wn * wn, -2 * zeta * wn]])
        phi = torch.matrix_exp(A * dt)
        cfg = ReferenceModelConfig(wn, zeta, dt)
        s = ActuatorReferenceShaper(
            num_joints=1, config=cfg, feedback_gain=0.0, initial_qpos=[0.0]
        )
        # start from a random state and advance one step under a held target
        q0 = torch.tensor([rng.uniform(-1, 1)])
        v0 = torch.tensor([rng.uniform(-1, 1)])
        s.reset(q0)
        s._ref_qvel = v0
        s.compute_command(torch.tensor([0.5]), q0)
        expected_q, expected_v = phi @ torch.tensor([q0.item() - 0.5, v0.item()])
        assert s.reference_qpos.item() - 0.5 == pytest.approx(
            expected_q.item(), abs=1e-3
        )
        assert s.reference_qvel.item() == pytest.approx(
            expected_v.item(), abs=1e-2
        )


def test_shaping_reduces_tracking_error_under_actuator_lag():
    """a loaded servo tracks the ideal reference trajectory better with
    shaping than without. the servo is modeled with a proportional internal
    loop and a constant gravity-like load, which leaves a steady-state error
    the shaper's feedback term drives down. error is measured against the
    reference model's trajectory (the dynamics the policy was trained on),
    since shaping intentionally slows the transient to match that reference."""
    n_steps = 300
    dt = 1 / 30
    target = torch.zeros(6)
    target[:3] = torch.tensor([0.8, -0.5, 0.3])

    def servo(q, cmd, kp=3.0, load=0.30):
        u = kp * (cmd - q) - load
        return q + u * dt

    # both arms are judged against the same ideal reference trajectory
    ref = _shaper(feedback_gain=0.0)
    q_unshaped, q_shaped = torch.zeros(6), torch.zeros(6)
    shaped = _shaper(feedback_gain=1.0)
    unshaped_errs, shaped_errs = [], []
    for _ in range(n_steps):
        ref.compute_command(target, q_unshaped)
        q_unshaped = servo(q_unshaped, target)
        unshaped_errs.append((ref.reference_qpos - q_unshaped).abs().max())

        cmd = shaped.compute_command(target, q_shaped)[0]
        q_shaped = servo(q_shaped, cmd)
        shaped_errs.append((shaped.reference_qpos - q_shaped).abs().max())

    assert max(shaped_errs) < max(unshaped_errs)
    assert torch.stack(shaped_errs).mean() < torch.stack(unshaped_errs).mean()
    assert shaped_errs[-1] < unshaped_errs[-1]
    # steady state error scales as load / (kp * (1 + feedback_gain)), so a
    # unity feedback gain roughly halves it
    assert unshaped_errs[-1] > 1.5 * shaped_errs[-1]


def test_feedback_disabled_is_feedforward_only():
    """feedback_gain=0 commands the reference trajectory itself"""
    s = _shaper(feedback_gain=0.0)
    cmd = s.compute_command(torch.zeros(6), torch.full((6,), 0.5))
    assert torch.allclose(cmd[0], s.reference_qpos, atol=1e-6)
    # and is independent of where the real joint actually is
    cmd2 = s.compute_command(torch.zeros(6), torch.full((6,), -0.5))
    assert torch.allclose(cmd[0], cmd2[0], atol=1e-6)


def test_qpos_limits_clamp_commands():
    limits = torch.stack(
        [torch.full((6,), -0.2), torch.full((6,), 0.2)]
    )
    s = _shaper(feedback_gain=5.0, qpos_limits=limits)
    # large tracking error would otherwise command far past the limit
    cmd = s.compute_command(torch.zeros(6), torch.full((6,), 5.0))
    assert cmd.abs().max() <= 0.2 + 1e-6


def test_reset_reseeds_reference_state():
    s = _shaper(feedback_gain=0.0)
    s.compute_command(torch.ones(6), torch.zeros(6))
    assert s.reference_qpos.abs().max() > 0
    s.reset(torch.full((6,), 0.25))
    assert torch.allclose(s.reference_qpos, torch.full((6,), 0.25))
    assert torch.allclose(s.reference_qvel, torch.zeros(6))
    # the first command after reset restarts from the reset state rather than
    # continuing the previous episode's reference trajectory
    cmd = s.compute_command(torch.full((6,), 0.25), torch.full((6,), 0.25))
    assert cmd[0].min() > 0.2


def test_validation_errors():
    with pytest.raises(ValueError):
        ActuatorReferenceShaper(num_joints=0)
    with pytest.raises(ValueError):
        ReferenceModelConfig(natural_frequency=0.0)
    s = _shaper(feedback_gain=0.0)
    with pytest.raises(ValueError):
        s.compute_command(torch.zeros(3), torch.zeros(3))


class _Robot:
    def __init__(self, qpos):
        self.qpos = qpos

    def set_qpos(self, qpos):
        self.qpos = qpos


class _Articulation:
    def __init__(self, drive_targets):
        self.drive_targets = drive_targets


class _Controller:
    sets_target_qpos = True
    sets_target_qvel = False

    def __init__(self, articulation):
        self.articulation = articulation

    def reset(self):
        pass


class _SimAgent:
    def __init__(self, drive_targets):
        articulation = _Articulation(drive_targets)
        self.controller = _Controller(articulation)
        self.robot = _Robot(torch.zeros(1, 6))

    def set_action(self, action):
        self.action = action


class _BaseSimEnv:
    def __init__(self, agent):
        self.agent = agent


class _SimEnvHandle:
    """Stands in for the wrapped sim env; Sim2RealEnv reads .unwrapped off it."""

    def __init__(self, base):
        self.unwrapped = base


class _FakeRealAgent:
    """Minimal stand-in for a BaseRealAgent: records what it is commanded."""

    def __init__(self, sim_agent):
        self.sent = []
        self.controller = sim_agent.controller
        self._sim_agent = sim_agent

    def set_target_qpos(self, qpos):
        self.sent.append(qpos.clone())

    def set_target_qvel(self, qvel):
        raise AssertionError("velocity targets are not part of this path")

    def reset(self, qpos):
        self.robot = _Robot(qpos)


def _make_env(agent):
    """Build a Sim2RealEnv carrying just what _step_action and reset read."""
    env = Sim2RealEnv.__new__(Sim2RealEnv)
    env.agent = agent
    env.actuator_shaper = None
    env.last_control_time = None
    env.control_dt = 0.0
    env._orig_single_action_space = type("S", (), {"shape": torch.Size([6])})()
    env._elapsed_steps = torch.zeros((1,), dtype=torch.int32)
    env._handle_wrappers = False
    env.sim_env = _SimEnvHandle(_BaseSimEnv(agent._sim_agent))
    return env


def test_sim2real_env_wiring_shapes_drive_targets():
    """Sim2RealEnv._step_action must route sim drive targets through the shaper."""
    drive_targets = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    sim_agent = _SimAgent(drive_targets)
    agent = _FakeRealAgent(sim_agent)
    agent.robot = _Robot(torch.zeros(1, 6))
    env = _make_env(agent)

    env._step_action(torch.zeros(1, 6))
    assert len(agent.sent) == 1
    # without a shaper the raw sim drive targets reach the robot unchanged
    assert torch.allclose(agent.sent[0], drive_targets)

    # with a shaper wired in the command must differ from the raw target
    env.actuator_shaper = _shaper(feedback_gain=1.0)
    env.actuator_shaper.reset(agent.robot.qpos)
    env.last_control_time = None
    agent.robot = _Robot(torch.full((1, 6), 0.05))
    env._step_action(torch.zeros(1, 6))
    assert len(agent.sent) == 2
    shaped_cmd = agent.sent[1]
    assert not torch.allclose(shaped_cmd, drive_targets)
    # feedback pushes the command past the raw target because the real joint
    # lags the reference trajectory
    assert (shaped_cmd - drive_targets).abs().max() > 1e-3


def test_sim2real_env_reset_reseeds_shaper():
    sim_agent = _SimAgent(torch.zeros(1, 6))
    agent = _FakeRealAgent(sim_agent)
    env = _make_env(agent)
    env.actuator_shaper = _shaper(feedback_gain=0.0)

    # advance the reference so it sits far from the reset state
    for _ in range(3):
        env.actuator_shaper.compute_command(torch.ones(6), torch.zeros(6))
    assert env.actuator_shaper.reference_qpos.abs().max() > 0.5

    class _EnvWithRealStepReset:
        def reset(self, seed=None, options=None):
            return None, {}

    env._env_with_real_step_reset = _EnvWithRealStepReset()
    env.real_reset_function = lambda env_self, seed=None, options=None: None
    agent.reset(torch.full((1, 6), 0.25))

    env.reset()
    assert torch.allclose(
        env.actuator_shaper.reference_qpos, torch.full((6,), 0.25)
    )
