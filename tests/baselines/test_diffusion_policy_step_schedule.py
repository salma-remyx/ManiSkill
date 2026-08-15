"""Tests for the dense-early / terminal-jump timestep schedule wired into the
Diffusion Policy baseline's Agent (examples/baselines/diffusion_policy/train.py).

These exercise the integration through the baseline's own code path: they
build the real ``Agent`` (with a stubbed observation/action space), run its
``compute_loss`` and ``get_action`` with ``dense_jump`` toggled on and off,
and check the schedule behaviour both via the Agent and via the underlying
module directly. Everything here runs on CPU with a tiny UNet.
"""

import pathlib
import sys

import numpy as np
import pytest
import torch

DIFFUSION_POLICY_DIR = (
    pathlib.Path(__file__, "..", "..", "..", "examples", "baselines", "diffusion_policy")
    .resolve()
)


def _import(name):
    """Import a module from examples/baselines/diffusion_policy."""
    import importlib

    sys.path.insert(0, str(DIFFUSION_POLICY_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(DIFFUSION_POLICY_DIR))


step_schedule = _import("diffusion_policy.step_schedule")
beta_timesteps = step_schedule.beta_timesteps
dense_jump_timesteps = step_schedule.dense_jump_timesteps


def _agent_classes():
    """Import the baseline's Agent and DDPMScheduler, skipping if the heavy
    training dependencies of examples/baselines/diffusion_policy are absent."""
    diffusers = pytest.importorskip("diffusers")
    try:
        return _import("train"), diffusers
    except ModuleNotFoundError as e:  # tensorboard and other training-only deps
        pytest.skip(f"diffusion policy training dependencies unavailable: {e}")


class _Box:
    def __init__(self, shape):
        self.shape = shape
        self.high = np.ones(shape, dtype=np.float32)
        self.low = -np.ones(shape, dtype=np.float32)


class _Spaces:
    def __init__(self, obs_dim, act_dim, obs_horizon):
        self.single_observation_space = _Box((obs_horizon, obs_dim))
        self.single_action_space = _Box((act_dim,))


class _Env:
    def __init__(self, obs_dim, act_dim, obs_horizon):
        self.single_observation_space = _Spaces(
            obs_dim, act_dim, obs_horizon
        ).single_observation_space
        self.single_action_space = _Spaces(obs_dim, act_dim, obs_horizon).single_action_space


class _Args:
    obs_horizon = 2
    act_horizon = 4
    pred_horizon = 8
    diffusion_step_embed_dim = 16
    unet_dims = [8]
    n_groups = 8
    dense_jump = False
    num_inference_timesteps = 6
    jump_fraction = 0.5
    time_beta_alpha = 0.2


def _make_agent(dense_jump, num_inference_timesteps=6, jump_fraction=0.5):
    dp_train, _ = _agent_classes()
    args = _Args()
    args.dense_jump = dense_jump
    args.num_inference_timesteps = num_inference_timesteps
    args.jump_fraction = jump_fraction
    args.unet_dims = [8]
    env = _Env(obs_dim=5, act_dim=3, obs_horizon=args.obs_horizon)
    agent = dp_train.Agent(env, args)
    agent.eval()
    return agent


def test_dense_jump_schedule_spends_budget_on_high_noise_timesteps():
    ts = dense_jump_timesteps(100, 6, 0.5)
    assert ts.shape == (6,)
    assert ts.dtype == torch.long
    # strictly decreasing, starting at the highest trained timestep
    assert ts[0].item() == 99
    assert (ts[:-1] > ts[1:]).all()
    # the last (terminal) timestep is the jump point t_jump = 0.5 * 99
    assert ts[-1].item() == 50
    # every dense step stays in the high-noise half, none lands in (0, t_jump)
    assert (ts >= 50).all()


def test_dense_jump_schedule_budget_one_per_evaluation():
    # one evaluation per element of the returned schedule, not per trained timestep
    assert dense_jump_timesteps(100, 4, 0.5).tolist() == [99, 83, 66, 50]
    assert dense_jump_timesteps(100, 2, 0.5).tolist() == [99, 50]


def test_dense_jump_schedule_rejects_bad_config():
    with pytest.raises(ValueError):
        dense_jump_timesteps(100, 1, 0.5)
    with pytest.raises(ValueError):
        dense_jump_timesteps(100, 6, 0.0)
    with pytest.raises(ValueError):
        dense_jump_timesteps(100, 6, 1.0)
    with pytest.raises(ValueError):
        dense_jump_timesteps(1, 6, 0.5)


def test_beta_timesteps_are_u_shaped():
    torch.manual_seed(0)
    ts = beta_timesteps(200_000, 100, alpha=0.2)
    assert ts.shape == (200_000,)
    assert ts.dtype == torch.long
    assert int(ts.min()) >= 0 and int(ts.max()) < 100
    counts = torch.bincount(ts, minlength=100).float()
    # mass concentrated near both ends of the timestep range, thinned mid-range
    ends = float(counts[:5].sum() + counts[-5:].sum())
    middle = float(counts[45:55].sum())
    assert ends > 15 * middle


def test_beta_timesteps_rejects_bad_alpha():
    with pytest.raises(ValueError):
        beta_timesteps(4, 100, alpha=0.0)


def test_agent_get_action_runs_dense_jump_schedule():
    agent = _make_agent(dense_jump=True, num_inference_timesteps=4)
    torch.manual_seed(0)
    obs = torch.randn(2, agent.obs_horizon, 5)
    with torch.no_grad():
        actions = agent.get_action(obs)
    assert actions.shape == (2, agent.act_horizon, agent.act_dim)
    # the schedule is built once in the constructor and drives the denoise loop
    assert agent.step_schedule is not None
    assert len(agent.step_schedule) == 4
    assert float(actions.abs().max()) <= 1.0  # clip_sample=True


def test_agent_get_action_uniform_when_flag_off():
    agent = _make_agent(dense_jump=False)
    assert agent.step_schedule is None
    torch.manual_seed(0)
    obs = torch.randn(2, agent.obs_horizon, 5)
    with torch.no_grad():
        actions = agent.get_action(obs)
    assert actions.shape == (2, agent.act_horizon, agent.act_dim)


def test_agent_compute_loss_uses_u_shaped_timesteps():
    agent = _make_agent(dense_jump=True)
    # compute_loss reads the module-level `device` that __main__ normally sets
    dp_train, _ = _agent_classes()
    dp_train.device = torch.device("cpu")
    obs = torch.randn(8, agent.obs_horizon, 5)
    act = torch.rand(8, agent.pred_horizon, agent.act_dim) * 2 - 1
    loss = agent.compute_loss(obs, act)
    assert torch.isfinite(loss)


def test_terminal_jump_recovers_clean_sample():
    _, diffusers = _agent_classes()
    terminal_jump = step_schedule.terminal_jump
    scheduler = diffusers.DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    torch.manual_seed(0)
    clean = torch.rand(4, 8, 3) * 2 - 1
    noise = torch.randn(4, 8, 3)
    t = torch.tensor(50)
    noisy = scheduler.add_noise(clean, noise, t)
    # perfect noise prediction should recover the clean actions exactly (up to clipping)
    recovered = terminal_jump(scheduler, noisy, noise, t)
    assert torch.allclose(recovered, clean.clamp(-1, 1), atol=1e-5)


def test_terminal_jump_is_bounded_by_the_clipping_config():
    torch.manual_seed(0)
    _, diffusers = _agent_classes()
    scheduler = diffusers.DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    terminal_jump = step_schedule.terminal_jump
    huge_noise = torch.full((4, 8, 3), 50.0)
    sample = torch.zeros(4, 8, 3)
    out = terminal_jump(scheduler, sample, huge_noise, torch.tensor(10))
    assert float(out.abs().max()) <= 1.0
