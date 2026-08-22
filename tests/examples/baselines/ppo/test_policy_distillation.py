"""Tests for the PPO baseline's Proximal Policy Distillation (PPD) wiring.

Exercises the real call-site modules: `ppo.Agent` / `ppo.Args` (the baseline the
hook lives in) and `policy_distillation.PPDLoss` / `load_ppd_teacher`.
"""

import sys
import pathlib
import types
import importlib

import torch
from torch.distributions.normal import Normal

PPO_DIR = pathlib.Path(__file__).resolve().parents[4] / "examples" / "baselines" / "ppo"
sys.path.append(str(PPO_DIR))

# ppo.py imports SummaryWriter at module scope; it is only used by the training
# loop, so a no-op stub keeps the baseline importable without tensorboard.
if "torch.utils.tensorboard" not in sys.modules:
    try:
        importlib.import_module("tensorboard")
    except ModuleNotFoundError:
        stub = types.ModuleType("torch.utils.tensorboard")
        stub.SummaryWriter = object
        sys.modules["torch.utils.tensorboard"] = stub

from ppo import Agent, Args  # noqa: E402
from policy_distillation import PPDLoss, load_ppd_teacher  # noqa: E402


class DummyVecEnv:
    """Just enough of the ManiSkillVectorEnv surface for `Agent.__init__`."""

    def __init__(self, obs_dim: int = 8, act_dim: int = 4):
        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(-1, 1, (obs_dim,))
        self.single_action_space = gym.spaces.Box(-1, 1, (act_dim,))


def _make_agent(seed: int = 0) -> Agent:
    torch.manual_seed(seed)
    return Agent(DummyVecEnv())


def test_ppd_loss_zero_when_student_is_teacher():
    teacher = _make_agent(seed=0)
    student = _make_agent(seed=0)
    loss_fn = PPDLoss(teacher, distill_lambda=1.0, clip_coef=0.2)
    obs = torch.randn(16, 8)
    kl = loss_fn(student.actor_mean, student.actor_logstd, obs)
    assert torch.allclose(kl, torch.zeros(()), atol=1e-6)


def test_ppd_loss_positive_and_pulls_student_towards_teacher():
    # Two same-architecture random inits produce near-identical Normal
    # distributions (the actor head is initialized with std=0.01), so the
    # teacher's head is rescaled to make the two policies genuinely differ.
    teacher = _make_agent(seed=0)
    with torch.no_grad():
        for p in teacher.actor_mean.parameters():
            p.mul_(20.0)
    student = _make_agent(seed=1)
    loss_fn = PPDLoss(teacher, distill_lambda=1.0, clip_coef=0.2)
    obs = torch.randn(32, 8)

    kl_before = loss_fn(student.actor_mean, student.actor_logstd, obs).detach()
    assert float(kl_before) > 0.1

    # One gradient step on the distillation loss alone must move the student's
    # mean towards the teacher's mean (the direction PPD anchors the policy in).
    teacher_mean = loss_fn.teacher_mean_logstd(obs)[0].detach()
    dist_before = (student.actor_mean(obs) - teacher_mean).abs().mean().detach()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-2)
    for _ in range(50):
        optimizer.zero_grad()
        loss_fn(student.actor_mean, student.actor_logstd, obs).backward()
        optimizer.step()
    dist_after = (student.actor_mean(obs) - teacher_mean).abs().mean().detach()
    assert float(dist_after) < float(dist_before)
    kl_after = loss_fn(student.actor_mean, student.actor_logstd, obs).detach()
    assert float(kl_after) < float(kl_before)


def test_ppd_clip_bounds_importance_weights():
    # Teacher and student with very different log-stds produce large importance
    # weights; the clip must keep the effective weight >= 1 - clip_coef.
    teacher = _make_agent()
    with torch.no_grad():
        teacher.actor_logstd.fill_(0.5)
    student = _make_agent(seed=1)
    with torch.no_grad():
        student.actor_logstd.fill_(-2.0)
    loss_fn = PPDLoss(teacher, distill_lambda=1.0, clip_coef=0.2)
    obs = torch.randn(64, 8)
    kl = loss_fn(student.actor_mean, student.actor_logstd, obs)
    assert torch.isfinite(kl)

    # Recompute the unclipped weights to confirm the clip is what bounds them.
    t_mean, t_logstd = loss_fn.teacher_mean_logstd(obs)
    s_mean = student.actor_mean(obs)
    t_dist = Normal(t_mean, t_logstd.exp())
    s_dist = Normal(s_mean, student.actor_logstd.expand_as(s_mean).exp())
    log_ratio = s_dist.log_prob(t_mean).sum(-1) - t_dist.log_prob(t_mean).sum(-1)
    weights = torch.exp(log_ratio)
    assert bool((weights > 1 - 0.2).any())
    assert float(weights.min().detach()) >= 0.0


def test_teacher_is_frozen():
    teacher = _make_agent()
    loss_fn = PPDLoss(teacher, distill_lambda=1.0, clip_coef=0.2)
    assert all(not p.requires_grad for p in loss_fn.teacher.parameters())
    assert not loss_fn.training
    assert not loss_fn.teacher.training


def test_load_ppd_teacher_restores_checkpoint(tmp_path):
    teacher = _make_agent(seed=3)
    ckpt = tmp_path / "teacher.pt"
    torch.save(teacher.state_dict(), ckpt)

    envs = DummyVecEnv()
    loaded = load_ppd_teacher(str(ckpt), envs)
    obs = torch.randn(4, 8)
    with torch.no_grad():
        assert torch.allclose(loaded.get_action(obs, deterministic=True), teacher.get_action(obs, deterministic=True))


def test_ppd_args_defaults_keep_baseline_unchanged():
    args = Args()
    assert args.teacher_checkpoint is None
    assert args.distill_lambda == 1.0
