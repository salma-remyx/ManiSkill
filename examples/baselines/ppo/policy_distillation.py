"""
Proximal Policy Distillation (PPD) support for the CleanRL-style PPO baseline.

Adapted from "Proximal Policy Distillation" (Spigler, TMLR 2025,
https://arxiv.org/abs/2407.15134). PPD fuses student-driven distillation with
PPO: the student keeps interacting with the environment and optimizing the usual
clipped PPO surrogate on its own rewards, while an additional lambda-weighted
KL(teacher || student) term pulls the student towards a frozen teacher policy.
Unlike student-distill / teacher-distill, the critic is NOT distilled --- it is
trained from scratch on the student's own returns, exactly as in PPO.

The teacher is another `Agent` with the same architecture, loaded from a
checkpoint produced by a previous `ppo.py` run and kept frozen. Because both
share the `get_action_and_value` interface, the KL term reduces to reusing the
rollout buffer's observations and querying the teacher's Normal distribution
alongside the student's.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class PPDLoss(nn.Module):
    """Clipped KL(teacher || student) distillation loss of Proximal Policy Distillation.

    The paper's Eq. 2 subtracts ``lambda * KL(pi_teacher || pi_theta)`` from the
    PPO surrogate, with the KL importance weights clipped from below by
    ``1 - eps`` (the same epsilon as the PPO clipping coefficient). Clipping is
    only strictly needed for reverse-KL, but the paper keeps it for generality,
    so the forward-KL implementation here applies the same clip.

    Registers `teacher` parameters with `requires_grad_(False)` and puts the
    module in eval mode; it is never optimized.
    """

    def __init__(
        self,
        teacher: nn.Module,
        distill_lambda: float = 1.0,
        clip_coef: float = 0.2,
    ):
        super().__init__()
        self.distill_lambda = distill_lambda
        self.clip_coef = clip_coef
        self.teacher = teacher
        # The teacher is never trained and has no dropout/batchnorm of its own;
        # eval mode + frozen params keeps it a fixed reference distribution.
        self.teacher.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def teacher_mean_logstd(self, obs: torch.Tensor):
        """Return the frozen teacher's Normal distribution over actions.

        Mirrors `Agent.get_action_and_value`: same `actor_mean` /
        `actor_logstd` modules, so a teacher is any checkpoint saved from this
        same baseline (possibly with different hidden sizes).
        """
        action_mean = self.teacher.actor_mean(obs)
        action_logstd = self.teacher.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def forward(
        self,
        student_actor_mean: nn.Module,
        student_actor_logstd: nn.Parameter,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the (unweighted-by-lambda) clipped KL distillation loss.

        Args:
            student_actor_mean: the student `Agent.actor_mean` module.
            student_actor_logstd: the student `Agent.actor_logstd` parameter.
            obs: the minibatch of observations the student was also evaluated
                on, so both distributions see identical inputs.

        Returns:
            Scalar KL loss with the ``1 - eps`` importance-weight clip applied.
            Multiply by ``-distill_lambda`` when adding to the PPO loss.
        """
        t_mean, t_logstd = self.teacher_mean_logstd(obs)
        s_mean = student_actor_mean(obs)
        s_logstd = student_actor_logstd.expand_as(s_mean)
        teacher_dist = Normal(t_mean, torch.exp(t_logstd))
        student_dist = Normal(s_mean, torch.exp(s_logstd))
        # Per-dimension KL(teacher || student), summed over the action dims to
        # match the per-sample log_prob convention in `get_action_and_value`.
        kl = torch.distributions.kl_divergence(teacher_dist, student_dist).sum(-1)
        # Clip the importance weights at 1 - eps as per Eq. 2. clamp(x, min=c)
        # IS max(x, c): the gradient flows through the max wherever it is
        # active and is zeroed where the clip binds.
        log_ratio = (
            student_dist.log_prob(t_mean).sum(-1)
            - teacher_dist.log_prob(t_mean).sum(-1)
        )
        weight = torch.clamp(torch.exp(log_ratio), min=1 - self.clip_coef)
        return (kl * weight).mean()


def load_ppd_teacher(
    checkpoint: str,
    envs,
    device: Optional[torch.device] = None,
):
    """Build an `Agent` from the enclosing baseline and load a frozen teacher.

    Imported lazily from `ppo` inside the baseline script so this module stays
    importable without a running training loop (and without torch tensorboard
    installed), which keeps it unit-testable.
    """
    from ppo import Agent

    teacher = Agent(envs)
    state_dict = torch.load(checkpoint, map_location="cpu")
    # A teacher trained with different hidden sizes (or by an older script)
    # carries keys this agent does not, and may miss keys it does. Drop the
    # former and let the latter keep their fresh initialization; only the
    # observation/action-facing shapes have to match for the KL term to be
    # well-defined.
    state_dict = {k: v for k, v in state_dict.items() if k in teacher.state_dict()}
    incompatible = teacher.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"PPD teacher: keys not in checkpoint (left at init): {incompatible.missing_keys}")
    if device is not None:
        teacher = teacher.to(device)
    return teacher
