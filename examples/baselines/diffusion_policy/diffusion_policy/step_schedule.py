"""Non-uniform timestep schedules for the Diffusion Policy baseline.

Conventional inference uses every DDPM timestep, i.e. a uniform schedule over
the whole noise range. That spends a large share of the denoising budget on
the final (already nearly clean) timesteps, where the learned noise predictor
is pushed towards memorised demonstration actions and only shrinks the step
size — extra steps there degrade rather than help.

This module provides a parameter-free alternative: concentrate the denoise
budget on the high-noise timesteps and finish with a single terminal step
from a jump timestep down to zero, skipping the dense low-noise tail
entirely. The training side mirrors this with a U-shaped timestep sampler
that draws more samples near both ends of the noise range, so the model sees
the terminal region often enough to predict it reliably in one step.

Adapted from "Dense-Jump Flow Matching with Non-Uniform Time Scheduling for
Robotic Policies: Mitigating Multi-Step Inference Degradation"
(https://arxiv.org/abs/2509.13574), which proposes this dense-early /
single-terminal-jump integration schedule and Beta(alpha, alpha) timestep
sampling for flow-matching policies. Here both are re-expressed for the
baseline's epsilon-prediction DDPM policy: timesteps run high-noise -> zero
(rather than flow time 0 -> 1), the velocity field is the DDPM posterior
step, and the Beta sampler is used to draw training timesteps.
"""

import numpy as np
import torch


def dense_jump_timesteps(
    num_train_timesteps: int,
    num_inference_timesteps: int,
    jump_fraction: float = 0.5,
    device=None,
) -> torch.Tensor:
    """Build a dense-early / single-terminal-jump inference schedule.

    The returned timesteps are in DDPM order (strictly decreasing) and
    uniform in timestep index, so the first ``num_inference_timesteps - 1``
    of them can each be passed straight to ``DDPMScheduler.step``. The last
    one is ``t_jump`` and is meant for :func:`terminal_jump` instead: it
    extrapolates to the clean sample in a single update, replacing all the
    small low-noise steps a uniform schedule would take over
    ``(0, t_jump)``.

    Args:
        num_train_timesteps: Number of timesteps the scheduler was trained
            with (``DDPMScheduler.config.num_train_timesteps``).
        num_inference_timesteps: Total number of denoise evaluations to
            spend. Must be >= 2 (one dense step plus the terminal jump).
        jump_fraction: Position of the jump timestep as a fraction of the
            trained timestep range, in (0, 1). 0.5 spends half the range on
            dense high-noise steps and jumps over the other half.
        device: Optional device to place the returned tensor on.

    Returns:
        LongTensor of shape ``(num_inference_timesteps,)`` whose final entry
        is ``t_jump``.
    """
    if num_inference_timesteps < 2:
        raise ValueError(
            "num_inference_timesteps must be >= 2 (dense steps + terminal jump)"
        )
    if not 0.0 < jump_fraction < 1.0:
        raise ValueError(f"jump_fraction must be in (0, 1), got {jump_fraction}")
    if num_train_timesteps < 2:
        raise ValueError(f"num_train_timesteps must be >= 2, got {num_train_timesteps}")

    t_jump = max(1, int(round(jump_fraction * (num_train_timesteps - 1))))

    # num_inference_timesteps evenly spaced evaluations over the high-noise
    # interval [t_jump, num_train_timesteps); the final one (at t_jump) is
    # consumed by the terminal jump rather than by scheduler.step.
    dense = np.linspace(
        num_train_timesteps - 1, t_jump, num_inference_timesteps, endpoint=True
    ).round()
    return torch.tensor(dense, dtype=torch.long, device=device)


def terminal_jump(scheduler, sample, noise_pred, timestep):
    """Extrapolate from ``timestep`` to the clean sample in a single update.

    This is the epsilon-parameterised one-step estimate of the denoised
    action, computed from the scheduler's own cumulative alpha constants (the
    same quantities ``DDPMScheduler.step`` uses internally, including its
    sample clipping). Used as the last update of a dense-jump schedule it
    spans ``(0, timestep)`` in one go, which is what lets the schedule skip
    the low-noise region instead of walking through it.

    Args:
        scheduler: The trained ``DDPMScheduler`` (epsilon prediction).
        sample: Noisy action sequence at ``timestep``.
        noise_pred: Predicted noise for ``sample`` at ``timestep``.
        timestep: The timestep ``sample`` is currently at; also the timestep
            the noise prediction was conditioned on.

    Returns:
        The estimated clean action sequence, clipped to [-1, 1] when the
        scheduler is configured to clip samples.
    """
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1.0 - alpha_prod_t
    pred_original_sample = (
        sample - beta_prod_t ** 0.5 * noise_pred
    ) / alpha_prod_t ** 0.5
    if scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)
    return pred_original_sample


def beta_timesteps(
    batch_size: int,
    num_train_timesteps: int,
    alpha: float = 0.2,
    device=None,
) -> torch.Tensor:
    """Sample training timesteps from a symmetric U-shaped Beta distribution.

    ``t ~ Beta(alpha, alpha)`` with ``alpha < 1`` concentrates probability
    mass near 0 and 1 and thins the mid-range, so the noise-prediction
    network is supervised more often on the almost-clean and almost-pure-noise
    regimes (the latter being what the terminal jump relies on) and less often
    on the mid-noise regime where it memorises individual demonstration
    actions. Sampled values are rescaled onto the trained timestep range and
    rounded.

    Args:
        batch_size: Number of timesteps to draw.
        num_train_timesteps: Number of timesteps the scheduler was trained
            with.
        alpha: Beta shape parameter; must be > 0. Smaller values give a more
            pronounced U-shape (the reference uses 0.2).
        device: Optional device to place the returned tensor on.

    Returns:
        LongTensor of shape ``(batch_size,)`` with values in
        ``[0, num_train_timesteps)``.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    # torch.distributions draws from the global torch RNG stream, so the
    # seeding already done by the training script governs reproducibility.
    samples = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)
    timesteps = (samples * num_train_timesteps).long().clamp(
        0, num_train_timesteps - 1
    )
    return timesteps
