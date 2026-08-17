"""Fast ODE samplers for diffusion policy inference.

Training-free NFE reduction: swap the 100-step DDPM ancestral sampler used at
inference for DEIS, a multistep exponential integrator over the diffusion ODE
that reuses the same epsilon-trained ``noise_pred_net`` untouched. Both
samplers share the ``0 .. num_train_timesteps-1`` timestep grid that
``compute_loss`` samples during training, so the timestep conditioning the
UNet saw at train time stays valid at inference and no retraining is needed.

Adapted from "Fast ODE-based Sampling for Diffusion Models in Around 5 Steps"
(Zhang & Chen, NeurIPS 2023, https://arxiv.org/abs/2312.00094v3), which shows
that the multistep DEIS update -- polynomial interpolation of the score in
log-SNR (lambda) space integrated exactly -- reaches DDPM-quality samples in
around 5 steps. `DEISMultistepScheduler` in diffusers implements that update;
this module only builds it from the training scheduler's noise schedule and
drives the existing denoise loop over it.

DEIS needs the *continuous* alpha/sigma schedule rather than DDPM's discrete
chained update, so the inference scheduler is constructed from the same
``beta_schedule`` and ``num_train_timesteps`` as the training scheduler. It is
not a drop-in per-step replacement: DEIS is multistep (it keeps the last
``solver_order`` model outputs) and it tracks its own step index, so the whole
loop lives here instead of in ``Agent.get_action``.
"""

import torch
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin

__all__ = ["build_deis_scheduler", "deis_denoise"]


def build_deis_scheduler(
    train_scheduler: DDPMScheduler, num_inference_steps: int
) -> DEISMultistepScheduler:
    """Build a DEIS sampler matching the training noise schedule.

    Shares ``beta_schedule`` and ``num_train_timesteps`` with the training
    scheduler so the alpha/sigma curve the solver integrates is the one the
    policy was trained under, and keeps ``prediction_type='epsilon'`` for the
    raw UNet output. DDPM's ``clip_sample=True`` has no DEIS counterpart (DEIS
    exposes thresholding instead); the action range is already enforced by the
    ``[-1, 1]`` action space the policy is trained against.
    """
    config = train_scheduler.config
    scheduler = DEISMultistepScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_schedule=config.beta_schedule,
        prediction_type=config.prediction_type,
    )
    scheduler.set_timesteps(num_inference_steps)  # in place, returns None
    return scheduler


def deis_denoise(
    noise_pred_net,
    scheduler: SchedulerMixin,
    obs_cond,
    shape,
    device,
):
    """Run the reverse diffusion ODE with a fast multistep solver.

    Identical control flow to the DDPM loop in ``Agent.get_action``, but the
    ``num_inference_steps`` timesteps come from the fast scheduler instead of
    the training grid, so the number of ``noise_pred_net`` evaluations (NFE)
    is decoupled from training.

    Returns the fully denoised action sequence of shape ``shape``.
    """
    # DEIS is multistep: it carries the last `solver_order` model outputs and a
    # step index across calls. A policy is queried once per control step, so
    # reset that state (set_timesteps does exactly this) before each rollout.
    scheduler.set_timesteps(len(scheduler.timesteps))

    # initialize action from Gaussian noise, like the DDPM loop
    sample = torch.randn(shape, device=device)

    for k in scheduler.timesteps:
        noise_pred = noise_pred_net(sample=sample, timestep=k, global_cond=obs_cond)
        sample = scheduler.step(
            model_output=noise_pred, timestep=k, sample=sample
        ).prev_sample

    return sample
