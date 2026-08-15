"""
Resolution-stratified feature caching for the diffusion policy denoising loop.

Adapted from "Clockwork Diffusion: Efficient Generation With Model-Step
Distillation" (arXiv:2312.08128). The paper's observation is that a diffusion
UNet does not need to recompute every block at every denoising step: blocks
operating on high-resolution feature maps are sensitive to small perturbations
and must be refreshed often, while the low-resolution blocks set the coarse
layout and can be refreshed on a slower clock. Staggering the refresh clocks
per resolution level buys inference speed at no retraining cost.

The paper applies this to 2D text-to-image UNets. Here the same
resolution-stratified cache is applied to the 1D action UNet
(`ConditionalUnet1D`) used by this baseline, where "resolution" is temporal
(horizon 16 -> 8 -> 4 across the down path) rather than spatial. Everything is
inference-time only: weights, the training objective, and the
`scheduler.step()` contract are untouched.

The paper's second contribution -- distilling the high-res sub-network into a
cheaper student -- is intentionally out of scope; this module is the
retrain-free caching half only.
"""

from typing import Callable, List, Optional

import torch


def stage_timesteps(timesteps, refresh_intervals: List[int]) -> List[List[int]]:
    """Split the denoising schedule into per-stage refresh points.

    For each stage i, returns the indices of `timesteps` at which that stage's
    cached activations must be recomputed. A stage with interval 1 recomputes
    at every step (i.e. it is never cached); a stage with interval k recomputes
    every k-th step and reuses its cache in between.

    Args:
        timesteps: the iterable handed to the denoising loop
            (`noise_scheduler.timesteps`).
        refresh_intervals: one positive interval per stage, coarsest (lowest
            resolution / slowest clock) first.
    """
    timesteps = list(timesteps)
    if any(k < 1 for k in refresh_intervals):
        raise ValueError(f"refresh intervals must be >= 1, got {refresh_intervals}")
    return [
        [j for j in range(len(timesteps)) if j % k == 0] for k in refresh_intervals
    ]


class ClockworkUnet1D(torch.nn.Module):
    """Wraps a `ConditionalUnet1D` so low-res stages refresh on slower clocks.

    Refresh intervals are given in down-path order, coarse-to-fine: the last
    down stage (and the mid blocks it feeds) sit on the lowest-resolution
    feature maps (horizon 4 for the default horizon 16), the first down stage
    on the highest (horizon 16). A stage bound to interval 1 is recomputed
    every step and is effectively uncached, so all-ones intervals reproduce the
    vanilla forward exactly.

    Only the encoder-side activations are reused; each decoder stage consumes
    the skip connection produced alongside its own refresh, so encoder and
    decoder clocks stay paired by depth.
    """

    def __init__(self, unet, refresh_intervals: Optional[List[int]] = None):
        super().__init__()
        self.unet = unet
        num_stages = len(unet.down_modules)
        if refresh_intervals is None:
            refresh_intervals = [1] * num_stages
        if len(refresh_intervals) != num_stages:
            raise ValueError(
                f"expected one refresh interval per down stage "
                f"({num_stages}), got {len(refresh_intervals)}"
            )
        self.refresh_intervals = list(refresh_intervals)
        self.cached_features: List[Optional[torch.Tensor]] = [None] * num_stages
        self.refresh_counts = [0] * num_stages
        self.reuse_counts = [0] * num_stages
        # position within the denoising schedule; driven by `clockwork_denoise`
        self.step_index = 0
        self.refresh_points: List[List[int]] = [[] for _ in range(num_stages)]

    def set_schedule(self, timesteps):
        """Fix the denoising schedule and precompute each stage's refresh points."""
        self.refresh_points = stage_timesteps(timesteps, self.refresh_intervals)

    def cache_stats(self):
        """Per-stage (refreshes, reuses) over the schedule run so far."""
        return list(zip(self.refresh_counts, self.reuse_counts))

    def forward(self, sample, timestep, global_cond=None):
        x = sample.moveaxis(-1, -2)
        diffusion_step_encoder = self.unet.diffusion_step_encoder

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.expand(x.shape[0])

        global_feature = diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.unet.down_modules):
            if self.step_index in self.refresh_points[idx]:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
                h.append(x)
                self.refresh_counts[idx] += 1
            else:
                # reuse the activations this stage produced at its last refresh
                x = self.cached_features[idx]
                h.append(x)
                self.reuse_counts[idx] += 1
            self.cached_features[idx] = x
            x = downsample(x)

        for mid_module in self.unet.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.unet.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.unet.final_conv(x)
        return x.moveaxis(-1, -2)


def clockwork_denoise(
    unet,
    scheduler,
    noisy_action_seq: torch.Tensor,
    global_cond: Optional[torch.Tensor],
    refresh_intervals: List[int],
    on_step: Optional[Callable] = None,
) -> torch.Tensor:
    """Run a DDPM denoising loop with per-resolution feature caching.

    Drop-in replacement for the ``for k in scheduler.timesteps`` block in
    ``Agent.get_action``: the wrapped UNet re-evaluates each stage only on its
    clock, and the scheduler contract is unchanged.

    Args:
        unet: the trained `ConditionalUnet1D`.
        scheduler: the DDPM scheduler, already configured with its timesteps.
        noisy_action_seq: (B, pred_horizon, act_dim) initial noise.
        global_cond: (B, global_cond_dim) FiLM conditioning.
        refresh_intervals: one interval per down stage, coarse-to-fine.
        on_step: optional callback ``(step_index, timestep)`` invoked once per
            denoising step, before the UNet call.
    """
    cached = ClockworkUnet1D(unet, refresh_intervals)
    cached.set_schedule(scheduler.timesteps)
    for step_index, k in enumerate(scheduler.timesteps):
        cached.step_index = step_index
        if on_step is not None:
            on_step(step_index, k)
        noise_pred = cached(
            sample=noisy_action_seq, timestep=k, global_cond=global_cond
        )
        noisy_action_seq = scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=noisy_action_seq,
        ).prev_sample
    return noisy_action_seq
