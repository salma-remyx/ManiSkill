# SAC MoE

Code for running the SAC MoE is adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl/) and our previous [ManiSkill Baselines](https://github.com/tongzhoumu/ManiSkill_Baselines/). It is written to be single-file and easy to follow/read, and supports state-based RL code.

This baseline was contributed by [@XuGW-Kevin](https://github.com/XuGW-Kevin), [@wang-muhan](https://github.com/wang-muhan) and [@lihe50hz](https://github.com/lihe50hz). If you have any questions please raise an issue here and tag [@XuGW-Kevin](https://github.com/XuGW-Kevin).

Currently, two enhancements have been integrated into the sac_moe.py and sac_moe_rgbd.py files: the [Mixture-of-Expert (MoE) Network](https://arxiv.org/abs/2402.08609) and the [Blended Exploration and Exploitation (BEE) Operator](https://arxiv.org/abs/2306.02865). More experimental features can be added in the future.

## Factorized (multilinear) experts

`sac_moe.py` can also factorize the expert stack instead of evaluating every expert on each forward pass — the multilinear MoE layer of [Multilinear Mixture of Experts: Scalable Expert Specialization through Factorization](https://arxiv.org/abs/2402.12550) (NeurIPS 2024). The experts' weight matrices are parameterized as a single tensor that is contracted with the routing coefficients *in factorized form*, so it is never materialized; routing stays fully differentiable (entmax1.5 rather than top-K), so no load-balancing loss is needed. The input/hidden layers of the expert MLPs are shared, and only their final linear layer becomes the factorized mixture.

```bash
python sac_moe.py --env_id="PushT-v1" --num_envs=32 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos" \
  --moe-mode=mumoe --num-experts=1024 --moe-rank=64
```

`--moe-variant=cp` (default) uses the CP factorization; `--moe-variant=tr` uses the Tensor Ring variant, which is cheaper still at high expert counts. Dropping `--moe-mode` keeps the original dense MoE, which is unchanged.

Because each expert is no longer evaluated separately, `--num-experts` can be raised well past what the dense MoE can afford. With the default `[256, 256, 1]` experts, raising the count from 64 to 1024 grows the whole value/Q network from 162K to 253K parameters, where the dense MoE would grow from 9.0M to 143M; and the forward pass stays a handful of matrix products instead of 1024 MLP evaluations (~25x faster on CPU at batch size 1024 in our measurement). As in the paper, larger expert counts are what buys finer-grained specialization; `factorized_mu_moe.materialize_experts` exposes the implicit expert weight matrices for inspecting that specialization.

## State Based RL

Below is a sample of various commands you can run to train a state-based policy to solve various tasks with SAC MoE. Note that control modes can be changed and can be important for improving sample efficiency.

```bash
python sac_moe.py --env_id="PushT-v1" \
  --num_envs=32 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=1_000_000 --eval_freq=50_000 --control-mode="pd_ee_delta_pos"
```

## Vision Based RL (RGBD)

Below is a sample of various commands for training a image-based policy with SAC MoE that are lightly tuned. You will need to tune the buffer size accordingly as image based observations can take up a lot of memory. The examples.sh file has a full list of tested commands for running visual based SAC successfully on many tasks. Change the `--obs_mode` argument to "rgb", "rgb+depth", "depth" to train on RGB or RGBD observations or Depth observations.

```bash
python sac_moe_rgbd.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --utd=0.5 --buffer_size=500_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000

python sac_moe_rgbd.py --env_id="PickCube-v1" --obs_mode rgb+depth \
  --num_envs=32 --utd=0.5 --buffer_size=500_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=64 --camera_height=64 \
  --total_timesteps=1_000_000 --eval_freq=10_000
```

### Notes and Optimization Tips

By default, most environments in ManiSkill generate 128x128 images or larger. However some tasks don't need such high resolution images to be solved like PushCube-v1 and PickCube-v1. Lower camera resolutions significantly reduce memory usage and can speed up training time. Currently the beta plaground rgbd baseline code is written to support 128x128 and 64x64 images only, for other resolutions you need to modify the neural network architecture accordingly.

You can add `--no-include-state` to exclude any state based information from observations. Note however use this with caution as many environements have goal specification information that is part of the state.

## Citation

If you use this SAC MoE please cite the following

```
@inproceedings{DBLP:conf/icml/HaarnojaZAL18,
  author       = {Tuomas Haarnoja and
                  Aurick Zhou and
                  Pieter Abbeel and
                  Sergey Levine},
  editor       = {Jennifer G. Dy and
                  Andreas Krause},
  title        = {Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
                  with a Stochastic Actor},
  booktitle    = {Proceedings of the 35th International Conference on Machine Learning,
                  {ICML} 2018, Stockholmsm{\"{a}}ssan, Stockholm, Sweden, July
                  10-15, 2018},
  series       = {Proceedings of Machine Learning Research},
  volume       = {80},
  pages        = {1856--1865},
  publisher    = {{PMLR}},
  year         = {2018},
  url          = {http://proceedings.mlr.press/v80/haarnoja18b.html},
  timestamp    = {Wed, 03 Apr 2019 18:17:30 +0200},
  biburl       = {https://dblp.org/rec/conf/icml/HaarnojaZAL18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{
ceron2024mixtures,
title={Mixtures of Experts Unlock Parameter Scaling for Deep {RL}},
author={Johan Samir Obando Ceron and Ghada Sokar and Timon Willi and Clare Lyle and Jesse Farebrother and Jakob Nicolaus Foerster and Gintare Karolina Dziugaite and Doina Precup and Pablo Samuel Castro},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=X9VMhfFxwn}
}

@inproceedings{
ji2024seizing,
title={Seizing Serendipity: Exploiting the Value of Past Success in Off-Policy Actor-Critic},
author={Tianying Ji and Yu Luo and Fuchun Sun and Xianyuan Zhan and Jianwei Zhang and Huazhe Xu},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=9Tq4L3Go9f}
}

@misc{huang2024mentormixtureofexpertsnetworktaskoriented,
      title={MENTOR: Mixture-of-Experts Network with Task-Oriented Perturbation for Visual Reinforcement Learning}, 
      author={Suning Huang and Zheyu Zhang and Tianhai Liang and Yihan Xu and Zhehao Kou and Chenhao Lu and Guowei Xu and Zhengrong Xue and Huazhe Xu},
      year={2024},
      eprint={2410.14972},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.14972}, 
}
```
