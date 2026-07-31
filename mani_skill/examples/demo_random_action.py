from dataclasses import dataclass
from typing import Annotated, Optional, Union

import numpy as np
import tyro

from mani_skill.envs import tasks  # noqa: F401 # pyright: ignore[reportUnusedImport]
from mani_skill.envs.make import make


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    physics_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "newton.mj_warp"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    render_backend: Annotated[str, tyro.conf.arg(aliases=["-rb"])] = "newton.warp"
    """Which render backend to use. Can be 'gpu', 'cpu', 'none'"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    render_mode: str | None = "rgb_array"
    """Render mode"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, list[int]]], tyro.conf.arg(aliases=["-s"])] = (
        None
    )
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers.
    Default is None (no seeds)"""


def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    if args.render_mode == "none":
        args.render_mode = None
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])  # seed np based RNG for action sampling
    verbose = not args.quiet
    env = make(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        num_envs=args.num_envs,
        physics_backend=args.physics_backend,
        render_backend=args.render_backend,
    )

    env.reset(seed=args.seed, reconfigure=True)

    while True:
        # action = env.action_space.sample() if env.action_space is not None else None
        _, reward, terminated, truncated, info = env.step(None)
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)
        if args.render_mode == "human":
            env.render()
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    env.close()


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
