from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp

from mani_skill.sim.newton.structs.articulation_joint import (
    NewtonArticulationJoint,
)
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


@dataclass(kw_only=True)
class NewtonArticulation(Articulation):
    _models: list[newton.ModelBuilder] = field(default_factory=list)
    """The Newton model builders that build this articulation."""
    sim: NewtonSim
    """The Newton simulation containing this articulation."""
    _initial_pose: Pose
    """The pose used when the articulation was added to the scene."""
    joints: list[NewtonArticulationJoint] = field(default_factory=list)
    """The joints of the articulation."""
    joints_map: dict[str, NewtonArticulationJoint] = field(default_factory=dict)
    """The map of joint names to the joints."""
    active_joints: list[NewtonArticulationJoint] = field(default_factory=list)
    """The active joints of the articulation."""
    active_joints_map: dict[str, NewtonArticulationJoint] = field(default_factory=dict)
    """The map of active joint names to the active joints."""

    @classmethod
    def create_from_model(
        cls,
        model: newton.ModelBuilder,
        sim: NewtonSim,
        name: str,
        initial_pose: Pose,
        xform: wp.transform,
    ):
        sim._scene_mb.add_builder(model, xform=xform, label_prefix=name or None)

        # Newton stores joint labels as paths, such as ``panda/panda_joint1``.
        all_joint_names = [
            label.rsplit("/", maxsplit=1)[-1] for label in model.joint_label
        ]
        active_joint_names = [
            joint_name
            for joint_name, dof_dim in zip(all_joint_names, model.joint_dof_dim)
            if sum(dof_dim) > 0
        ]
        active_joint_indices = [
            all_joint_names.index(joint_name) for joint_name in active_joint_names
        ]

        scene_idxs = list(range(sim.num_envs))
        joints = []
        for joint_index, joint_name in enumerate(all_joint_names):
            active_joint_index = (
                active_joint_indices.index(joint_index)
                if joint_index in active_joint_indices
                else None
            )
            joints.append(
                NewtonArticulationJoint.create(
                    sim=sim,
                    scene_idxs=scene_idxs,
                    name=joint_name,
                    articulation=None,
                    joint_index=torch.zeros(
                        sim.num_envs,
                        dtype=torch.int32,
                        device=sim.sim_device_torch,
                    )
                    + joint_index,
                    active_joint_index=(
                        torch.zeros(
                            sim.num_envs,
                            dtype=torch.int32,
                            device=sim.sim_device_torch,
                        )
                        + active_joint_index
                        if active_joint_index is not None
                        else None
                    ),
                )
            )

        active_joints = [joints[index] for index in active_joint_indices]
        articulation = cls(
            _models=[model],
            sim=sim,
            name=name,
            _initial_pose=initial_pose,
            joints=joints,
            joints_map={joint.name: joint for joint in joints},
            active_joints=active_joints,
            active_joints_map={joint.name: joint for joint in active_joints},
            _scene_idxs=scene_idxs,
        )

        for joint in joints:
            joint.articulation = articulation

        return articulation

    @cached_property
    def max_dof(self) -> int:
        return max(model.joint_dof_count for model in self._models)

    @cached_property
    def qlimits(self):
        padded_qlimits = np.array(
            [
                np.concatenate(
                    [
                        np.stack([model.joint_limit_lower, model.joint_limit_upper]).T,
                        np.zeros((self.max_dof - model.joint_dof_count, 2)),
                    ]
                )
                for model in self._models
            ]
        )
        padded_qlimits = torch.from_numpy(padded_qlimits).float()
        return padded_qlimits.to(self.device)
