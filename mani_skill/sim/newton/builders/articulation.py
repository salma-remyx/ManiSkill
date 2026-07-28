from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import warp as wp

from mani_skill.sim.builders.articulation import BaseArticulationBuilder
from mani_skill.sim.newton.structs.articulation import NewtonArticulation
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


class NewtonArticulationBuilder(BaseArticulationBuilder):
    sim: NewtonSim

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this articulation in. If None, the articulation will
    be built in all scenes."""

    _mb: newton.ModelBuilder
    """The model builder for the articulation."""

    def __init__(self):
        super().__init__()
        self._mb = newton.ModelBuilder()
        self._initial_pose = Pose.create_from_pq()
        self.name = ""

    def set_name(self, name: str):
        self.name = name
        return self

    def set_scene_idxs(self, scene_idxs: list[int] | None = None):
        self.scene_idxs = scene_idxs
        return self

    @property
    def initial_pose(self) -> Pose:
        return Pose.create(self._initial_pose)

    @initial_pose.setter
    def initial_pose(self, initial_pose: Pose):
        self._initial_pose = Pose.create(initial_pose)

    def build(self, name: str | None = None) -> NewtonArticulation:
        if name is not None:
            self.name = name

        position = self._initial_pose.p[0].tolist()
        quaternion = self._initial_pose.q[0].tolist()
        xform = wp.transform(
            position,
            # ManiSkill stores quaternions as wxyz; Warp expects xyzw.
            (quaternion[1], quaternion[2], quaternion[3], quaternion[0]),
        )
        articulation = NewtonArticulation.create_from_model(
            model=self._mb,
            sim=self.sim,
            name=self.name,
            initial_pose=self._initial_pose,
            xform=xform,
        )
        self.sim.articulations[self.name] = articulation
        return articulation
