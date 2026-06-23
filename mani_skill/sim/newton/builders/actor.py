from __future__ import annotations

from typing import TYPE_CHECKING, Any

import newton
import warp as wp

from mani_skill.sim.builders.actor import BaseActorBuilder
from mani_skill.sim.newton.structs.actor import NewtonActor
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Vec3

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


class NewtonActorBuilder(BaseActorBuilder):
    sim: NewtonSim

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this actor in. If None, the actor will be
    built in all scenes."""

    _mb: newton.ModelBuilder
    """The model builder for the actor."""

    body_ids: list[int] = []

    _root_body_id: int = -1

    def __init__(self):
        super().__init__()
        self._mb = newton.ModelBuilder()
        self._root_body_id = self._mb.add_body(
            xform=None,
            is_kinematic=False,
        )
        self._initial_pose = Pose.create_from_pq()

    def set_scene_idxs(self, scene_idxs: list[int] | None = None):
        self.scene_idxs = scene_idxs
        return self

    @property
    def initial_pose(self) -> Pose:
        """The initial pose of the actor when it gets built and spawned into the simulation."""
        return Pose.create(self._initial_pose)

    @initial_pose.setter
    def initial_pose(self, initial_pose: Pose):
        self._initial_pose = initial_pose

        self._mb.body_q[self._root_body_id] = wp.transform(
            wp.vec3(*initial_pose.p), wp.quat(*initial_pose.q)
        )

    def build(self, name: str):
        actor = NewtonActor.create_from_model(
            self._mb, self.sim, name, self._initial_pose
        )
        self.sim.actors[name] = actor
        return actor

    def build_kinematic(self, name: str):
        self._mb.body_flags[self._root_body_id] = newton.BodyFlags.KINEMATIC
        actor = NewtonActor.create_from_model(
            self._mb, self.sim, name, self._initial_pose
        )
        self.sim.actors[name] = actor
        return actor

    def build_static(self, name: str):
        # TODO: this might not be correct way of initializing truly static bodies.
        self._mb.body_flags[self._root_body_id] = newton.BodyFlags.KINEMATIC
        actor = NewtonActor.create_from_model(
            self._mb, self.sim, name, self._initial_pose
        )
        self.sim.actors[name] = actor
        return actor

    def add_plane_collision(
        self,
        pose: Pose | None = None,
        material: Any | None = None,
        patch_radius: float = 0,
        min_patch_radius: float = 0,
    ):
        xform = None
        if pose is not None:
            # NOTE (stao): sapien's plane pose is in a different frame to newton's
            rot_q = Pose.create_from_pq(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]).q
            transformed_q = quaternion_multiply(rot_q, pose.q)
            xform = wp.transform(wp.vec3(*pose.p), wp.quat(*transformed_q))
        body_id = self._mb.add_shape_plane(
            xform=xform,
        )
        self.body_ids.append(body_id)
        return self

    def add_box_collision(
        self,
        pose: Pose | None = None,
        half_size: Vec3 = (1.0, 1.0, 1.0),
        material: Any | None = None,
        density: float = 1000.0,
    ):
        xform = None
        if pose is not None:
            xform = wp.transform(wp.vec3(*pose.p), wp.quat(*pose.q))
        body_id = self._mb.add_shape_box(
            body=self._root_body_id,
            xform=xform,
            hx=half_size[0],
            hy=half_size[1],
            hz=half_size[2],
            color=[
                1,
                0,
                0,
            ],
        )
        self.body_ids.append(body_id)
        return self
