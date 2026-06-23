from __future__ import annotations

from typing import TYPE_CHECKING, Any

import newton
import warp as wp

from mani_skill.sim.builders.actor import BaseActorBuilder
from mani_skill.sim.newton.structs.actor import NewtonActor
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


class NewtonActorBuilder(BaseActorBuilder):
    sim: NewtonSim

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this actor in. If None, the actor will be
    built in all scenes."""

    _mb: newton.ModelBuilder
    """The model builder for the actor."""

    def __init__(self):
        super().__init__()
        self._mb = newton.ModelBuilder()

    def set_scene_idxs(self, scene_idxs: list[int] | None = None):
        self.scene_idxs = scene_idxs
        return self

    def build(self, name: str):
        actor = NewtonActor.create_from_model(self._mb, self.sim, name)
        self.sim.actors[name] = actor
        return actor

    def build_kinematic(self, name: str):
        actor = NewtonActor.create_from_model(self._mb, self.sim, name)
        self.sim.actors[name] = actor
        return actor

    def build_static(self, name: str):
        actor = NewtonActor.create_from_model(self._mb, self.sim, name)
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
            xform = wp.transform(wp.vec3(pose.p), wp.quat(pose.q))
        self._mb.add_shape_plane(
            xform=xform,
        )
        return self
