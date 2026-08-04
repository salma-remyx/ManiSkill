# Defines the backend-neutral actor interface exposed to tasks and environments.

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from mani_skill.sim.core.entities.base_entity import BaseEntity
from mani_skill.sim.core.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.core.base_sim import BaseSim


class Actor(BaseEntity):
    """Delegate actor state access to backend-specific actor objects."""

    def __init__(
        self,
        *,
        name: str,
        physics_sim: BaseSim,
        render_sim: BaseSim,
        physics_actor: Actor,
        render_actor: Actor,
    ) -> None:
        """Create an actor spanning its physics and rendering backends.

        Args:
            name: Actor name.
            physics_sim: Simulation backend that owns physical state.
            render_sim: Simulation backend that owns rendering state.
            physics_actor: Backend-specific physical actor.
            render_actor: Backend-specific rendering actor.
        """
        self.name = name
        self.physics_sim = physics_sim
        self.render_sim = render_sim
        self.physics_actor = physics_actor
        self.render_actor = render_actor
        self.scene_idxs = range(physics_sim.num_envs)

    @property
    def pose(self) -> Pose:
        """Return the batched pose from the physics backend."""
        return self.physics_actor.pose

    @pose.setter
    def pose(self, value: Pose) -> None:
        """Set the batched pose on the authoritative physics backend."""
        self.physics_actor.pose = value

    @property
    def linear_velocity(self) -> wp.array[wp.vec3f]:
        """Return the batched linear velocity from the physics backend."""
        return self.physics_actor.linear_velocity

    @linear_velocity.setter
    def linear_velocity(self, value: wp.array[wp.vec3f]) -> None:
        """Set the batched linear velocity on the physics backend."""
        self.physics_actor.linear_velocity = value

    @property
    def angular_velocity(self) -> wp.array[wp.vec3f]:
        """Return the batched angular velocity from the physics backend."""
        return self.physics_actor.angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value: wp.array[wp.vec3f]) -> None:
        """Set the batched angular velocity on the physics backend."""
        self.physics_actor.angular_velocity = value
