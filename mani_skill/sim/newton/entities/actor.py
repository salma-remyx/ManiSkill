# Provides batched Newton actor state access through Warp kernels.
# pyright: basic, reportIndexIssue=false, reportArgumentType=false
# pyright: reportReturnType=false, reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import warp as wp

from mani_skill.sim.core.builders.actor import ActorBuilder
from mani_skill.sim.core.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.newton.sim import NewtonSim


@wp.kernel
def _gather_pose_kernel(
    body_q: wp.array[wp.transformf],
    body_indices: wp.array[wp.int32],
    output: wp.array[wp.transformf],
):
    index = wp.tid()
    output[index] = body_q[body_indices[index]]


@wp.kernel
def _scatter_pose_kernel(
    body_q: wp.array[wp.transformf],
    body_indices: wp.array[wp.int32],
    values: wp.array[wp.transformf],
):
    index = wp.tid()
    body_q[body_indices[index]] = values[index]


@wp.kernel
def _gather_linear_velocity_kernel(
    body_qd: wp.array[wp.spatial_vectorf],
    body_indices: wp.array[wp.int32],
    output: wp.array[wp.vec3f],
):
    index = wp.tid()
    output[index] = wp.spatial_top(body_qd[body_indices[index]])


@wp.kernel
def _scatter_linear_velocity_kernel(
    body_qd: wp.array[wp.spatial_vectorf],
    body_indices: wp.array[wp.int32],
    values: wp.array[wp.vec3f],
):
    index = wp.tid()
    body_index = body_indices[index]
    current = body_qd[body_index]
    value = values[index]
    body_qd[body_index] = wp.spatial_vectorf(
        value[0], value[1], value[2], current[3], current[4], current[5]
    )


@wp.kernel
def _gather_angular_velocity_kernel(
    body_qd: wp.array[wp.spatial_vectorf],
    body_indices: wp.array[wp.int32],
    output: wp.array[wp.vec3f],
):
    index = wp.tid()
    output[index] = wp.spatial_bottom(body_qd[body_indices[index]])


@wp.kernel
def _scatter_angular_velocity_kernel(
    body_qd: wp.array[wp.spatial_vectorf],
    body_indices: wp.array[wp.int32],
    values: wp.array[wp.vec3f],
):
    index = wp.tid()
    body_index = body_indices[index]
    current = body_qd[body_index]
    value = values[index]
    body_qd[body_index] = wp.spatial_vectorf(
        current[0], current[1], current[2], value[0], value[1], value[2]
    )


class NewtonActor:
    """Class to manage an actor in newton simulations."""

    def __init__(
        self,
        sim: NewtonSim,
        actor_model_builder: newton.ModelBuilder,
        builder: ActorBuilder,
        body_index: int,
    ) -> None:
        """Create a binding from a template-scene body index.

        Args:
            sim: Newton simulation that owns the actor data.
            actor_model_builder: The newton model builder for the actor.
            builder: The ActorBuilder for the actor.
            body_index: Body index in the unreplicated scene model builder.
        """
        self.sim = sim
        self.mb = actor_model_builder
        """The newton model builder for the actor."""
        self.builder = builder
        """The ActorBuilder for the actor."""
        self.body_index = body_index
        self._body_indices: wp.array[wp.int32]

    def set_body_indices(self, body_indices: wp.array[wp.int32]) -> None:
        """Bind the actor to its packed body indices after scene compilation.

        Args:
            body_indices: One body index for every replicated environment.
        """
        self._body_indices = body_indices

    @property
    def pose(self) -> Pose:
        """Gather and return the actor pose for every environment."""
        output = wp.empty(
            len(self._body_indices),
            dtype=wp.transformf,
            device=self.sim.state.body_q.device,
        )
        wp.launch(
            _gather_pose_kernel,
            dim=len(self._body_indices),
            inputs=[self.sim.state.body_q, self._body_indices],
            outputs=[output],
            device=output.device,
        )
        return Pose.create(output, device=output.device, copy=False)

    @pose.setter
    def pose(self, value: Pose) -> None:
        """Scatter batched actor poses into Newton's rigid-body state."""
        wp.launch(
            _scatter_pose_kernel,
            dim=len(self._body_indices),
            inputs=[self.sim.state.body_q, self._body_indices, value.raw_pose],
            device=self.sim.state.body_q.device,
        )

    @property
    def linear_velocity(self) -> wp.array[wp.vec3f]:
        """Gather linear velocity for every environment."""
        output = wp.empty(
            len(self._body_indices),
            dtype=wp.vec3f,
            device=self.sim.state.body_qd.device,
        )
        wp.launch(
            _gather_linear_velocity_kernel,
            dim=len(self._body_indices),
            inputs=[self.sim.state.body_qd, self._body_indices],
            outputs=[output],
            device=output.device,
        )
        return output

    @linear_velocity.setter
    def linear_velocity(self, value: wp.array[wp.vec3f]) -> None:
        """Scatter batched linear velocity into Newton's rigid-body state."""
        wp.launch(
            _scatter_linear_velocity_kernel,
            dim=len(self._body_indices),
            inputs=[self.sim.state.body_qd, self._body_indices, value],
            device=self.sim.state.body_qd.device,
        )

    @property
    def angular_velocity(self) -> wp.array[wp.vec3f]:
        """Gather angular velocity for every environment."""
        output = wp.empty(
            len(self._body_indices),
            dtype=wp.vec3f,
            device=self.sim.state.body_qd.device,
        )
        wp.launch(
            _gather_angular_velocity_kernel,
            dim=len(self._body_indices),
            inputs=[self.sim.state.body_qd, self._body_indices],
            outputs=[output],
            device=output.device,
        )
        return output

    @angular_velocity.setter
    def angular_velocity(self, value: wp.array[wp.vec3f]) -> None:
        """Scatter batched angular velocity into Newton's rigid-body state."""
        wp.launch(
            _scatter_angular_velocity_kernel,
            dim=len(self._body_indices),
            inputs=[self.sim.state.body_qd, self._body_indices, value],
            device=self.sim.state.body_qd.device,
        )
