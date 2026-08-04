# Provides a batched, Warp-native rigid-pose abstraction for simulation code.
# pyright: basic, reportIndexIssue=false, reportArgumentType=false
# pyright: reportReturnType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Any, TypeAlias, Union

import numpy as np
import warp as wp

from mani_skill.sim.core.types import ArrayLike, DeviceLike

PoseInput: TypeAlias = Union["Pose", ArrayLike, wp.transformf]


@wp.kernel
def _create_transform_kernel(
    positions: wp.array[wp.vec3],
    orientations: wp.array[wp.quat],
    position_is_single: bool,
    orientation_is_single: bool,
    output: wp.array[wp.transform],
):
    index = wp.tid()
    position_index = index
    orientation_index = index
    if position_is_single:
        position_index = 0
    if orientation_is_single:
        orientation_index = 0
    output[index] = wp.transform(
        positions[position_index], orientations[orientation_index]
    )


@wp.kernel
def _multiply_transform_kernel(
    left: wp.array[wp.transform],
    right: wp.array[wp.transform],
    output: wp.array[wp.transform],
):
    index = wp.tid()
    output[index] = wp.transform_multiply(left[index], right[index])


@wp.kernel
def _subtract_transform_kernel(
    left: wp.array[wp.transform],
    right: wp.array[wp.transform],
    output: wp.array[wp.transform],
):
    index = wp.tid()
    output[index] = wp.transform_multiply(
        wp.transform_inverse(right[index]), left[index]
    )


@wp.kernel
def _inverse_transform_kernel(
    transforms: wp.array[wp.transform],
    output: wp.array[wp.transform],
):
    index = wp.tid()
    output[index] = wp.transform_inverse(transforms[index])


@wp.kernel
def _set_position_kernel(
    transforms: wp.array[wp.transform],
    positions: wp.array[wp.vec3],
    position_is_single: bool,
):
    index = wp.tid()
    position_index = index
    if position_is_single:
        position_index = 0
    transforms[index] = wp.transform(
        positions[position_index], wp.transform_get_rotation(transforms[index])
    )


@wp.kernel
def _set_orientation_kernel(
    transforms: wp.array[wp.transform],
    orientations: wp.array[wp.quat],
    orientation_is_single: bool,
):
    index = wp.tid()
    orientation_index = index
    if orientation_is_single:
        orientation_index = 0
    transforms[index] = wp.transform(
        wp.transform_get_translation(transforms[index]),
        orientations[orientation_index],
    )


@wp.kernel
def _transformation_matrix_kernel(
    transforms: wp.array[wp.transform],
    output: wp.array[wp.mat44],
):
    index = wp.tid()
    output[index] = wp.transform_to_matrix(transforms[index])


def _coerce_component(
    value: ArrayLike,
    *,
    dtype: type,
    width: int,
    device: wp.Device,
    name: str,
) -> wp.array:
    """Convert one position or orientation input into a batched Warp array."""
    if isinstance(value, wp.array):
        result = value
        if result.dtype == wp.float32:
            if result.ndim == 1 and result.shape == (width,):
                result = result.reshape((1, width))
            if result.ndim != 2 or result.shape[1] != width:
                raise ValueError(
                    f"{name} must have shape ({width},) or (N, {width}); "
                    f"got {result.shape}"
                )
            result = result.view(dtype)
        elif result.dtype != dtype or result.ndim != 1:
            raise TypeError(
                f"Warp {name} arrays must have dtype {dtype.__name__} or "
                f"float32 with shape (N, {width}); got dtype {result.dtype} "
                f"and shape {result.shape}"
            )
        if len(result) == 0:
            raise ValueError(f"{name} must contain at least one value")
        return result.to(device)

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{name} must have shape ({width},) or (N, {width}); got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value")
    return wp.array(array, dtype=dtype, device=device)


def _resolve_device(
    first: object | None,
    second: object | None,
    requested: DeviceLike | None,
) -> wp.Device:
    """Resolve an explicit or input-derived device for a new pose."""
    if requested is not None:
        return wp.get_device(requested)
    if isinstance(first, wp.array):
        return first.device
    if isinstance(second, wp.array):
        return second.device
    return wp.get_device(None)


def _copy_or_alias_transform_array(
    raw_pose: wp.array[wp.transformf],
    *,
    device: DeviceLike | None,
    copy: bool,
) -> wp.array[wp.transformf]:
    """Copy transform storage by default or explicitly return an alias."""
    resolved_device = raw_pose.device if device is None else wp.get_device(device)
    if not copy:
        if raw_pose.device != resolved_device:
            raise ValueError(
                "copy=False requires the source and target device to match"
            )
        return raw_pose
    if raw_pose.device == resolved_device:
        return wp.clone(raw_pose)
    return raw_pose.to(resolved_device)


def _broadcast_batch_size(left: int, right: int) -> int:
    """Validate two pose batch sizes and return their broadcast size."""
    if left == right:
        return left
    if left == 1:
        return right
    if right == 1:
        return left
    raise ValueError(
        "Pose batch sizes must match or one batch must contain one pose; "
        f"got {left} and {right}"
    )


class Pose:
    """Store a batch of rigid poses as a Warp transform array.

    Positions use three components and orientations use Warp's ``xyzw`` quaternion
    convention. Unbatched inputs are promoted to batches of size one.

    Args:
        p: Position data shaped ``(3,)`` or ``(N, 3)``. Defaults to zero.
        q: Quaternion data shaped ``(4,)`` or ``(N, 4)`` in ``xyzw`` order.
            Defaults to the identity quaternion.
        device: Warp device for the stored transforms. Defaults to CPU.
    """

    raw_pose: wp.array[wp.transformf]

    def __init__(
        self,
        p: ArrayLike | None = None,
        q: ArrayLike | None = None,
        device: DeviceLike = "cpu",
    ) -> None:
        """Create a batched pose from position and quaternion components."""
        created = self.create_from_pq(p=p, q=q, device=device)
        self.raw_pose = created.raw_pose

    @classmethod
    def _from_raw_pose(cls, raw_pose: wp.array[wp.transformf]) -> Pose:
        """Construct a pose around an already validated Warp transform array."""
        pose = cls.__new__(cls)
        pose.raw_pose = raw_pose
        return pose

    @classmethod
    def create_from_pq(
        cls,
        p: ArrayLike | None = None,
        q: ArrayLike | None = None,
        device: DeviceLike = "cpu",
    ) -> Pose:
        """Create a pose from optionally batched position and quaternion data.

        Args:
            p: Position data shaped ``(3,)`` or ``(N, 3)``. Defaults to zero.
            q: Quaternion data shaped ``(4,)`` or ``(N, 4)`` in ``xyzw`` order.
                Defaults to the identity quaternion.
            device: Warp device for the resulting transforms. Defaults to CPU.

        Returns:
            A batched pose containing one or more Warp transforms.

        Raises:
            ValueError: If an input has an invalid shape or incompatible batch size.
            TypeError: If a Warp input has an incompatible data type.
        """
        resolved_device = _resolve_device(p, q, device)
        if p is None:
            p = [[0.0, 0.0, 0.0]]
        if q is None:
            q = [[0.0, 0.0, 0.0, 1.0]]

        positions = _coerce_component(
            p, dtype=wp.vec3, width=3, device=resolved_device, name="position"
        )
        orientations = _coerce_component(
            q, dtype=wp.quat, width=4, device=resolved_device, name="orientation"
        )
        batch_size = _broadcast_batch_size(len(positions), len(orientations))
        raw_pose = wp.empty(
            batch_size,
            dtype=wp.transform,
            device=resolved_device,
            requires_grad=positions.requires_grad or orientations.requires_grad,
        )
        wp.launch(
            _create_transform_kernel,
            dim=batch_size,
            inputs=[
                positions,
                orientations,
                len(positions) == 1,
                len(orientations) == 1,
            ],
            outputs=[raw_pose],
            device=resolved_device,
        )
        return cls._from_raw_pose(raw_pose)

    @classmethod
    def create(
        cls,
        pose: PoseInput | None = None,
        device: DeviceLike = "cpu",
        copy: bool = True,
    ) -> Pose:
        """Create a pose from a pose, transform array, or packed pose data.

        Packed data must have shape ``(7,)`` or ``(N, 7)`` and follow Warp's
        ``[px, py, pz, qx, qy, qz, qw]`` transform layout. Three-component data
        is interpreted as position data with identity orientation.

        Args:
            pose: Source pose data. ``None`` creates one identity pose.
            device: Warp device for the resulting transforms. Defaults to CPU.
            copy: Whether to copy existing pose or transform storage. Set to
                ``False`` to alias storage on the same device. By default, the
                pose data is copied.

        Returns:
            A batched pose on the requested or inferred device.

        Raises:
            ValueError: If packed data has an invalid shape or ``copy=False`` is
                requested with a different target device.
            TypeError: If a Warp array has an unsupported data type.
        """
        if pose is None:
            return cls(device=device)
        if isinstance(pose, cls):
            return cls._from_raw_pose(
                _copy_or_alias_transform_array(
                    pose.raw_pose,
                    device=device,
                    copy=copy,
                )
            )
        if isinstance(pose, wp.transformf):
            return cls._from_raw_pose(
                wp.array([pose], dtype=wp.transform, device=device)
            )
        if isinstance(pose, wp.array):
            resolved_device = _resolve_device(pose, None, device)
            raw_pose = pose
            if raw_pose.dtype == wp.transform and raw_pose.ndim == 1:
                if len(raw_pose) == 0:
                    raise ValueError("pose must contain at least one transform")
                return cls._from_raw_pose(
                    _copy_or_alias_transform_array(
                        raw_pose,
                        device=resolved_device,
                        copy=copy,
                    )
                )
            if raw_pose.dtype != wp.float32:
                raise TypeError(
                    "Warp pose arrays must have dtype transformf or float32; "
                    f"got {raw_pose.dtype}"
                )
            if raw_pose.ndim == 1:
                raw_pose = raw_pose.reshape((1, len(raw_pose)))
            if raw_pose.ndim != 2:
                raise ValueError(
                    f"packed pose data must have one or two dimensions; got {raw_pose.shape}"
                )
            if raw_pose.shape[1] == 3:
                return cls(p=raw_pose, device=resolved_device)
            if raw_pose.shape[1] != 7 or raw_pose.shape[0] == 0:
                raise ValueError(
                    "packed pose data must have shape (7,), (N, 7), (3,), or "
                    f"(N, 3); got {raw_pose.shape}"
                )
            return cls._from_raw_pose(
                _copy_or_alias_transform_array(
                    raw_pose.view(wp.transform),
                    device=resolved_device,
                    copy=copy,
                )
            )

        array = np.asarray(pose, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(
                f"packed pose data must have one or two dimensions; got {array.shape}"
            )
        if array.shape[1] == 3:
            return cls(p=array, device=device)
        if array.shape[1] != 7 or array.shape[0] == 0:
            raise ValueError(
                "packed pose data must have shape (7,), (N, 7), (3,), or "
                f"(N, 3); got {array.shape}"
            )
        return cls._from_raw_pose(wp.array(array, dtype=wp.transform, device=device))

    def __len__(self) -> int:
        """Return the number of poses in the batch."""
        return len(self.raw_pose)

    def __getitem__(self, index: int | slice) -> Pose:
        """Return a batched view containing the selected poses.

        Args:
            index: Integer or slice selecting along the batch dimension.

        Returns:
            A pose whose transform array aliases the selected storage.

        Raises:
            IndexError: If an integer index is outside the batch.
            TypeError: If the index is not an integer or slice.
        """
        if isinstance(index, int):
            normalized_index = index + len(self) if index < 0 else index
            if normalized_index < 0 or normalized_index >= len(self):
                raise IndexError("pose index out of range")
            return self._from_raw_pose(
                self.raw_pose[normalized_index : normalized_index + 1]
            )
        if isinstance(index, slice):
            return self._from_raw_pose(self.raw_pose[index])
        raise TypeError(
            f"pose indices must be integers or slices, not {type(index).__name__}"
        )

    def __repr__(self) -> str:
        """Return a representation that does not synchronize device data."""
        return f"Pose(batch_size={len(self)}, device='{self.device}')"

    def __mul__(self, other: Pose) -> Pose:
        """Compose this pose with another pose of the same batch size.

        Args:
            other: Right-hand pose to compose with this pose.

        Returns:
            The composed pose on this pose's device.

        Raises:
            ValueError: If the poses have different batch sizes.
        """
        right = other.to(self.device)
        batch_size = len(self)
        if batch_size != len(right):
            raise ValueError(
                "Pose multiplication requires equal batch sizes; "
                f"got {batch_size} and {len(right)}"
            )
        output = wp.empty(
            batch_size,
            dtype=wp.transform,
            device=self.device,
            requires_grad=self.raw_pose.requires_grad or right.raw_pose.requires_grad,
        )
        wp.launch(
            _multiply_transform_kernel,
            dim=batch_size,
            inputs=[self.raw_pose, right.raw_pose],
            outputs=[output],
            device=self.device,
        )
        return self._from_raw_pose(output)

    def __sub__(self, other: Pose) -> Pose:
        """Compute this pose relative to another pose.

        The result is ``other.inv() * self``, so ``other * (self - other)``
        reconstructs ``self`` up to floating-point precision.

        Args:
            other: Reference pose of the same batch size.

        Returns:
            This pose expressed in the other pose's reference frame.

        Raises:
            ValueError: If the poses have different batch sizes.
        """
        right = other.to(self.device)
        batch_size = len(self)
        if batch_size != len(right):
            raise ValueError(
                "Pose subtraction requires equal batch sizes; "
                f"got {batch_size} and {len(right)}"
            )
        output = wp.empty(
            batch_size,
            dtype=wp.transform,
            device=self.device,
            requires_grad=self.raw_pose.requires_grad or right.raw_pose.requires_grad,
        )
        wp.launch(
            _subtract_transform_kernel,
            dim=batch_size,
            inputs=[self.raw_pose, right.raw_pose],
            outputs=[output],
            device=self.device,
        )
        return self._from_raw_pose(output)

    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the underlying pose data, which is ``(N, 7)``."""
        return self.raw_pose.shape

    @property
    def device(self) -> wp.Device:
        """Return the Warp device holding the pose data."""
        return self.raw_pose.device

    @property
    def p(self) -> wp.array:
        """Return a zero-copy ``wp.vec3`` view of the batched positions."""
        return self.raw_pose.view(wp.float32)[:, :3].view(wp.vec3)

    @p.setter
    def p(self, value: ArrayLike) -> None:
        """Set positions in place."""
        self.set_p(value)

    @property
    def q(self) -> wp.array:
        """Return a zero-copy ``wp.quat`` view of the batched orientations."""
        return self.raw_pose.view(wp.float32)[:, 3:].view(wp.quat)

    @q.setter
    def q(self, value: ArrayLike) -> None:
        """Set quaternions in place."""
        self.set_q(value)

    def get_p(self) -> wp.array:
        """Return the positions."""
        return self.p

    def get_q(self) -> wp.array:
        """Return the quaternions in ``xyzw`` order."""
        return self.q

    def set_p(self, p: ArrayLike) -> None:
        """Set positions in place."""
        positions = _coerce_component(
            p, dtype=wp.vec3, width=3, device=self.device, name="position"
        )
        _broadcast_batch_size(len(self), len(positions))
        if len(positions) not in (1, len(self)):
            raise ValueError(
                f"position batch of size {len(positions)} cannot replace "
                f"pose batch of size {len(self)}"
            )
        wp.launch(
            _set_position_kernel,
            dim=len(self),
            inputs=[self.raw_pose, positions, len(positions) == 1],
            device=self.device,
        )

    def set_q(self, q: ArrayLike) -> None:
        """Set quaternions in place."""
        orientations = _coerce_component(
            q, dtype=wp.quat, width=4, device=self.device, name="orientation"
        )
        _broadcast_batch_size(len(self), len(orientations))
        if len(orientations) not in (1, len(self)):
            raise ValueError(
                f"orientation batch of size {len(orientations)} cannot replace "
                f"pose batch of size {len(self)}"
            )
        wp.launch(
            _set_orientation_kernel,
            dim=len(self),
            inputs=[self.raw_pose, orientations, len(orientations) == 1],
            device=self.device,
        )

    def inv(self) -> Pose:
        """Return the inverse of the pose data."""
        output = wp.empty(
            len(self),
            dtype=wp.transform,
            device=self.device,
            requires_grad=self.raw_pose.requires_grad,
        )
        wp.launch(
            _inverse_transform_kernel,
            dim=len(self),
            inputs=[self.raw_pose],
            outputs=[output],
            device=self.device,
        )
        return self._from_raw_pose(output)

    def to(self, device: DeviceLike) -> Pose:
        """Move the pose data to a specified Warp device."""
        resolved_device = wp.get_device(device)
        if self.device == resolved_device:
            return self
        return self._from_raw_pose(self.raw_pose.to(resolved_device))

    def numpy(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Return the pose data as a NumPy array with shape ``(N, 7)``."""
        return self.raw_pose.numpy()

    def to_transformation_matrix(self) -> wp.array:
        """Return the pose data as a Warp array of ``wp.mat44`` matrices."""
        output = wp.empty(len(self), dtype=wp.mat44, device=self.device)
        wp.launch(
            _transformation_matrix_kernel,
            dim=len(self),
            inputs=[self.raw_pose],
            outputs=[output],
            device=self.device,
        )
        return output
