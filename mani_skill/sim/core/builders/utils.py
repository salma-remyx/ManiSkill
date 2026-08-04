# Validates public builder inputs and converts poses into internal records.

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Callable, ParamSpec, TypeVar

from mani_skill.sim.core.builders.records import PoseRecord
from mani_skill.sim.core.pose import Pose

P = ParamSpec("P")
R = TypeVar("R")


def validate_pose(method: Callable[P, R]) -> Callable[P, R]:
    """Validate the pose argument before calling a builder method.

    Args:
        method: Builder method whose signature contains a pose parameter.

    Returns:
        A wrapped method that accepts only singleton poses or None.

    Raises:
        TypeError: If a supplied pose is neither a Pose nor None.
        ValueError: If the decorated method has no pose parameter or the pose has a
            batch size other than 1.
    """
    method_signature = signature(method)
    if "pose" not in method_signature.parameters:
        raise ValueError("validate_pose needs a method with a pose parameter")

    @wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound_arguments = method_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        pose = bound_arguments.arguments["pose"]
        if pose is not None:
            if not isinstance(pose, Pose):
                raise TypeError("pose must be a Pose object")
            if len(pose) != 1:
                raise ValueError(f"pose must have a batch size of 1; got {len(pose)}")
        return method(*args, **kwargs)

    return wrapper


def pose_to_record(pose: Pose | None) -> PoseRecord:
    """Convert a pose object to a PoseRecord object for builders.

    Args:
        pose: Pose object, or None for identity.

    Returns:
        A PoseRecord.
    """
    if pose is None:
        return PoseRecord()
    packed_pose = pose.numpy()[0]
    return PoseRecord(
        position=(
            float(packed_pose[0]),
            float(packed_pose[1]),
            float(packed_pose[2]),
        ),
        orientation=(
            float(packed_pose[3]),
            float(packed_pose[4]),
            float(packed_pose[5]),
            float(packed_pose[6]),
        ),
    )
