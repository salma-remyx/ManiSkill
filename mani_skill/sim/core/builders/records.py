# Defines typed geometry and joint records shared by backend-agnostic builders.

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
JointLimits = tuple[float, float]

UNBOUNDED_LIMITS: JointLimits = (float("-inf"), float("inf"))


@dataclass(frozen=True, slots=True, kw_only=True)
class PoseRecord:
    """Describe a backend-agnostic rigid transform."""

    position: Vector3 = (0.0, 0.0, 0.0)
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class _CollisionRecordBase:
    """Provide fields shared by every typed collision record."""

    name: str = ""
    pose: PoseRecord = PoseRecord()
    material: object | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaneCollisionRecord(_CollisionRecordBase):
    """Describe an infinite plane collision."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BoxCollisionRecord(_CollisionRecordBase):
    """Describe a box collision."""

    half_size: Vector3 = (1.0, 1.0, 1.0)
    density: float = 1000.0


@dataclass(frozen=True, slots=True, kw_only=True)
class CapsuleCollisionRecord(_CollisionRecordBase):
    """Describe a capsule collision."""

    radius: float = 1.0
    half_length: float = 1.0
    density: float = 1000.0


@dataclass(frozen=True, slots=True, kw_only=True)
class CylinderCollisionRecord(_CollisionRecordBase):
    """Describe a cylinder collision."""

    radius: float = 1.0
    half_length: float = 1.0
    density: float = 1000.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SphereCollisionRecord(_CollisionRecordBase):
    """Describe a sphere collision."""

    radius: float = 1.0
    density: float = 1000.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvexMeshCollisionRecord(_CollisionRecordBase):
    """Describe one convex collision mesh loaded from a file."""

    filename: str
    scale: Vector3 = (1.0, 1.0, 1.0)
    density: float = 1000.0


@dataclass(frozen=True, slots=True, kw_only=True)
class MultipleConvexMeshCollisionRecord(_CollisionRecordBase):
    """Describe multiple convex collision meshes loaded from a file."""

    filename: str
    scale: Vector3 = (1.0, 1.0, 1.0)
    density: float = 1000.0
    decomposition: Literal["none", "coacd"] = "none"
    decomposition_params: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class NonconvexMeshCollisionRecord(_CollisionRecordBase):
    """Describe one non-convex collision mesh loaded from a file."""

    filename: str
    scale: Vector3 = (1.0, 1.0, 1.0)
    density: float = 1000.0


CollisionRecord: TypeAlias = (
    PlaneCollisionRecord
    | BoxCollisionRecord
    | CapsuleCollisionRecord
    | CylinderCollisionRecord
    | SphereCollisionRecord
    | ConvexMeshCollisionRecord
    | MultipleConvexMeshCollisionRecord
    | NonconvexMeshCollisionRecord
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _VisualRecordBase:
    """Provide fields shared by every typed visual record."""

    name: str = ""
    pose: PoseRecord = PoseRecord()
    material: object | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaneVisualRecord(_VisualRecordBase):
    """Describe a plane visual."""

    width: float = 10.0
    length: float = 10.0


@dataclass(frozen=True, slots=True, kw_only=True)
class BoxVisualRecord(_VisualRecordBase):
    """Describe a box visual."""

    half_size: Vector3 = (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class CapsuleVisualRecord(_VisualRecordBase):
    """Describe a capsule visual."""

    radius: float = 1.0
    half_length: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class CylinderVisualRecord(_VisualRecordBase):
    """Describe a cylinder visual."""

    radius: float = 1.0
    half_length: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SphereVisualRecord(_VisualRecordBase):
    """Describe a sphere visual."""

    radius: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class MeshVisualRecord(_VisualRecordBase):
    """Describe a visual mesh loaded from a file."""

    filename: str
    scale: Vector3 = (1.0, 1.0, 1.0)


VisualRecord: TypeAlias = (
    PlaneVisualRecord
    | BoxVisualRecord
    | CapsuleVisualRecord
    | CylinderVisualRecord
    | SphereVisualRecord
    | MeshVisualRecord
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _JointRecordBase:
    """Provide fields shared by every typed articulation joint record."""

    name: str = ""
    pose_in_parent: PoseRecord = PoseRecord()
    pose_in_child: PoseRecord = PoseRecord()
    friction: float = 0.0
    damping: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class UndefinedJointRecord(_JointRecordBase):
    """Describe an unspecified root joint."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedJointRecord(_JointRecordBase):
    """Describe a fixed joint."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RevoluteJointRecord(_JointRecordBase):
    """Describe a bounded revolute joint."""

    limits: JointLimits = UNBOUNDED_LIMITS


@dataclass(frozen=True, slots=True, kw_only=True)
class RevoluteUnwrappedJointRecord(_JointRecordBase):
    """Describe an unwrapped revolute joint."""

    limits: JointLimits = UNBOUNDED_LIMITS


@dataclass(frozen=True, slots=True, kw_only=True)
class PrismaticJointRecord(_JointRecordBase):
    """Describe a bounded prismatic joint."""

    limits: JointLimits = UNBOUNDED_LIMITS


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuousJointRecord(_JointRecordBase):
    """Describe a continuous revolute joint."""


JointRecord: TypeAlias = (
    UndefinedJointRecord
    | FixedJointRecord
    | RevoluteJointRecord
    | RevoluteUnwrappedJointRecord
    | PrismaticJointRecord
    | ContinuousJointRecord
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MimicJointRecord:
    """Describe a linear mimic relationship between two named joints."""

    joint: str
    mimic: str
    multiplier: float = 1.0
    offset: float = 0.0
