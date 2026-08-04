# Defines backend-agnostic actor construction records and their builder API.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, Self, Sequence, cast

from mani_skill.sim.core.builders.records import (
    BoxCollisionRecord,
    BoxVisualRecord,
    CapsuleCollisionRecord,
    CapsuleVisualRecord,
    CollisionRecord,
    ConvexMeshCollisionRecord,
    CylinderCollisionRecord,
    CylinderVisualRecord,
    MeshVisualRecord,
    MultipleConvexMeshCollisionRecord,
    NonconvexMeshCollisionRecord,
    PlaneCollisionRecord,
    PlaneVisualRecord,
    PoseRecord,
    SphereCollisionRecord,
    SphereVisualRecord,
    Vector3,
    VisualRecord,
)
from mani_skill.sim.core.builders.utils import pose_to_record, validate_pose
from mani_skill.sim.core.pose import Pose

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.sim.core.entities.actor import Actor

ActorBodyType = Literal["dynamic", "kinematic", "static"]


def _vector3(value: Sequence[float], parameter_name: str) -> Vector3:
    """Convert any unbatched three-element sequence to Vector3 (a tuple of three floats).

    Args:
        value: Sequence to convert.
        parameter_name: Parameter name to include in validation errors.

    Returns:
        A Vector3 (a tuple of three floats).

    Raises:
        ValueError: If the sequence does not contain exactly three elements.
    """
    result = tuple(float(component) for component in value)
    if len(result) != 3:
        raise ValueError(f"{parameter_name} must contain exactly three elements")
    return result


class ActorBuilder:
    """A class for programmatically creating an Actor via collision and visual shapes."""

    def __init__(self, scene: ManiSkillScene | None = None) -> None:
        """Initialize an empty actor description.

        Args:
            scene: Scene that will eventually coordinate backend compilation.
        """
        self.scene = scene
        """The ManiSkillScene this actor will be built in and managed by."""
        self.initial_pose = PoseRecord()
        """The initial world pose of the actor."""
        self.collision_records: list[CollisionRecord] = []
        """The collision records of the actor."""
        self.visual_records: list[VisualRecord] = []
        """The visual records of the actor."""
        self.name = ""
        """The name of the actor once built."""
        self.body_type: ActorBodyType = "dynamic"
        """The body type of the actor once built. Can be "dynamic", "kinematic", or "static".
        Defaults to "dynamic"."""
        self.scene_idxs: list[int] | None = None
        """The scene indices of the actor. If none, actor will be replicated in every parallel
        environment."""

    def set_scene_idxs(self, scene_idxs: list[int]) -> Self:
        """Restrict this actor to selected parallel scene indices.

        Args:
            scene_idxs: Parallel scene IDs in the desired actor batch order.

        Returns:
            This builder.
        """
        self.scene_idxs = scene_idxs
        return self

    @validate_pose
    def set_initial_pose(self, pose: Pose) -> Self:
        """Set the initial world pose.

        Args:
            pose: Initial world pose.

        Raises:
            TypeError: If pose is not a Pose object.
            ValueError: If pose has a batch size other than 1.

        Returns:
            This builder.
        """
        self.initial_pose = pose_to_record(pose)
        return self

    def set_name(self, name: str) -> Self:
        """Set the name.

        Args:
            name: Name of the actor once built.

        Returns:
            This builder.
        """
        self.name = name
        return self

    @validate_pose
    def add_plane_collision(
        self,
        *,
        pose: Pose | None = None,
        material: object | None = None,
        name: str = "",
    ) -> Self:
        """Record an infinite plane collision.

        Args:
            name: Optional shape name.
            pose: Plane pose in the actor frame.
            material: Backend-agnostic or backend-native material descriptor.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.collision_records.append(
            PlaneCollisionRecord(
                name=name,
                pose=pose_to_record(pose),
                material=material,
            )
        )
        return self

    @validate_pose
    def add_box_collision(
        self,
        *,
        pose: Pose | None = None,
        half_size: Sequence[float] = (1.0, 1.0, 1.0),
        material: object | None = None,
        density: float = 1000.0,
        name: str = "",
    ) -> Self:
        """Record a box collision.

        Args:
            half_size: Half extents along the local x, y, and z axes.
            name: Optional shape name.
            pose: Shape pose in the actor frame.
            material: Backend-agnostic or backend-native material descriptor.
            density: Shape density used for automatic mass calculation.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.collision_records.append(
            BoxCollisionRecord(
                name=name,
                pose=pose_to_record(pose),
                half_size=_vector3(half_size, "half_size"),
                material=material,
                density=density,
            )
        )
        return self

    @validate_pose
    def add_capsule_collision(
        self,
        *,
        pose: Pose | None = None,
        radius: float = 1.0,
        half_length: float = 1.0,
        material: object | None = None,
        density: float = 1000.0,
        name: str = "",
    ) -> Self:
        """Record a capsule collision whose length is aligned with the local x-axis.

        Args:
            radius: Capsule radius.
            half_length: Half length of the cylindrical section.
            name: Optional shape name.
            pose: Shape pose in the actor frame.
            material: Backend-agnostic or backend-native material descriptor.
            density: Shape density used for automatic mass calculation.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.collision_records.append(
            CapsuleCollisionRecord(
                name=name,
                pose=pose_to_record(pose),
                radius=float(radius),
                half_length=float(half_length),
                material=material,
                density=density,
            )
        )
        return self

    @validate_pose
    def add_cylinder_collision(
        self,
        *,
        pose: Pose | None = None,
        radius: float = 1.0,
        half_length: float = 1.0,
        material: object | None = None,
        density: float = 1000.0,
        name: str = "",
    ) -> Self:
        """Record a cylinder collision whose length is aligned with the local x-axis.

        Args:
            radius: Cylinder radius.
            half_length: Half length along the cylinder axis.
            name: Optional shape name.
            pose: Shape pose in the actor frame.
            material: Backend-agnostic or backend-native material descriptor.
            density: Shape density used for automatic mass calculation.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.collision_records.append(
            CylinderCollisionRecord(
                name=name,
                pose=pose_to_record(pose),
                radius=float(radius),
                half_length=float(half_length),
                material=material,
                density=density,
            )
        )
        return self

    @validate_pose
    def add_sphere_collision(
        self,
        *,
        pose: Pose | None = None,
        radius: float = 1.0,
        material: object | None = None,
        density: float = 1000.0,
        name: str = "",
    ) -> Self:
        """Record a sphere collision.

        Args:
            radius: Sphere radius.
            name: Optional shape name.
            pose: Shape pose in the actor frame.
            material: Backend-agnostic or backend-native material descriptor.
            density: Shape density used for automatic mass calculation.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.collision_records.append(
            SphereCollisionRecord(
                name=name,
                pose=pose_to_record(pose),
                radius=float(radius),
                material=material,
                density=density,
            )
        )
        return self

    @validate_pose
    def add_convex_collision_from_file(
        self,
        filename: str,
        *,
        pose: Pose | None = None,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
        material: object | None = None,
        density: float = 1000.0,
        name: str = "",
    ) -> Self:
        """Record a convex collision mesh loaded from a file.

        Args:
            filename: Mesh file to load during backend compilation.
            scale: Mesh scale along each local axis.
            name: Optional shape name.
            pose: Shape pose in the actor frame.
            material: Backend-agnostic or backend-native material descriptor.
            density: Shape density used for automatic mass calculation.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.collision_records.append(
            ConvexMeshCollisionRecord(
                filename=filename,
                scale=_vector3(scale, "scale"),
                name=name,
                pose=pose_to_record(pose),
                material=material,
                density=density,
            )
        )
        return self

    @validate_pose
    def add_multiple_convex_collisions_from_file(
        self,
        filename: str,
        *,
        pose: Pose | None = None,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
        material: object | None = None,
        density: float = 1000.0,
        decomposition: Literal["none", "coacd"] = "none",
        decomposition_params: Mapping[str, object] | None = None,
        name: str = "",
    ) -> Self:
        """Record multiple convex collision meshes loaded from a file.

        Args:
            filename: Mesh file to load during backend compilation.
            scale: Mesh scale along each local axis.
            name: Optional shape name.
            pose: Shape pose in the actor frame.
            material: Backend-agnostic or backend-native material descriptor.
            density: Shape density used for automatic mass calculation.
            decomposition: Optional convex decomposition strategy.
            decomposition_params: Strategy-specific decomposition options.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.collision_records.append(
            MultipleConvexMeshCollisionRecord(
                filename=filename,
                scale=_vector3(scale, "scale"),
                name=name,
                pose=pose_to_record(pose),
                material=material,
                density=density,
                decomposition=decomposition,
                decomposition_params=decomposition_params,
            )
        )
        return self

    @validate_pose
    def add_nonconvex_collision_from_file(
        self,
        filename: str,
        *,
        pose: Pose | None = None,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
        material: object | None = None,
        density: float = 1000.0,
        name: str = "",
    ) -> Self:
        """Record a non-convex collision mesh loaded from a file.

        Args:
            filename: Mesh file to load during backend compilation.
            scale: Mesh scale along each local axis.
            name: Optional shape name.
            pose: Shape pose in the actor frame.
            material: Backend-agnostic or backend-native material descriptor.
            density: Shape density used for automatic mass calculation.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.collision_records.append(
            NonconvexMeshCollisionRecord(
                filename=filename,
                scale=_vector3(scale, "scale"),
                name=name,
                pose=pose_to_record(pose),
                material=material,
                density=density,
            )
        )
        return self

    @validate_pose
    def add_plane_visual(
        self,
        *,
        pose: Pose | None = None,
        width: float = 10.0,
        length: float = 10.0,
        material: object | None = None,
        name: str = "",
    ) -> Self:
        """Record a plane visual.

        Args:
            width: Plane width.
            length: Plane length.
            material: Backend-agnostic or backend-native material descriptor.
            name: Optional shape name.
            pose: Shape pose in the actor frame.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.visual_records.append(
            PlaneVisualRecord(
                width=width,
                length=length,
                material=material,
                name=name,
                pose=pose_to_record(pose),
            )
        )
        return self

    @validate_pose
    def add_box_visual(
        self,
        *,
        pose: Pose | None = None,
        half_size: Sequence[float] = (1.0, 1.0, 1.0),
        material: object | None = None,
        name: str = "",
    ) -> Self:
        """Record a box visual.

        Args:
            half_size: Half extents along the local x, y, and z axes.
            material: Backend-agnostic or backend-native material descriptor.
            name: Optional shape name.
            pose: Shape pose in the actor frame.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.visual_records.append(
            BoxVisualRecord(
                half_size=_vector3(half_size, "half_size"),
                material=material,
                name=name,
                pose=pose_to_record(pose),
            )
        )
        return self

    @validate_pose
    def add_capsule_visual(
        self,
        *,
        pose: Pose | None = None,
        radius: float = 1.0,
        half_length: float = 1.0,
        material: object | None = None,
        name: str = "",
    ) -> Self:
        """Record a capsule visual whose length is aligned with the local x-axis.

        Args:
            radius: Capsule radius.
            half_length: Half length of the cylindrical section.
            material: Backend-agnostic or backend-native material descriptor.
            name: Optional shape name.
            pose: Shape pose in the actor frame.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.visual_records.append(
            CapsuleVisualRecord(
                radius=float(radius),
                half_length=float(half_length),
                material=material,
                name=name,
                pose=pose_to_record(pose),
            )
        )
        return self

    @validate_pose
    def add_cylinder_visual(
        self,
        *,
        pose: Pose | None = None,
        radius: float = 1.0,
        half_length: float = 1.0,
        material: object | None = None,
        name: str = "",
    ) -> Self:
        """Record a cylinder visual whose length is aligned with the local x-axis.

        Args:
            radius: Cylinder radius.
            half_length: Half length along the cylinder axis.
            material: Backend-agnostic or backend-native material descriptor.
            name: Optional shape name.
            pose: Shape pose in the actor frame.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.visual_records.append(
            CylinderVisualRecord(
                radius=float(radius),
                half_length=float(half_length),
                material=material,
                name=name,
                pose=pose_to_record(pose),
            )
        )
        return self

    @validate_pose
    def add_sphere_visual(
        self,
        *,
        pose: Pose | None = None,
        radius: float = 1.0,
        material: object | None = None,
        name: str = "",
    ) -> Self:
        """Record a sphere visual.

        Args:
            radius: Sphere radius.
            material: Backend-agnostic or backend-native material descriptor.
            name: Optional shape name.
            pose: Shape pose in the actor frame.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.visual_records.append(
            SphereVisualRecord(
                radius=float(radius),
                material=material,
                name=name,
                pose=pose_to_record(pose),
            )
        )
        return self

    @validate_pose
    def add_visual_from_file(
        self,
        filename: str,
        *,
        pose: Pose | None = None,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
        material: object | None = None,
        name: str = "",
    ) -> Self:
        """Record a visual mesh loaded from a file.

        Args:
            filename: Mesh file to load during backend compilation.
            scale: Mesh scale along each local axis.
            material: Optional material override.
            name: Optional shape name.
            pose: Shape pose in the actor frame.

        Raises:
            TypeError: If pose is not a Pose.
            ValueError: If pose does not contain exactly one transform.

        Returns:
            This builder.
        """
        self.visual_records.append(
            MeshVisualRecord(
                filename=filename,
                scale=_vector3(scale, "scale"),
                material=material,
                name=name,
                pose=pose_to_record(pose),
            )
        )
        return self

    def _build(self, name: str, body_type: ActorBodyType) -> Actor:
        """Compile and register an actor with the scene's simulation backends.

        Args:
            name: Unique actor name.
            body_type: Physical behavior of the actor.

        Returns:
            The backend-neutral actor spanning the scene's backends.
        """
        from mani_skill.sim.core.entities.actor import Actor

        self.name = name
        self.body_type = body_type

        scene = cast("ManiSkillScene", self.scene)
        physics_sim = scene.physics_sim
        render_sim = scene.render_sim
        shared_sim = physics_sim is render_sim

        physics_actor = physics_sim.add_actor_builder(
            self,
            build_physics=True,
            build_render=shared_sim,
        )
        render_actor = physics_actor
        if not shared_sim:
            render_actor = render_sim.add_actor_builder(
                self,
                build_physics=False,
                build_render=True,
            )

        actor = Actor(
            name=name,
            physics_sim=physics_sim,
            render_sim=render_sim,
            physics_actor=physics_actor,
            render_actor=render_actor,
        )
        actor.scene_idxs = (
            range(physics_sim.num_envs) if self.scene_idxs is None else self.scene_idxs
        )
        return actor

    def build(self, name: str = "") -> Actor:
        """Build a dynamic actor in the scene's simulation backends.

        Args:
            name: Unique actor name.

        Returns:
            The backend-neutral actor spanning the scene's backends.
        """
        return self._build(name, "dynamic")

    def build_kinematic(self, name: str = "") -> Actor:
        """Build a kinematic actor in the scene's simulation backends.

        Args:
            name: Unique actor name.

        Returns:
            The backend-neutral actor spanning the scene's backends.
        """
        return self._build(name, "kinematic")

    def build_static(self, name: str = "") -> Actor:
        """Build a static actor in the scene's simulation backends.

        Args:
            name: Unique actor name.

        Returns:
            The backend-neutral actor spanning the scene's backends.
        """
        return self._build(name, "static")
