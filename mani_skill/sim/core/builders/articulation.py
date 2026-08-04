# Defines backend-agnostic articulation and link construction APIs.

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Self

from mani_skill.sim.core.builders.actor import ActorBuilder
from mani_skill.sim.core.builders.records import (
    FixedJointRecord,
    JointRecord,
    MimicJointRecord,
    PoseRecord,
    UndefinedJointRecord,
)
from mani_skill.sim.core.builders.utils import pose_to_record, validate_pose
from mani_skill.sim.core.pose import Pose

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.sim.core.entities.articulation import Articulation


class LinkBuilder(ActorBuilder):
    """A class for programmatically creating an articulation link. It's a light wrapper
    around ActorBuilder that adds joint-specific functionality.
    """

    def __init__(
        self,
        index: int,
        parent: LinkBuilder | None,
        scene: ManiSkillScene | None = None,
    ) -> None:
        """Initialize an empty articulation link description.

        Args:
            index: Stable index of this link within its articulation.
            parent: Parent link, or ``None`` for the root link.
            scene: Scene that will eventually coordinate backend compilation.
        """
        super().__init__(scene=scene)
        self.index = index
        self.parent = parent
        self.joint_record: JointRecord = UndefinedJointRecord()

    def set_joint_name(self, name: str) -> Self:
        """Set the current joint record's name.

        Args:
            name: Joint name to store.

        Returns:
            This builder.
        """
        self.joint_record = replace(self.joint_record, name=name)
        return self

    def set_joint_properties(self, joint_record: JointRecord) -> Self:
        """Replace this link's joint description with a typed record.

        Args:
            joint_record: Typed joint record connecting this link to its parent.

        Returns:
            This builder.
        """
        self.joint_record = joint_record
        return self

    def validate(self) -> None:
        """Validate whether the joint record is valid for this link position.

        Raises:
            ValueError: If a root has a movable joint or a child has an undefined joint.
        """
        if self.parent is None:
            if not isinstance(
                self.joint_record, (UndefinedJointRecord, FixedJointRecord)
            ):
                raise ValueError("the root link joint must be undefined or fixed")
        elif isinstance(self.joint_record, UndefinedJointRecord):
            raise ValueError("a non-root link must define its joint type")


class ArticulationBuilder:
    """A class for programmatically creating an Articulation."""

    def __init__(self, scene: ManiSkillScene | None = None) -> None:
        """Initialize an empty articulation description.
        Args:
            scene: ManiSkillScene this articulation will be built in and managed by.
        """
        self.scene = scene
        """The ManiSkillScene this articulation will be built in and managed by."""
        self.initial_pose = PoseRecord()
        """The initial world pose of the articulation."""
        self.link_builders: list[LinkBuilder] = []
        """The link builders of the articulation."""
        self.mimic_joint_records: list[MimicJointRecord] = []
        """The mimic joint records of the articulation."""
        self.name = ""
        """The name of the articulation once built."""

    @validate_pose
    def set_initial_pose(self, pose: Pose) -> Self:
        """Set the articulation's initial world pose.

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

    def create_link_builder(self, parent: LinkBuilder | None = None) -> LinkBuilder:
        """Create and register a link builder.

        Args:
            parent: Existing parent link, or ``None`` when creating the root.

        Returns:
            The new link builder.

        Raises:
            ValueError: If the root already exists or the parent is not registered.
        """
        if self.link_builders and parent is None:
            raise ValueError("only the first articulation link can omit its parent")
        if parent is not None and parent not in self.link_builders:
            raise ValueError("the parent link must belong to this articulation builder")

        link_builder = LinkBuilder(
            index=len(self.link_builders),
            parent=parent,
            scene=self.scene,
        )
        self.link_builders.append(link_builder)
        return link_builder

    def add_mimic_joint(
        self,
        joint: str,
        mimic: str,
        *,
        multiplier: float = 1.0,
        offset: float = 0.0,
    ) -> Self:
        """Record a linear mimic relationship between two joints.

        Args:
            joint: Name of the joint that follows another joint.
            mimic: Name of the joint being followed.
            multiplier: Scale applied to the source joint position.
            offset: Offset added after scaling.

        Returns:
            This builder.
        """
        self.mimic_joint_records.append(
            MimicJointRecord(
                joint=joint,
                mimic=mimic,
                multiplier=multiplier,
                offset=offset,
            )
        )
        return self

    def build(
        self,
        name: str = "",
        *,
        fix_root_link: bool | None = None,
        build_mimic_joints: bool = True,
    ) -> Articulation:
        """Build an articulation after backend compilation support is implemented.

        Args:
            name: Unique articulation name.
            fix_root_link: Whether to override the root joint as fixed.
            build_mimic_joints: Whether to compile recorded mimic relationships.

        Raises:
            NotImplementedError: Until simulation backends implement record compilation.
        """
        self.name = name
        for link_builder in self.link_builders:
            link_builder.validate()
        raise NotImplementedError(
            "Articulation record compilation is not implemented yet"
        )
