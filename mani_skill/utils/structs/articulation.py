from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch

from mani_skill.utils.structs.articulation_joint import ArticulationJoint
from mani_skill.utils.structs.base import BaseStruct
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

T = TypeVar("T", bound=BaseStruct)


@dataclass(kw_only=True)
class Articulation(Generic[T]):
    merged: bool = False
    """
    Whether or not this articulation object is a merged articulation where it is managing many
    articulations with different DOFs.

    There are a number of caveats when it comes to merged articulations. While merging
    articulations means you can easily fetch padded qpos, qvel, etc. type data, a number of
    attributes and functions will make little sense and you should avoid using them unless you
    are an advanced user. In particular, the list of Links, Joints, their corresponding maps,
    net contact forces of multiple links, no longer make "sense"
    """

    name: str = ""
    """The name of the articulation. Must be unique within the scene."""

    links: list[Link] = field(default_factory=list)
    """List of Link objects forming the articulation"""
    links_map: dict[str, Link] = field(default_factory=dict)
    """Maps link name to the Link object"""

    root: Link = None
    """The root Link"""

    joints: list[ArticulationJoint] = field(default_factory=list)
    """List of Joint objects forming the articulation"""
    joints_map: dict[str, ArticulationJoint] = field(default_factory=dict)
    """Maps joint name to the Joint object"""
    active_joints: list[ArticulationJoint] = field(default_factory=list)
    """List of active Joint objects, referencing elements in self.joints"""
    active_joints_map: dict[str, ArticulationJoint] = field(default_factory=dict)
    """Maps active joint name to the Joint object, referencing elements in self.joints"""

    @classmethod
    def merge(
        cls, articulations: list["Articulation"], name: str, merge_links: bool = False
    ):
        """
        Merge a list of articulations into a single articulation for easy access of data across
        multiple possibly different articulations.

        Args:
            articulations: A list of articulations objects to merge.
            name: The name of the merged articulation.
            merge_links: Whether to merge the links of the articulations. This is by default False
                as often times you merge articulations that have different number of links. Set
                this true if you want to try and merge articulations that have the same number of
                links.
        """
        articulation_cls: "Articulation" = articulations[0].__class__  # type: ignore
        return articulation_cls.merge(articulations, name, merge_links)

    def get_links(self):
        return self.links

    def get_name(self) -> str:
        return self.name

    def get_pose(self) -> Pose:
        return self.pose

    # def get_qacc(self) -> numpy.ndarray[numpy.float32, _Shape[m, 1]]: ...
    def get_qf(self):
        return self.qf

    def get_qlimits(self):
        return self.qlimits

    def get_qpos(self):
        return self.qpos

    def get_qvel(self):
        return self.qvel

    def get_root(self):
        return self.root

    def get_root_angular_velocity(self) -> torch.Tensor:
        return self.root_angular_velocity

    def get_root_linear_velocity(self) -> torch.Tensor:
        return self.root_linear_velocity

    def get_root_pose(self):
        return self.root_pose

    def set_pose(self, arg0: Pose) -> None:
        self.pose = arg0

    @property
    def qf(self) -> torch.Tensor:
        raise NotImplementedError()

    @qf.setter
    def qf(self, qf: Array) -> None:
        raise NotImplementedError()

    @property
    def qpos(self) -> torch.Tensor:
        raise NotImplementedError()

    @qpos.setter
    def qpos(self, qpos: Array) -> None:
        raise NotImplementedError()

    def set_qpos(self, qpos: torch.Tensor):
        """
        Set the qpos of the articulation.
        """
        self.qpos = qpos

    @property
    def qvel(self) -> torch.Tensor:
        raise NotImplementedError()

    @qvel.setter
    def qvel(self, qvel: Array) -> None:
        raise NotImplementedError()

    def set_qvel(self, qvel: torch.Tensor):
        """
        Set the qvel of the articulation.
        """
        self.qvel = qvel

    @property
    def root_angular_velocity(self) -> torch.Tensor:
        raise NotImplementedError()

    @root_angular_velocity.setter
    def root_angular_velocity(self, velocity: Array) -> None:
        raise NotImplementedError()

    def set_root_angular_velocity(self, velocity: torch.Tensor):
        """
        Set the angular velocity of the root link.
        """
        self.root_angular_velocity = velocity

    @property
    def root_linear_velocity(self) -> torch.Tensor:
        raise NotImplementedError()

    @root_linear_velocity.setter
    def root_linear_velocity(self, velocity: Array) -> None:
        raise NotImplementedError()

    def set_root_linear_velocity(self, velocity: torch.Tensor):
        """
        Set the linear velocity of the root link.
        """
        self.root_linear_velocity = velocity

    @property
    def root_pose(self) -> Pose:
        raise NotImplementedError()

    @root_pose.setter
    def root_pose(self, pose: "Pose") -> None:
        raise NotImplementedError()

    def set_root_pose(self, pose: Pose):
        """
        Set the pose of the root link.
        """
        self.root_pose = pose

    @property
    def linear_velocity(self) -> torch.Tensor:
        """
        Get the linear velocity of the articulation.
        """
        raise NotImplementedError()

    @property
    def max_dof(self) -> int:
        """
        The maximum number of DOFs of the articulation.
        """
        raise NotImplementedError()
