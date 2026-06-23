from dataclasses import dataclass

from mani_skill.utils.structs.base import BaseStruct
from mani_skill.utils.structs.pose import Pose


@dataclass
class Link(BaseStruct):
    @classmethod
    def merge(cls, links: list["Link"], name: str):
        """
        Merge a list of links into a single link for easy access of data across
        multiple possibly different links.

        Args:
            links: A list of links objects to merge.
            name: The name of the merged link. If none, the name will default to the first link's
                name.
        """
        link_cls: "Link" = links[0].__class__  # type: ignore
        return link_cls.merge(links, name)

    @property
    def pose(self) -> Pose:
        """
        Get the pose of the actor.
        """
        raise NotImplementedError()

    @pose.setter
    def pose(self, pose: Pose):
        """
        Set the pose of the actor.
        """

    def set_pose(self, pose: Pose):
        """
        Set the pose of the actor.
        """
        self.pose = pose
