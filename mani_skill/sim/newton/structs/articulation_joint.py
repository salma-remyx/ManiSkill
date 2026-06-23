from dataclasses import dataclass

from mani_skill.utils.structs.articulation_joint import ArticulationJoint


@dataclass(kw_only=True)
class NewtonArticulationJoint(ArticulationJoint):
    pass
