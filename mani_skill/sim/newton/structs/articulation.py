from dataclasses import dataclass

from mani_skill.utils.structs.articulation import Articulation


@dataclass(kw_only=True)
class NewtonArticulation(Articulation):
    pass
