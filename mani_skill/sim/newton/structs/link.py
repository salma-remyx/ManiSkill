from dataclasses import dataclass

from mani_skill.utils.structs.link import Link


@dataclass(kw_only=True)
class NewtonLink(Link):
    pass
