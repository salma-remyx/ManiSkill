from dataclasses import dataclass

from mani_skill.utils.structs.actor import Actor


@dataclass(kw_only=True)
class NewtonActor(Actor):
    pass
