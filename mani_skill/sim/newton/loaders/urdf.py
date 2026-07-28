from __future__ import annotations

import warnings
from typing import Any, TypedDict

from mani_skill.sim.loaders.urdf import BaseURDFLoader
from mani_skill.sim.newton.builders.actor import NewtonActorBuilder
from mani_skill.sim.newton.builders.articulation import NewtonArticulationBuilder
from mani_skill.sim.newton.sim import NewtonSim
from mani_skill.sim.newton.structs.articulation import NewtonArticulation


class ParsedURDFData(TypedDict):
    articulation_builders: list[NewtonArticulationBuilder]
    actor_builders: list[NewtonActorBuilder]
    cameras: list[Any]


class NewtonURDFLoader(BaseURDFLoader):
    """Barebones URDF loader backed by Newton's native URDF importer."""

    sim: NewtonSim
    name: str = ""
    disable_self_collisions: bool = False
    fix_root_link: bool = True
    load_multiple_collisions_from_file: bool = False
    scale: float = 1.0

    def parse_urdf_config(self, config_dict: dict) -> dict:
        if config_dict:
            warnings.warn(
                "Newton does not currently support ManiSkill URDF material "
                "configuration; ignoring it",
                stacklevel=2,
            )
        return {}

    def parse(
        self,
        urdf_file: str,
        srdf_file: str | None = None,
        package_dir: str | None = None,
    ) -> ParsedURDFData:
        if srdf_file is not None:
            raise NotImplementedError("Newton does not currently support SRDF files")

        # Newton resolves relative mesh paths against the URDF itself. Its importer does
        # not currently expose an equivalent to SAPIEN's package_dir argument.
        del package_dir

        builder = NewtonArticulationBuilder()
        builder.sim = self.sim
        builder.set_name(self.name)
        builder._mb.add_urdf(
            source=urdf_file,
            floating=not self.fix_root_link,
            scale=self.scale,
            enable_self_collisions=not self.disable_self_collisions,
        )
        return {
            "articulation_builders": [builder],
            "actor_builders": [],
            "cameras": [],
        }

    def load_file_as_articulation_builder(
        self,
        urdf_file: str,
        srdf_file: str | None = None,
        package_dir: str | None = None,
    ) -> NewtonArticulationBuilder:
        return self.parse(urdf_file, srdf_file, package_dir)["articulation_builders"][0]

    def load(
        self,
        urdf_file: str,
        srdf_file: str | None = None,
        package_dir: str | None = None,
        name: str | None = None,
        scene_idxs: list[int] | None = None,
    ) -> NewtonArticulation:
        if name is not None:
            self.name = name

        builder = self.load_file_as_articulation_builder(
            urdf_file, srdf_file, package_dir
        )
        builder.set_scene_idxs(scene_idxs)
        return builder.build()
