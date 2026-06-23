from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sim.builders.articulation import BaseArticulationBuilder

from .robel import build_robel_valve as build_robel_valve


def get_articulation_builder(
    scene: ManiSkillScene,
    id: str,
    fix_root_link: bool = True,
    urdf_config: dict | None = None,
) -> BaseArticulationBuilder:
    """Builds or returns an articulation builder for an ID, specifying dataset and articulation ID.

    Currently these IDs are hardcoded for a few datasets.
    """
    if urdf_config is None:
        urdf_config = dict()
    splits = id.split(":")
    dataset_source = splits[0]
    articulation_id = ":".join(splits[1:])

    if dataset_source == "partnet-mobility":
        from .partnet_mobility import get_partnet_mobility_builder

        builder = get_partnet_mobility_builder(
            scene=scene,
            id=articulation_id,
            fix_root_link=fix_root_link,
            urdf_config=urdf_config,
        )
    else:
        raise RuntimeError(f"No dataset with id {dataset_source} was found")

    return builder
