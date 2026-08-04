# Defines the PickCube task and its tabletop workspace.

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.envs.base_env import BaseEnv
from mani_skill.envs.registry import register_env
from mani_skill.sim.core import Pose
from mani_skill.sim.newton.sim import NewtonSimConfig

_TABLE_MODEL_FILE = str(PACKAGE_ASSET_DIR / "table.glb")
_TABLE_HEIGHT = 0.9196429
_TABLE_ORIENTATION = (0.0, 0.0, 0.70710678, 0.70710678)


@register_env("PickCube-v1")
class PickCube(BaseEnv):
    @property
    def _default_sim_config(self) -> NewtonSimConfig:
        return NewtonSimConfig(viewer_backend="viser")

    def _load_scene(self) -> None:
        super()._load_scene()

        table_builder = self.scene.create_actor_builder()
        table_builder.add_box_collision(
            pose=Pose((0.0, 0.0, _TABLE_HEIGHT / 2)),
            half_size=(2.418 / 2, 1.209 / 2, _TABLE_HEIGHT / 2),
        )
        table_builder.add_visual_from_file(
            filename=_TABLE_MODEL_FILE,
            scale=(1.75, 1.75, 1.75),
            pose=Pose(q=_TABLE_ORIENTATION),
        )
        table_builder.set_initial_pose(
            Pose(
                (-0.12, 0.0, -_TABLE_HEIGHT),
                q=_TABLE_ORIENTATION,
            )
        )
        self.table = table_builder.build_kinematic(name="table-workspace")

        builder = self.scene.create_actor_builder()
        builder.add_plane_collision(
            pose=Pose((0.0, 0.0, -_TABLE_HEIGHT)),
        )
        builder.add_plane_visual(
            pose=Pose((0.0, 0.0, -_TABLE_HEIGHT)),
            width=0.0,
            length=0.0,
        )
        builder.set_scene_idxs([0])
        builder.build_static(name="plane")

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=(0.05, 0.05, 0.05),
        )
        builder.add_box_visual(
            half_size=(0.05, 0.05, 0.05),
        )
        builder.set_initial_pose(Pose((0.0, 0.0, 0.05)))
        self.cube = builder.build(name="cube")

    def _initialize_episode(self) -> None:
        self.cube.pose = Pose((0.0, 0.0, 0.05))
