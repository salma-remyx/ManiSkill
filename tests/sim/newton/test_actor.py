# Verifies Newton actor compilation and batched rigid-body state access.

from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
import warp as wp

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sim.core.entities.actor import Actor
from mani_skill.sim.core.pose import Pose
from mani_skill.sim.newton.entities.actor import NewtonActor
from mani_skill.sim.newton.sim import NewtonSim, NewtonSimConfig


def _warp_to_numpy(value: wp.array[Any]) -> np.ndarray[Any, Any]:
    """Return a Warp array as a statically typed NumPy array."""
    return cast(np.ndarray[Any, Any], value.numpy())


def _add_box(
    scene: ManiSkillScene,
    name: str,
    position: list[float],
    scene_idxs: list[int] | None = None,
) -> Actor:
    """Build one box actor in a scene for a test.

    Args:
        scene: Scene that receives the actor.
        name: Actor name.
        position: Initial actor position.
        scene_idxs: Optional parallel scene IDs in which to create the actor.

    Returns:
        The backend-neutral box actor.
    """
    builder = scene.create_actor_builder()
    if scene_idxs is not None:
        builder.set_scene_idxs(scene_idxs)
    builder.set_initial_pose(Pose(position))
    builder.add_box_collision(half_size=(0.5, 0.5, 0.5))
    builder.add_box_visual(half_size=(0.5, 0.5, 0.5))
    return builder.build(name)


def test_actor_builder_reuses_shared_backend_actor() -> None:
    sim = NewtonSim()
    scene = ManiSkillScene(physics_sim=sim, render_sim=sim)

    actor = _add_box(scene, "box", [0.0, 0.0, 0.0])

    assert isinstance(actor.physics_actor, NewtonActor)
    assert actor.render_actor is actor.physics_actor
    assert actor.physics_sim is sim
    assert actor.render_sim is sim
    assert len(sim.scene_mb.body_q) == 1
    assert len(sim.scene_mb.shape_body) == 0
    assert len(cast(NewtonActor, actor.physics_actor).mb.shape_body) == 2
    assert sim.actors == {"box": actor.physics_actor}


def test_actor_builder_splits_physics_and_render_backends() -> None:
    physics_sim = NewtonSim()
    render_sim = NewtonSim()
    scene = ManiSkillScene(physics_sim=physics_sim, render_sim=render_sim)

    actor = _add_box(scene, "box", [0.0, 0.0, 0.0])

    assert actor.physics_actor is not actor.render_actor
    assert len(cast(NewtonActor, actor.physics_actor).mb.shape_body) == 1
    assert len(cast(NewtonActor, actor.render_actor).mb.shape_body) == 1
    assert physics_sim.actors == {"box": actor.physics_actor}
    assert render_sim.actors == {"box": actor.render_actor}


def test_newton_compiles_primitive_collision_and_visual_records() -> None:
    sim = NewtonSim()
    scene = ManiSkillScene(physics_sim=sim, render_sim=sim)
    builder = scene.create_actor_builder()
    builder.add_plane_collision()
    builder.add_box_collision()
    builder.add_capsule_collision()
    builder.add_cylinder_collision()
    builder.add_sphere_collision()
    builder.add_plane_visual()
    builder.add_box_visual()
    builder.add_capsule_visual()
    builder.add_cylinder_visual()
    builder.add_sphere_visual()

    builder.build_static("primitives")
    sim.compile_physical_scene()

    assert sim.model.body_count == 1
    assert sim.model.shape_count == 10


def test_newton_compiles_mesh_collision_and_visual_records(tmp_path: Path) -> None:
    mesh_path = tmp_path / "box.obj"
    trimesh.creation.box().export(mesh_path)  # pyright: ignore[reportUnknownMemberType]
    sim = NewtonSim()
    scene = ManiSkillScene(physics_sim=sim, render_sim=sim)
    builder = scene.create_actor_builder()
    builder.add_convex_collision_from_file(str(mesh_path))
    builder.add_multiple_convex_collisions_from_file(str(mesh_path))
    builder.add_nonconvex_collision_from_file(str(mesh_path))
    builder.add_visual_from_file(str(mesh_path))

    builder.build_static("meshes")
    sim.compile_physical_scene()

    assert sim.model.body_count == 1
    assert sim.model.shape_count == 4


def test_newton_actor_batched_state_round_trip() -> None:
    sim = NewtonSim(num_envs=2, cfg=NewtonSimConfig(spacing=0.0))
    scene = ManiSkillScene(physics_sim=sim, render_sim=sim)
    first = _add_box(scene, "first", [1.0, 2.0, 3.0])
    second = _add_box(scene, "second", [4.0, 5.0, 6.0])

    sim.compile_physical_scene()

    np.testing.assert_array_equal(
        _warp_to_numpy(cast(NewtonActor, first.physics_actor)._body_indices),
        [1, 4],
    )
    np.testing.assert_array_equal(
        _warp_to_numpy(cast(NewtonActor, second.physics_actor)._body_indices),
        [2, 5],
    )
    np.testing.assert_allclose(
        first.pose.numpy(),
        [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]] * 2,
    )

    first.pose = Pose([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device="cpu")
    first.linear_velocity = wp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=wp.vec3,
        device="cpu",
    )
    first.angular_velocity = wp.array(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        dtype=wp.vec3,
        device="cpu",
    )

    np.testing.assert_allclose(
        first.pose.numpy(),
        [
            [7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )
    np.testing.assert_allclose(
        _warp_to_numpy(first.linear_velocity),
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    np.testing.assert_allclose(
        _warp_to_numpy(first.angular_velocity),
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    )


def test_newton_actor_replication_respects_scene_idxs() -> None:
    sim = NewtonSim(num_envs=3, cfg=NewtonSimConfig(spacing=0.0))
    sim.physics_device = None
    scene = ManiSkillScene(physics_sim=sim, render_sim=sim)
    all_scenes = _add_box(scene, "all", [1.0, 2.0, 3.0])
    selected_scenes = _add_box(
        scene,
        "selected",
        [4.0, 5.0, 6.0],
        scene_idxs=[2, 0],
    )

    sim.compile_render_scene()

    assert sim.model.body_count == 8
    assert selected_scenes.scene_idxs == [2, 0]
    np.testing.assert_array_equal(
        _warp_to_numpy(cast(NewtonActor, all_scenes.physics_actor)._body_indices),
        [1, 4, 6],
    )
    np.testing.assert_array_equal(
        _warp_to_numpy(cast(NewtonActor, selected_scenes.physics_actor)._body_indices),
        [7, 2],
    )
    np.testing.assert_allclose(
        selected_scenes.pose.numpy(),
        [[4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]] * 2,
    )
