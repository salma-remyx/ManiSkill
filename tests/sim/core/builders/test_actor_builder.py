# Verifies that ActorBuilder records every supported collision and visual shape.

from mani_skill.sim.core.builders.actor import ActorBuilder
from mani_skill.sim.core.builders.records import (
    BoxCollisionRecord,
    BoxVisualRecord,
    CapsuleCollisionRecord,
    CapsuleVisualRecord,
    ConvexMeshCollisionRecord,
    CylinderCollisionRecord,
    CylinderVisualRecord,
    MeshVisualRecord,
    MultipleConvexMeshCollisionRecord,
    NonconvexMeshCollisionRecord,
    PlaneCollisionRecord,
    PlaneVisualRecord,
    PoseRecord,
    SphereCollisionRecord,
    SphereVisualRecord,
)
from mani_skill.sim.core.pose import Pose


def test_actor_builder_records_all_collisions_and_visuals() -> None:
    builder = ActorBuilder()
    pose = Pose([1, 2, 3], [0, 0, 1, 0])
    pose_record = PoseRecord(
        position=(1.0, 2.0, 3.0),
        orientation=(0.0, 0.0, 1.0, 0.0),
    )
    material = object()
    decomposition_params = {"max_convex_hulls": 8}

    assert builder.set_initial_pose(pose) is builder

    builder.add_plane_collision(
        pose=pose,
        material=material,
        name="plane_collision",
    )
    builder.add_box_collision(
        pose=pose,
        half_size=[1, 2, 3],
        material=material,
        density=10,
        name="box_collision",
    )
    builder.add_capsule_collision(
        pose=pose,
        radius=0.2,
        half_length=0.4,
        material=material,
        density=20,
        name="capsule_collision",
    )
    builder.add_cylinder_collision(
        pose=pose,
        radius=0.3,
        half_length=0.6,
        material=material,
        density=30,
        name="cylinder_collision",
    )
    builder.add_sphere_collision(
        pose=pose,
        radius=0.5,
        material=material,
        density=40,
        name="sphere_collision",
    )
    builder.add_convex_collision_from_file(
        "convex.obj",
        pose=pose,
        scale=[1, 2, 3],
        material=material,
        density=50,
        name="convex_collision",
    )
    builder.add_multiple_convex_collisions_from_file(
        "multiple.obj",
        pose=pose,
        scale=[2, 3, 4],
        material=material,
        density=60,
        decomposition="coacd",
        decomposition_params=decomposition_params,
        name="multiple_convex_collision",
    )
    builder.add_nonconvex_collision_from_file(
        "nonconvex.obj",
        pose=pose,
        scale=[3, 4, 5],
        material=material,
        density=70,
        name="nonconvex_collision",
    )

    builder.add_plane_visual(
        pose=pose,
        scale=[1, 2, 3],
        material=material,
        name="plane_visual",
    )
    builder.add_box_visual(
        pose=pose,
        half_size=[2, 3, 4],
        material=material,
        name="box_visual",
    )
    builder.add_capsule_visual(
        pose=pose,
        radius=0.7,
        half_length=0.8,
        material=material,
        name="capsule_visual",
    )
    builder.add_cylinder_visual(
        pose=pose,
        radius=0.9,
        half_length=1.0,
        material=material,
        name="cylinder_visual",
    )
    builder.add_sphere_visual(
        pose=pose,
        radius=1.1,
        material=material,
        name="sphere_visual",
    )
    builder.add_visual_from_file(
        "visual.obj",
        pose=pose,
        scale=[4, 5, 6],
        material=material,
        name="mesh_visual",
    )

    assert builder.initial_pose == pose_record
    assert builder.collision_records == [
        PlaneCollisionRecord(
            pose=pose_record,
            material=material,
            name="plane_collision",
        ),
        BoxCollisionRecord(
            pose=pose_record,
            half_size=(1.0, 2.0, 3.0),
            material=material,
            density=10,
            name="box_collision",
        ),
        CapsuleCollisionRecord(
            pose=pose_record,
            radius=0.2,
            half_length=0.4,
            material=material,
            density=20,
            name="capsule_collision",
        ),
        CylinderCollisionRecord(
            pose=pose_record,
            radius=0.3,
            half_length=0.6,
            material=material,
            density=30,
            name="cylinder_collision",
        ),
        SphereCollisionRecord(
            pose=pose_record,
            radius=0.5,
            material=material,
            density=40,
            name="sphere_collision",
        ),
        ConvexMeshCollisionRecord(
            filename="convex.obj",
            pose=pose_record,
            scale=(1.0, 2.0, 3.0),
            material=material,
            density=50,
            name="convex_collision",
        ),
        MultipleConvexMeshCollisionRecord(
            filename="multiple.obj",
            pose=pose_record,
            scale=(2.0, 3.0, 4.0),
            material=material,
            density=60,
            decomposition="coacd",
            decomposition_params=decomposition_params,
            name="multiple_convex_collision",
        ),
        NonconvexMeshCollisionRecord(
            filename="nonconvex.obj",
            pose=pose_record,
            scale=(3.0, 4.0, 5.0),
            material=material,
            density=70,
            name="nonconvex_collision",
        ),
    ]
    assert builder.visual_records == [
        PlaneVisualRecord(
            pose=pose_record,
            scale=(1.0, 2.0, 3.0),
            material=material,
            name="plane_visual",
        ),
        BoxVisualRecord(
            pose=pose_record,
            half_size=(2.0, 3.0, 4.0),
            material=material,
            name="box_visual",
        ),
        CapsuleVisualRecord(
            pose=pose_record,
            radius=0.7,
            half_length=0.8,
            material=material,
            name="capsule_visual",
        ),
        CylinderVisualRecord(
            pose=pose_record,
            radius=0.9,
            half_length=1.0,
            material=material,
            name="cylinder_visual",
        ),
        SphereVisualRecord(
            pose=pose_record,
            radius=1.1,
            material=material,
            name="sphere_visual",
        ),
        MeshVisualRecord(
            filename="visual.obj",
            pose=pose_record,
            scale=(4.0, 5.0, 6.0),
            material=material,
            name="mesh_visual",
        ),
    ]
