# Processes backend-neutral actor shape records into Newton model builders.

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import newton
import numpy as np
import trimesh
import warp as wp

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

if TYPE_CHECKING:
    from mani_skill.sim.core.builders.actor import ActorBuilder


def pose_record_to_transform(pose: PoseRecord) -> wp.transform:
    """Convert a backend-neutral pose record into a Warp transform.

    Args:
        pose: Pose record to convert.

    Returns:
        The equivalent Warp transform.
    """
    return wp.transform(pose.position, pose.orientation)


def _collision_config(
    material: object | None,
    density: float,
) -> newton.ModelBuilder.ShapeConfig:
    """Create an invisible Newton collision configuration.

    Args:
        material: Optional native Newton shape configuration.
        density: Shape density used for mass and inertia computation.

    Returns:
        A shape configuration for collision-only geometry.
    """
    # TODO (stao): fix and refactor how we handle multiple backend shape configs
    cfg = cast(
        newton.ModelBuilder.ShapeConfig,
        (
            copy.copy(material)
            if material is not None
            else newton.ModelBuilder.ShapeConfig()
        ),
    )
    cfg.density = density
    cfg.is_visible = False
    return cfg


def _visual_config() -> newton.ModelBuilder.ShapeConfig:
    """Create a visible, non-colliding Newton shape configuration.

    Returns:
        A shape configuration for visual-only geometry.
    """
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        collision_group=0,
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )


def _load_mesh(filename: str, *, compute_inertia: bool = True) -> newton.Mesh:
    """Load one triangle mesh and its visual material as Newton geometry.

    Args:
        filename: Mesh file to load.
        compute_inertia: Whether Newton should compute mesh inertia.

    Returns:
        Newton mesh geometry.
    """
    mesh = trimesh.load_mesh(  # pyright: ignore[reportUnknownMemberType]
        filename, force="mesh", process=False
    )
    visual = mesh.visual
    material = getattr(visual, "material", None)

    uvs = getattr(visual, "uv", None)
    texture = None
    color = None
    roughness = None
    metallic = None
    if material is not None:
        texture = getattr(material, "baseColorTexture", None)
        if texture is None:
            texture = getattr(material, "image", None)

        base_color = getattr(material, "baseColorFactor", None)
        if base_color is None:
            base_color = getattr(material, "diffuse", None)
        if base_color is not None:
            color_array = np.asarray(base_color, dtype=np.float32).reshape(-1)
            if len(color_array) >= 3:
                color_array = color_array[:3]
                if np.max(color_array) > 1.0:
                    color_array /= 255.0
                color = (
                    float(color_array[0]),
                    float(color_array[1]),
                    float(color_array[2]),
                )

        roughness_factor = getattr(material, "roughnessFactor", None)
        if roughness_factor is not None:
            roughness = float(roughness_factor)
        metallic_factor = getattr(material, "metallicFactor", None)
        if metallic_factor is not None:
            metallic = float(metallic_factor)

    # Textures are multiplied by the mesh color in Newton. Use white when the
    # source material does not specify a base-color factor to avoid palette tinting.
    if texture is not None and color is None:
        color = (1.0, 1.0, 1.0)

    return newton.Mesh(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32).reshape(-1),
        normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        uvs=None if uvs is None else np.asarray(uvs, dtype=np.float32),
        compute_inertia=compute_inertia,
        color=color,
        roughness=roughness,
        metallic=metallic,
        texture=None if texture is None else np.asarray(texture),
    )


def _load_mesh_pieces(filename: str) -> list[trimesh.Trimesh]:
    """Load each transformed geometry in a mesh file as a separate piece.

    Args:
        filename: Mesh file to load.

    Returns:
        The file's triangle-mesh geometries in scene coordinates.
    """
    scene = trimesh.load_scene(filename, process=False)  # pyright: ignore[reportUnknownMemberType]
    pieces: list[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        piece = scene.geometry[geometry_name].copy()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        piece.apply_transform(transform)  # pyright: ignore[reportUnknownMemberType]
        pieces.append(piece)  # pyright: ignore[reportUnknownArgumentType]
    return pieces


def add_collision_records(
    model_builder: newton.ModelBuilder,
    body_index: int,
    builder: ActorBuilder,
) -> None:
    """Add every collision record to a Newton body.

    Args:
        model_builder: Newton scene model builder to update.
        body_index: Parent Newton body index.
        builder: Backend-neutral actor description.
    """
    for record in builder.collision_records:
        density = getattr(record, "density", 0.0)
        cfg = _collision_config(record.material, density)
        xform = pose_record_to_transform(record.pose)

        if isinstance(record, PlaneCollisionRecord):
            model_builder.add_shape_plane(
                body=body_index,
                xform=xform,
                width=0.0,
                length=0.0,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, BoxCollisionRecord):
            model_builder.add_shape_box(
                body_index,
                xform=xform,
                hx=record.half_size[0],
                hy=record.half_size[1],
                hz=record.half_size[2],
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, CapsuleCollisionRecord):
            model_builder.add_shape_capsule(
                body_index,
                xform=xform,
                radius=record.radius,
                half_height=record.half_length,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, CylinderCollisionRecord):
            model_builder.add_shape_cylinder(
                body_index,
                xform=xform,
                radius=record.radius,
                half_height=record.half_length,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, SphereCollisionRecord):
            model_builder.add_shape_sphere(
                body_index,
                xform=xform,
                radius=record.radius,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, ConvexMeshCollisionRecord):
            model_builder.add_shape_convex_hull(
                body_index,
                xform=xform,
                mesh=_load_mesh(record.filename),
                scale=record.scale,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, MultipleConvexMeshCollisionRecord):
            pieces = _load_mesh_pieces(record.filename)
            if record.decomposition == "coacd":
                raise NotImplementedError(
                    "COACD decomposition is not supported yet for newton"
                )
            if isinstance(pieces, trimesh.Trimesh):
                pieces = [pieces]
            for piece_index, piece in enumerate(pieces):
                geometry = newton.Mesh(
                    np.asarray(piece.vertices, dtype=np.float32),
                    np.asarray(piece.faces, dtype=np.int32).reshape(-1),
                )
                model_builder.add_shape_convex_hull(
                    body_index,
                    xform=xform,
                    mesh=geometry,
                    scale=record.scale,
                    cfg=copy.copy(cfg),
                    label=(f"{record.name}_{piece_index}" if record.name else None),
                )
        elif isinstance(record, NonconvexMeshCollisionRecord):  # pyright: ignore[reportUnnecessaryIsInstance]
            model_builder.add_shape_mesh(
                body_index,
                xform=xform,
                mesh=_load_mesh(record.filename),
                scale=record.scale,
                cfg=cfg,
                label=record.name or None,
            )


def add_visual_records(
    model_builder: newton.ModelBuilder,
    body_index: int,
    builder: ActorBuilder,
) -> None:
    """Add every visual record to a Newton body.

    Args:
        model_builder: Newton scene model builder to update.
        body_index: Parent Newton body index.
        builder: Backend-neutral actor description.
    """
    for record in builder.visual_records:
        cfg = _visual_config()
        xform = pose_record_to_transform(record.pose)

        if isinstance(record, PlaneVisualRecord):
            model_builder.add_shape_plane(
                body=body_index,
                xform=xform,
                width=record.width,
                length=record.length,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, BoxVisualRecord):
            model_builder.add_shape_box(
                body_index,
                xform=xform,
                hx=record.half_size[0],
                hy=record.half_size[1],
                hz=record.half_size[2],
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, CapsuleVisualRecord):
            model_builder.add_shape_capsule(
                body_index,
                xform=xform,
                radius=record.radius,
                half_height=record.half_length,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, CylinderVisualRecord):
            model_builder.add_shape_cylinder(
                body_index,
                xform=xform,
                radius=record.radius,
                half_height=record.half_length,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, SphereVisualRecord):
            model_builder.add_shape_sphere(
                body_index,
                xform=xform,
                radius=record.radius,
                cfg=cfg,
                label=record.name or None,
            )
        elif isinstance(record, MeshVisualRecord):  # pyright: ignore[reportUnnecessaryIsInstance]
            model_builder.add_shape_mesh(
                body_index,
                xform=xform,
                mesh=_load_mesh(record.filename, compute_inertia=False),
                scale=record.scale,
                cfg=cfg,
                label=record.name or None,
            )
