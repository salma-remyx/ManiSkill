from dataclasses import dataclass

from mani_skill.utils.logging import logger


@dataclass
class BackendInfo:
    physics_backend_package: str
    """the package name of the physics simulation backend"""
    physics_backend: str
    """the full backend name of the physics simulation"""
    physics_device_id: str | None
    """the device id of the physics simulation"""
    render_backend_package: str
    """the package name of the renderer"""
    render_backend: str
    """the full backend name of the renderer"""
    render_device_id: str | None
    """the device id of the renderer"""


def _parse_backend_device_id(
    backend: str, physics_backend: bool = True
) -> tuple[str, str, str | None]:
    if "." in backend:
        package_name, backend_name = backend.split(".")
        parts = backend_name.split(":")
        if len(parts) >= 2:
            if parts[1] == "cpu":
                return package_name, parts[0], parts[1]
            if parts[1] == "cuda":
                device_id = "0"
                if len(parts) == 3:
                    device_id = parts[2]
                return package_name, parts[0], f"cuda:{device_id}"
        return package_name, backend_name, None
    else:
        # Backward compatability for old backend format
        logger.warning(
            f"Using legacy backend naming: {backend}. Please use the new format "
            "<package_name.backend_name> instead."
        )

        if physics_backend:
            if backend == "physx_cpu":
                return "sapien", "physx_cpu", None
            elif backend == "physx_cuda":
                return "sapien", "physx_cuda", None
            elif backend == "gpu":
                return "sapien", "physx_cuda", None
            elif backend == "cuda":
                return "sapien", "physx_cuda", None
            elif backend == "cpu":
                return "sapien", "physx_cpu", None
        else:
            if backend == "sapien_cpu":
                return "sapien", "sapien_cpu", None
            elif backend == "sapien_cuda":
                return "sapien", "sapien_cuda", None
            elif backend == "cuda":
                return "sapien", "sapien_cuda", None
            elif backend == "gpu":
                return "sapien", "sapien_cuda", None
            elif backend == "cpu":
                return "sapien", "sapien_cpu", None
    raise ValueError(
        f"Invalid backend: {backend}. Should be in the format "
        "<package_name.backend_name> or <package_name.backend_name:cpu> or "
        "<package_name.backend_name:cuda:<device_id>>."
    )


def parse_sim_and_render_backend(
    physics_backend: str, render_backend: str
) -> BackendInfo:
    # Parse sim_backend string to check for CUDA device specification
    package_name, physics_backend, sim_device_id = _parse_backend_device_id(
        physics_backend, physics_backend=True
    )
    render_package_name, render_backend, render_device_id = _parse_backend_device_id(
        render_backend, physics_backend=False
    )
    return BackendInfo(
        physics_backend_package=package_name,
        physics_backend=f"{package_name}.{physics_backend}",
        physics_device_id=sim_device_id,
        render_backend_package=render_package_name,
        render_backend=f"{render_package_name}.{render_backend}",
        render_device_id=render_device_id,
    )
