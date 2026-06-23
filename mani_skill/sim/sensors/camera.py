from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence, TypedDict, TypeVar

import torch

from mani_skill.sim.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim


class CameraParams(TypedDict):
    extrinsic_cv: torch.Tensor
    cam2world_gl: torch.Tensor
    intrinsic_cv: torch.Tensor


@dataclass
class CameraConfig(BaseSensorConfig):
    uid: str
    """uid (str): unique id of the camera"""
    pose: Pose
    """Pose of the camera"""
    width: int
    """width of the camera"""
    height: int
    """height of the camera"""
    fov: float | None = None
    """The field of view of the camera. Either fov or intrinsic must be given"""
    near: float = 0.01
    """near plane of the camera"""
    far: float = 100
    """far plane of the camera"""
    intrinsic: Array | None = None
    """intrinsics matrix of the camera. Either fov or intrinsic must be given"""
    entity_uid: str | None = None
    """unique id of the entity to mount the camera. Defaults to None. Only used by agent classes that want to define mounted cameras."""
    mount: Actor | Link | None = None
    """the Actor or Link to mount the camera on top of. This means the global pose of the mounted camera is now mount.pose * local_pose"""
    sapien_kwargs: dict[str, str] = field(default_factory=dict)
    """kwargs to pass to the sapien camera constructor"""

    @classmethod
    def from_generic_camera_config(cls, cfg: CameraConfig):
        """
        Parse a generic CameraConfig into the appropriate CameraConfig subclass.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + str(self.__dict__) + ")"


T = TypeVar("T", bound=BaseSensorConfig)


def update_sensor_configs_from_dict(
    sensor_configs: dict[str, T],
    config_dict: dict[str, dict],
    render_backend_package: str,
):
    # First, apply global configuration
    for k, v in config_dict.items():
        if k in sensor_configs:
            continue
        for config in sensor_configs.values():
            if isinstance(config, CameraConfig):
                if not hasattr(config, k):
                    raise AttributeError(
                        f"{k} is not a valid attribute of CameraConfig"
                    )
                else:
                    setattr(config, k, v)
    # Then, apply camera-specific configuration
    for name, v in config_dict.items():
        if name not in sensor_configs:
            continue

        config = sensor_configs[name]
        if isinstance(config, CameraConfig):
            for kk in v:
                assert hasattr(config, kk), (
                    f"{kk} is not a valid attribute of CameraConfig"
                )
            v = copy.deepcopy(v)
            # for json serailizable gym.make args, user has to pass a list, not a Pose object.
            if "pose" in v and isinstance(v["pose"], list):
                import sapien  # TODO (stao): remove this

                v["pose"] = sapien.Pose(v["pose"][:3], v["pose"][3:])
            config.__dict__.update(v)
    for k in sensor_configs.keys():
        cfg = sensor_configs[k]
        if isinstance(cfg, CameraConfig):
            if render_backend_package == "sapien":
                from mani_skill.sim.sapien.sensors.camera import SapienCameraConfig

                sensor_configs[k] = SapienCameraConfig.from_generic_camera_config(cfg)


def parse_sensor_configs(
    sensor_configs: Sequence[T] | dict[str, T] | T,
) -> dict[str, T]:
    """
    Given sensor configs defined in any form, parse them into a dictionary of CameraConfig objects.
    Depending on the render backend package, the sensor configs will be parsed into the appropriate
    CameraConfig subclass.

    Args:
        sensor_configs: Sensor configs defined in any form.
        render_backend_package: The render backend package to use.

    Returns:
        A dictionary of CameraConfig objects.
    """
    if isinstance(sensor_configs, (tuple, list)):
        return dict([(config.uid, config) for config in sensor_configs])
    elif isinstance(sensor_configs, dict):
        return dict(sensor_configs)
    elif isinstance(sensor_configs, BaseSensorConfig):
        return dict([(sensor_configs.uid, sensor_configs)])
    else:
        raise TypeError(type(sensor_configs))


class Camera(BaseSensor):
    """Implementation of the Camera sensor which uses the sapien Camera."""

    config: CameraConfig
    sim: BaseSim

    def __init__(
        self,
        camera_config: CameraConfig,
        sim: BaseSim,
        articulation: Articulation | None = None,
    ) -> None:
        super().__init__(config=camera_config)
        self.sim = sim
        self.articulation = articulation

    def capture(self):
        """
        Capture the data from the camera. Should be non-blocking if possible.
        """
        raise NotImplementedError()

    def get_obs(
        self,
        rgb: bool = True,
        depth: bool = True,
        position: bool = True,
        segmentation: bool = True,
        normal: bool = False,
        albedo: bool = False,
        apply_texture_transforms: bool = True,
    ):
        """
        Get the desired image observations from the camera.

        Args:
            rgb: Whether to return the RGB image.
            depth: Whether to return the depth image.
            position: Whether to return the position image.
            segmentation: Whether to return the segmentation image.
            normal: Whether to return the normal image.
            albedo: Whether to return the albedo image.
            apply_texture_transforms: Whether to apply texture transforms to the simulated sensor
                data to map to standard texture formats. Usually you want to leave this to
                it's default True value. If false, it will return raw values from the renderer.
                For details on standard texture formats, see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html#shaders-and-textures

        Returns:
            A dictionary of the desired image observations.
        """
        raise NotImplementedError()

    def get_images(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Get RGB versions of images in the observation dictionary
        """
        raise NotImplementedError()

    # TODO (stao): Computing camera parameters on GPU sim is not that fast, especially with mounted cameras and for model_matrix computation.
    def get_params(self) -> CameraParams:
        raise NotImplementedError()
