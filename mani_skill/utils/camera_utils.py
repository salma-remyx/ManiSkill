import torch

from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_quaternion,
)
from mani_skill.utils.structs.pose import Pose


def look_at(eye, target, up=(0, 0, 1), device=None) -> Pose:
    """Get the camera pose in SAPIEN by the Look-At method.

    Note:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        The SAPIEN camera follows the convention: (forward, right, up) = (x, -y, z)
        while the OpenGL camera follows (forward, right, up) = (-z, x, y)
        Note that the camera coordinate system (OpenGL) is left-hand.

    Args:
        eye: camera location
        target: looking-at location
        up: a general direction of "up" from the camera.
        device: device to put the pose on.
    Returns:
        Pose: camera pose
    """
    # only accept batched input as tensors
    # accept all other input as 1 dimensional
    if not isinstance(eye, torch.Tensor):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
        assert eye.ndim == 1, eye.ndim
        assert len(eye) == 3, len(eye)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32, device=device)
        assert target.ndim == 1, target.ndim
        assert len(target) == 3, len(target)
    if not isinstance(up, torch.Tensor):
        up = torch.tensor(up, dtype=torch.float32, device=device)
        assert up.ndim == 1, up.ndim
        assert len(up) == 3, len(up)

    def normalize_tensor(x, eps=1e-6):
        x = x.view(-1, 3)
        norm = torch.linalg.norm(x, dim=-1)
        zero_vectors = norm < eps
        x[zero_vectors] = torch.zeros(3, device=x.device).float()
        x[~zero_vectors] /= norm[~zero_vectors].view(-1, 1)
        return x

    forward = normalize_tensor(target - eye)

    up = normalize_tensor(up)
    left = torch.cross(up, forward, dim=-1)
    left = normalize_tensor(left)
    up = torch.cross(forward, left, dim=-1)
    rotation = torch.stack([forward, left, up], dim=-1)
    return Pose.create_from_pq(p=eye, q=matrix_to_quaternion(rotation))
