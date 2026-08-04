# pyright: basic, reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false
# Verifies the Warp-native batched pose data structure and rigid-transform operations.

import numpy as np
import pytest
import warp as wp
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill.sim.core.pose import Pose


def _xyzw_from_euler(ai: float, aj: float, ak: float) -> np.ndarray:
    wxyz = euler2quat(ai, aj, ak)
    return np.roll(wxyz, -1).astype(np.float32)


def _transformation_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quat2mat(np.roll(orientation, 1))
    matrix[:3, 3] = position
    return matrix


def _assert_pose_matrices(
    pose: Pose,
    expected: list[np.ndarray] | np.ndarray,
    *,
    atol: float = 1e-6,
) -> None:
    expected_array = np.asarray(expected)
    if expected_array.ndim == 2:
        expected_array = expected_array[None]
    np.testing.assert_allclose(
        pose.to_transformation_matrix().numpy(),
        expected_array,
        atol=atol,
    )


def test_pose_creation() -> None:
    pose = Pose.create_from_pq(device="cpu")

    assert pose.raw_pose.shape == (1,)
    assert isinstance(pose.raw_pose, wp.array)
    assert pose.raw_pose.dtype == wp.transformf
    np.testing.assert_allclose(pose.numpy(), [[0, 0, 0, 0, 0, 0, 1]])


def test_pose_creation_defaults_to_cpu() -> None:
    assert Pose().device == wp.get_device("cpu")
    assert Pose.create().device == wp.get_device("cpu")
    assert Pose.create_from_pq().device == wp.get_device("cpu")
    assert Pose.create([1, 2, 3, 0, 0, 0, 1]).device == wp.get_device("cpu")


def test_pose_create_with_position() -> None:
    pose = Pose.create_from_pq(p=[1, 0, 2], device="cpu")
    np.testing.assert_allclose(pose.numpy(), [[1, 0, 2, 0, 0, 0, 1]])

    pose = Pose.create_from_pq(
        p=np.array([[1, 0, 2], [1, 0, -2]], dtype=np.float64),
        device="cpu",
    )
    np.testing.assert_allclose(
        pose.numpy(),
        [[1, 0, 2, 0, 0, 0, 1], [1, 0, -2, 0, 0, 0, 1]],
    )


def test_pose_create_with_orientation_and_component_broadcasting() -> None:
    orientation = _xyzw_from_euler(0.3, 0.4, -0.2)
    pose = Pose.create_from_pq(q=orientation, device="cpu")
    np.testing.assert_allclose(
        pose.numpy(),
        np.concatenate([np.zeros((1, 3)), orientation[None]], axis=1),
    )

    orientations = np.stack(
        [
            orientation,
            _xyzw_from_euler(-0.5, 0.1, 0.7),
        ]
    )
    pose = Pose.create_from_pq(p=[1, 2, 3], q=orientations, device="cpu")
    np.testing.assert_allclose(
        pose.p.numpy(),
        [[1, 2, 3], [1, 2, 3]],
    )
    np.testing.assert_allclose(pose.q.numpy(), orientations)


def test_pose_create_from_packed_and_warp_data() -> None:
    packed = [1, 2, 3, 0, 0, 0, 1]
    np.testing.assert_allclose(
        Pose.create(packed, device="cpu").numpy(),
        [packed],
    )

    transforms = wp.array(
        [
            [1, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0, 0, 1],
        ],
        dtype=wp.transformf,
        device="cpu",
    )
    pose = Pose.create(transforms)

    assert pose.raw_pose is not transforms
    np.testing.assert_allclose(pose.numpy(), transforms.numpy())

    aliased_pose = Pose.create(transforms, copy=False)
    assert aliased_pose.raw_pose is transforms


def test_pose_create_copy_semantics() -> None:
    source = Pose([[1, 2, 3], [4, 5, 6]], device="cpu")

    copied = Pose.create(source)
    assert copied.raw_pose is not source.raw_pose
    np.testing.assert_allclose(copied.numpy(), source.numpy())
    copied.p = [7, 8, 9]
    np.testing.assert_allclose(source.p.numpy(), [[1, 2, 3], [4, 5, 6]])

    aliased = Pose.create(source, copy=False)
    assert aliased.raw_pose is source.raw_pose
    aliased.p = [9, 8, 7]
    np.testing.assert_allclose(source.p.numpy(), [[9, 8, 7], [9, 8, 7]])


def test_pose_multiplication() -> None:
    position = np.array([1, 2, 4], dtype=np.float32)
    orientation = _xyzw_from_euler(0, -0.3, 1)
    pose = Pose(position, orientation, device="cpu")
    expected = _transformation_matrix(position, orientation)
    _assert_pose_matrices(pose * pose, expected @ expected)

    positions = np.array([[1, 2, 3], [-2.5, 3, 0]], dtype=np.float32)
    orientations = np.stack(
        [
            _xyzw_from_euler(0, -0.3, 1),
            _xyzw_from_euler(0.9, 0.3, -1),
        ]
    )
    pose = Pose(positions, orientations, device="cpu")
    matrices = [
        _transformation_matrix(position, orientation)
        for position, orientation in zip(positions, orientations)
    ]
    _assert_pose_matrices(
        pose * pose,
        [matrix @ matrix for matrix in matrices],
    )


def test_pose_subtraction() -> None:
    reference = Pose(
        [[1, 0, 0], [0, 2, 0]],
        [
            _xyzw_from_euler(0, 0, np.pi / 2),
            _xyzw_from_euler(np.pi / 3, 0, 0),
        ],
        device="cpu",
    )
    local = Pose(
        [[2, 0, 0], [0, 0, 3]],
        [
            _xyzw_from_euler(0.1, 0.2, 0.3),
            _xyzw_from_euler(-0.2, 0.4, 0.1),
        ],
        device="cpu",
    )

    recovered = (reference * local) - reference

    np.testing.assert_allclose(
        recovered.to_transformation_matrix().numpy(),
        local.to_transformation_matrix().numpy(),
        atol=1e-5,
    )


def test_pose_inverse() -> None:
    positions = np.array([[1, 2, 3], [-2.5, 3, 0]], dtype=np.float32)
    orientations = np.stack(
        [
            _xyzw_from_euler(0, -0.3, 1),
            _xyzw_from_euler(0.9, 0.3, -1),
        ]
    )
    pose = Pose(positions, orientations, device="cpu")
    expected = [
        np.linalg.inv(_transformation_matrix(position, orientation))
        for position, orientation in zip(positions, orientations)
    ]

    _assert_pose_matrices(pose.inv(), expected)


def test_pose_transformation_matrix() -> None:
    positions = np.array([[1, 2, 4], [-2.5, 3, 0]], dtype=np.float32)
    orientations = np.stack(
        [
            _xyzw_from_euler(0, -0.3, 1),
            _xyzw_from_euler(0.9, 0.3, -1),
        ]
    )
    pose = Pose(positions, orientations, device="cpu")
    expected = [
        _transformation_matrix(position, orientation)
        for position, orientation in zip(positions, orientations)
    ]

    _assert_pose_matrices(pose, expected)


def test_pose_indexing_and_component_setters() -> None:
    pose = Pose([[1, 2, 3], [4, 5, 6]], device="cpu")

    assert pose[0].shape == (1,)
    assert pose[-1].shape == (1,)
    np.testing.assert_allclose(pose[1:].numpy(), pose.numpy()[1:])

    pose.p = [7, 8, 9]
    pose.set_q([0, 0, 1, 0])
    np.testing.assert_allclose(
        pose.numpy(),
        [[7, 8, 9, 0, 0, 1, 0], [7, 8, 9, 0, 0, 1, 0]],
    )


def test_pose_operators_require_equal_batch_sizes() -> None:
    batched = Pose([[1, 0, 0], [2, 0, 0]], device="cpu")
    singleton = Pose([1, 0, 0], device="cpu")

    with pytest.raises(ValueError, match="multiplication requires equal batch sizes"):
        _ = batched * singleton
    with pytest.raises(ValueError, match="subtraction requires equal batch sizes"):
        _ = batched - singleton
