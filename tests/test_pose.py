import pytest
from emv.features.pose_utils import compute_hips_angles, filter_poses
import numpy as np
import pandas as pd


def assert_almost_equal(arr1, arr2, tol=1e-6):
    assert np.allclose(
        arr1, arr2, atol=tol), f"Arrays not almost equal: total diff: {np.sum(np.abs(arr1 - arr2))}"


def test_basic_pose():
    # testing basic positions in the pose. This covers basic cases like isocoles triangle, right angle triangle, etc.
    keypoints = [
        [1, 3], [3, 3], [1, 0], [3, 0], [5, 2], [0, 2], [2, 2],
        [1, 1], [3, 1],  # idx 7, 8 are the hips and therefore the reference keypoints
        [2, 2], [2, 2], [100, 2], [2, 100]
    ]
    keypoints = [k + [1] for k in keypoints] # Add confidence score to each keypoint

    expected = [
        0.50, 0.25, 0.25,
        0.25, 0.25, 0.50,
        0.50, 0.35, 0.15,
        0.15, 0.35, 0.50,
        0.08, 0.07, 0.85,
        0.75, 0.15, 0.10,
        0.25, 0.50, 0.25,
        0.25, 0.50, 0.25,
        0.25, 0.50, 0.25,
        0.00, 0.00, 1.00,
        0.50, 0.01, 0.50,
    ]
    angles, angles_scores = compute_hips_angles(keypoints)
    # round result_1 to .2f
    angles = np.round(angles, 2)
    assert_almost_equal(angles, np.array(expected))


def test_missing_values():
    keypoints = [
        [1, 3], [3, 3], [1, 0], [3, 0], [5, 2], [0, 2], [2, 2]
    ]
    keypoints = [k + [1] for k in keypoints] # Add confidence score to each keypoint

    with pytest.raises(IndexError):
        compute_hips_angles(keypoints)


def test_invalid_keypoints():
    # points can't be the same as the reference points
    keypoints = [
        [1, 1], [3, 1], [1, 0], [3, 0], [5, 2], [0, 2], [2, 2],
        [1, 1], [3, 1],  # idx 7, 8 are the hips and therefore the reference keypoints
        [2, 2], [2, 2], [100, 2], [2, 100]
    ]
    keypoints = [k + [1] for k in keypoints] # Add confidence score to each keypoint

    with pytest.raises(ValueError):
        compute_hips_angles(keypoints)


def test_filter_poses():
    df = pd.DataFrame({
        'angle_vec': np.array([[0.9, 1], [1, 1], [1.1, 1.1], [1, 2], [1, 3], [1, 4], [1, 4.1]]).tolist(),
        'id': [0, 1, 2, 3, 4, 5, 6]
    })

    result = filter_poses(df, threshold=1)
    assert set([0, 3, 5]) == set(result.id.tolist())
