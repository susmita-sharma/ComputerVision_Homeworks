import cv2
import numpy as np

from Homework.hw5_motion_sfm import sfm_planar


def _look_at_pose(C, target, up=(0.0, 1.0, 0.0)):
    C = np.asarray(C, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - C
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)

    R = np.vstack([right, down, forward])
    t = -R @ C
    return R, t


def test_sfm_recovers_known_planar_scene():
    width_mm, height_mm = 200.0, 140.0
    world_corners = sfm_planar.rectangle_world_corners(width_mm, height_mm)
    centroid = world_corners.mean(axis=0)

    # a handful of extra boundary points along the rectangle's edges,
    # in world coords - "ground truth" we'll try to recover blind
    boundary_world = np.array([
        [width_mm / 2, 0.0, 0.0],
        [width_mm, height_mm / 2, 0.0],
        [width_mm / 2, height_mm, 0.0],
        [0.0, height_mm / 2, 0.0],
        [width_mm * 0.25, height_mm * 0.25, 0.0],
    ])

    K = np.array([[900.0, 0, 320.0], [0, 900.0, 240.0], [0, 0, 1.0]])
    image_size = (640, 480)

    camera_centers = [
        (centroid[0] - 300, centroid[1] - 100, 700),
        (centroid[0] + 350, centroid[1] + 50, 650),
        (centroid[0], centroid[1] + 400, 900),
        (centroid[0] - 200, centroid[1] + 300, 500),
    ]

    views = []
    true_poses = []
    for C in camera_centers:
        R, t = _look_at_pose(C, centroid)
        rvec, _ = cv2.Rodrigues(R)
        corners_2d, _ = cv2.projectPoints(world_corners, rvec, t, K, None)
        boundary_2d, _ = cv2.projectPoints(boundary_world, rvec, t, K, None)
        corners_2d = corners_2d.reshape(-1, 2)
        boundary_2d = boundary_2d.reshape(-1, 2)

        assert np.all(corners_2d[:, 0] > 0) and np.all(corners_2d[:, 0] < image_size[0])
        assert np.all(corners_2d[:, 1] > 0) and np.all(corners_2d[:, 1] < image_size[1])

        views.append({
            "corners": [tuple(p) for p in corners_2d],
            "boundary": [tuple(p) for p in boundary_2d],
        })
        true_poses.append((R, np.asarray(C)))

    result = sfm_planar.reconstruct(views, world_corners, K)

    for pose, (R_true, C_true) in zip(result["poses"], true_poses):
        assert np.linalg.norm(pose["C"] - C_true) < 1e-2
        assert np.allclose(pose["R"], R_true, atol=1e-3)

    for j, X in enumerate(result["boundary_3d"]):
        assert np.linalg.norm(np.array(X) - boundary_world[j]) < 1e-2

    assert max(result["corner_recovery_error"]) < 1e-2
    assert result["planarity_rms"] < 1e-2
