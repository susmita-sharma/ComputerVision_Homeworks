# HW5 part 3 - structure from motion for a flat (planar) object seen from
# 4 viewpoints.
#
# ----------------------------------------------------------------------
# Camera model
# ----------------------------------------------------------------------
# A pinhole camera maps a 3D world point Xw = (X, Y, Z) to a homogeneous
# image point via
#     s * [x, y, 1]^T = K [R | t] [X, Y, Z, 1]^T = P * Xw_h
# K (3x3) holds the intrinsics (fx, fy, cx, cy); [R | t] (3x4) is the
# extrinsic pose - rotation + translation from world frame to that
# camera's frame. P = K[R|t] is the camera's 3x4 projection matrix.
#
# ----------------------------------------------------------------------
# Step A - recover each camera's pose (the "motion" in structure-from-motion)
# ----------------------------------------------------------------------
# The object is planar, so we get to *define* the world frame to sit on
# the object's own plane (Z = 0) and put its known real-world size on
# that plane. The 4 physical corners of a rectangle of width W, height H
# then have KNOWN world coordinates:
#     (0,0,0), (W,0,0), (W,H,0), (0,H,0)
# Clicking those same 4 corners in an image gives 4 image<->world
# correspondences, which is exactly the minimum PnP (Perspective-n-Point)
# needs to solve for that view's (R, t) given K:
#     solvePnP({Xw_corners}, {x_pixels}, K)  ->  R, t
# Do this once per photo -> 4 independent camera poses = the "motion".
#
# ----------------------------------------------------------------------
# Step B - triangulate the rest of the boundary (the "structure")
# ----------------------------------------------------------------------
# For any OTHER point on the object's boundary, its world position is
# NOT assumed known - it is recovered from where it lands in >= 2 views
# with now-known cameras. For view i with P_i, a 2D observation (x_i,y_i)
# of the SAME 3D point X satisfies (up to scale) x_i x (P_i X) = 0, which
# is linear in X and gives 2 independent equations per view:
#     x_i * P_i[2,:] - P_i[0,:]
#     y_i * P_i[2,:] - P_i[1,:]
# Stacking these rows from all views that see the point gives A X = 0;
# the least-squares solution is the right singular vector of A with the
# smallest singular value (linear/DLT triangulation). Doing this for
# every clicked boundary point across the 4 views recovers the object's
# 3D boundary.
#
# ----------------------------------------------------------------------
# Step C - sanity checks that make this a *validated* reconstruction
# ----------------------------------------------------------------------
# 1. Re-triangulate the 4 corners themselves (ignoring that we "know"
#    them) and compare to the ground-truth rectangle -> recovery error.
# 2. Fit a plane to every reconstructed point via SVD and report the RMS
#    distance to that plane -> should be small, since the true object is
#    flat. Both numbers are reported alongside the reconstruction.

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

PRIMARY = "#4f46e5"
ACCENT = "#f97316"
GRID_COLOR = "#e2e8f0"


def rectangle_world_corners(width_mm, height_mm):
    """World coords (Z=0 plane) of a rectangle's 4 corners, in
    TL, TR, BR, BL order - matches the click order used in the UI."""
    return np.array([
        [0.0, 0.0, 0.0],
        [width_mm, 0.0, 0.0],
        [width_mm, height_mm, 0.0],
        [0.0, height_mm, 0.0],
    ])


def default_intrinsics(image_w, image_h, fov_deg=60.0):
    """Rough camera matrix when no real calibration is available - a
    typical phone/webcam horizontal field of view of ~60 deg, principal
    point at the image center. Flagged in the UI as approximate; using a
    real HW2 calibration (scaled to this image's resolution) is preferred.
    """
    fx = fy = image_w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    K = np.array([
        [fx, 0, image_w / 2.0],
        [0, fy, image_h / 2.0],
        [0, 0, 1.0],
    ])
    return K


def scale_intrinsics(K, from_size, to_size):
    """Rescale a camera matrix calibrated at `from_size` (w, h) so it
    applies to images captured at `to_size` (w, h)."""
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def solve_camera_pose(image_corners, world_corners, K, dist_coeffs=None):
    dist = dist_coeffs if dist_coeffs is not None else np.zeros(5)
    obj = np.asarray(world_corners, dtype=np.float64)
    img = np.asarray(image_corners, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), bool(ok)


def projection_matrix(K, R, t):
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def camera_center(R, t):
    """World-frame position of the camera: where R*Xw + t = 0."""
    return -R.T @ t


def triangulate_point(P_list, pts2d_list):
    """Linear (DLT) least-squares triangulation of one 3D point from its
    2D observation in >= 2 views with known projection matrices."""
    rows = []
    for P, (x, y) in zip(P_list, pts2d_list):
        rows.append(x * P[2, :] - P[0, :])
        rows.append(y * P[2, :] - P[1, :])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


def fit_plane(points):
    """Best-fit plane through a point cloud via SVD; returns centroid,
    unit normal, and the RMS perpendicular distance (planarity residual).
    """
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    normal = normal / np.linalg.norm(normal)
    residuals = (pts - centroid) @ normal
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return centroid, normal, rms


def reconstruct(views, world_corners, K, dist_coeffs=None):
    """views: list of 4 dicts, each {"corners": [4 (x,y)], "boundary": [N (x,y)]}
    in the SAME point order across all views (same physical points).
    Returns per-camera poses, triangulated 3D boundary + corners, and the
    validation metrics described in the module docstring.
    """
    poses, P_list = [], []
    for v in views:
        R, t, ok = solve_camera_pose(v["corners"], world_corners, K, dist_coeffs)
        P = projection_matrix(K, R, t)
        poses.append({"R": R, "t": t, "C": camera_center(R, t), "ok": ok})
        P_list.append(P)

    n_boundary = len(views[0]["boundary"])
    boundary_3d = [
        triangulate_point(P_list, [v["boundary"][j] for v in views])
        for j in range(n_boundary)
    ]

    n_corners = len(world_corners)
    corner_3d = [
        triangulate_point(P_list, [v["corners"][j] for v in views])
        for j in range(n_corners)
    ]

    corner_recovery_error = [
        float(np.linalg.norm(np.array(corner_3d[j]) - np.array(world_corners[j])))
        for j in range(n_corners)
    ]

    all_points = boundary_3d + corner_3d
    centroid, normal, planarity_rms = fit_plane(all_points)

    return {
        "poses": poses,
        "P_list": P_list,
        "K": K,
        "boundary_3d": boundary_3d,
        "corner_3d": corner_3d,
        "corner_recovery_error": corner_recovery_error,
        "plane_centroid": centroid,
        "plane_normal": normal,
        "planarity_rms": planarity_rms,
    }


def _frustum(C, R, scale):
    """Apex + 4 base-corner points for a small pyramid drawn at camera
    center C, oriented by rotation R (world_to_cam), for the 3D plot."""
    cam_x = R.T @ np.array([1.0, 0, 0])
    cam_y = R.T @ np.array([0, 1.0, 0])
    cam_z = R.T @ np.array([0, 0, 1.0])  # camera's forward/viewing axis, in world coords
    base_center = C + cam_z * scale
    corners = [
        base_center + (sx * cam_x + sy * cam_y) * scale * 0.5
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ]
    return C, corners


def plot_scene(result, world_corners, out_path):
    """3D figure: reconstructed boundary polygon + the 4 recovered camera
    positions/viewing directions, so camera placement is visible at a
    glance next to the reconstructed shape."""
    boundary = np.array(result["boundary_3d"])
    corners = np.array(result["corner_3d"])
    world_corners = np.asarray(world_corners)

    extent = max(np.ptp(world_corners[:, 0]), np.ptp(world_corners[:, 1]), 1.0)
    cam_scale = extent * 0.5

    fig = plt.figure(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")

    poly = np.vstack([boundary, boundary[:1]])
    ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], "-o", color=PRIMARY, label="reconstructed boundary")

    wc = np.vstack([world_corners, world_corners[:1]])
    ax.plot(wc[:, 0], wc[:, 1], wc[:, 2], "--", color=GRID_COLOR, linewidth=1.5, label="known rectangle (ground truth)")
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], color=ACCENT, s=40, label="re-triangulated corners")

    for i, pose in enumerate(result["poses"]):
        apex, base = _frustum(pose["C"], pose["R"], cam_scale)
        for b in base:
            ax.plot([apex[0], b[0]], [apex[1], b[1]], [apex[2], b[2]], color="#334155", linewidth=1)
        base_loop = base + [base[0]]
        bx, by, bz = zip(*base_loop)
        ax.plot(bx, by, bz, color="#334155", linewidth=1)
        ax.scatter(*apex, color="#334155", s=25)
        ax.text(apex[0], apex[1], apex[2], f"  cam{i+1}", fontsize=9)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Recovered camera poses and reconstructed planar boundary", fontsize=12, fontweight="700")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_topdown(result, world_corners, out_path):
    """2D top-down (looking straight down the fitted plane normal) view:
    reconstructed boundary vs. the nominal rectangle, for a clean shape
    comparison independent of the 3D camera clutter."""
    normal = result["plane_normal"]
    centroid = result["plane_centroid"]

    ref = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u = ref - normal * np.dot(ref, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    def to_plane(points):
        pts = np.asarray(points) - centroid
        return np.stack([pts @ u, pts @ v], axis=-1)

    boundary_2d = to_plane(result["boundary_3d"])
    corners_2d = to_plane(result["corner_3d"])
    world_2d = to_plane(np.asarray(world_corners))

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    poly = np.vstack([boundary_2d, boundary_2d[:1]])
    ax.plot(poly[:, 0], poly[:, 1], "-o", color=PRIMARY, label="reconstructed boundary")
    wc = np.vstack([world_2d, world_2d[:1]])
    ax.plot(wc[:, 0], wc[:, 1], "--", color="#94a3b8", label="known rectangle")
    ax.scatter(corners_2d[:, 0], corners_2d[:, 1], color=ACCENT, s=40, label="re-triangulated corners")
    ax.set_aspect("equal")
    ax.set_xlabel("in-plane u (mm)")
    ax.set_ylabel("in-plane v (mm)")
    ax.set_title("Top-down view of reconstructed boundary", fontsize=12, fontweight="700")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
