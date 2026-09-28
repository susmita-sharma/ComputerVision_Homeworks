import cv2
import numpy as np

from Homework.hw5_motion_sfm import tracking


def _textured_image(size=200):
    """A patch with gradients in both directions (checkerboard-ish
    blobs), so the LK normal equations are well-conditioned everywhere
    we test - a flat region has no unique optical flow solution."""
    rng = np.random.default_rng(1)
    img = np.zeros((size, size), dtype=np.float64)
    for _ in range(40):
        cx, cy = rng.uniform(20, size - 20, size=2)
        r = rng.uniform(5, 15)
        yy, xx = np.mgrid[0:size, 0:size]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * 255
        img = np.maximum(img, blob)
    return img


def test_manual_lk_recovers_known_small_translation():
    img1 = _textured_image()
    # cv2.warpAffine with this M shifts the image CONTENT by (dx, dy)
    # (verified empirically: a point at (px,py) in img1 shows up at
    # (px+dx, py+dy) in img2), so a point tracked from img1 to img2
    # should be found displaced by exactly (dx, dy).
    dx, dy = 2.3, -1.6
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    img2 = cv2.warpAffine(img1.astype(np.float32), M, (img1.shape[1], img1.shape[0]),
                           flags=cv2.INTER_LINEAR).astype(np.float64)

    x0, y0 = 100.0, 100.0
    x_pred, y_pred, converged, _ = tracking.lucas_kanade_track_point(img1, img2, x0, y0, window=21, iterations=50)

    assert converged
    assert abs((x_pred - x0) - dx) < 0.1
    assert abs((y_pred - y0) - dy) < 0.1


def test_manual_tracker_agrees_with_opencv_reference():
    img1 = _textured_image().astype(np.uint8)
    dx, dy = 3.0, 1.5
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR)

    points = [(80.0, 90.0), (110.0, 120.0), (60.0, 140.0)]
    manual = tracking.track_points_manual(img1.astype(np.float64), img2.astype(np.float64), points, window=21, iterations=50)
    reference = tracking.track_points_reference(img1, img2, points, window=21)

    for m, r in zip(manual, reference):
        err = np.hypot(m["x_pred"] - r["x_actual"], m["y_pred"] - r["y_actual"])
        assert err < 0.5
