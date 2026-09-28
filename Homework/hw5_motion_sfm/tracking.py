# HW5 part 2 - deriving and then actually running the Lucas-Kanade point
# tracker between two consecutive frames.
#
# ----------------------------------------------------------------------
# 1. Brightness constancy
# ----------------------------------------------------------------------
# Assume a point that moves by (dx, dy) over a small time dt keeps its
# intensity:
#     I(x + dx, y + dy, t + dt) = I(x, y, t)
#
# First-order Taylor-expand the left side around (x, y, t):
#     I(x,y,t) + Ix*dx + Iy*dy + It*dt ~= I(x,y,t)
# which collapses to the *optical flow equation*
#     Ix*u + Iy*v + It = 0,   u = dx/dt, v = dy/dt
#
# One equation, two unknowns (u, v) per pixel - the aperture problem: a
# single pixel's gradient only constrains flow along the gradient
# direction, not across it.
#
# ----------------------------------------------------------------------
# 2. Lucas-Kanade: resolve the aperture problem with a local window
# ----------------------------------------------------------------------
# Assume (u, v) is constant over a small window W around the point (a
# few pixels, "the whole window moved together"). That turns the one
# equation above into one equation per pixel in the window - now an
# overdetermined system, solved by least squares:
#     minimize  sum_{(x,y) in W} (Ix u + Iy v + It)^2
# Differentiate w.r.t. (u, v) and set to zero:
#     [ sum Ix^2    sum IxIy ] [u]   [ -sum IxIt ]
#     [ sum IxIy    sum Iy^2 ] [v] = [ -sum IyIt ]
#           A^T A                      -A^T b
# i.e.  (A^T A) [u v]^T = -A^T b.  A^T A is invertible exactly when the
# window has gradients in at least two directions (a corner-like patch,
# not a flat region or a single edge) - the same structure-tensor
# condition Harris corner detection uses.
#
# ----------------------------------------------------------------------
# 3. From one frame pair to sub-pixel tracking
# ----------------------------------------------------------------------
# It above is a *finite* difference (frame2 - frame1), which is only a
# good linear approximation for small displacements. For a real
# displacement of several pixels, Newton's method is used: solve for a
# small correction, warp/re-sample frame2 at the updated guess using
# BILINEAR INTERPOLATION (frame2 has no pixel exactly at a fractional
# location), recompute It against the warped patch, and repeat until it
# converges. That warping step is exactly why bilinear.py exists.

import cv2
import numpy as np

from . import bilinear


def lucas_kanade_track_point(prev_gray, next_gray, x0, y0, window=15, iterations=40, eps=0.02):
    """From-scratch iterative Lucas-Kanade for ONE point, single scale.

    prev_gray/next_gray: float64 grayscale frames, same size.
    (x0, y0): point in prev_gray to track.
    Returns (x_new, y_new, converged, n_iters).
    """
    half = window // 2

    # Ix, Iy from the FIRST frame only - LK assumes the gradient near the
    # point doesn't change much over the small motion, so it's computed
    # once and reused every iteration (this is what makes it cheap).
    gx = cv2.Sobel(prev_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(prev_gray, cv2.CV_64F, 0, 1, ksize=3)

    Ix = bilinear.bilinear_sample_patch(gx, x0, y0, half)
    Iy = bilinear.bilinear_sample_patch(gy, x0, y0, half)
    I1 = bilinear.bilinear_sample_patch(prev_gray, x0, y0, half)

    Sxx = float(np.sum(Ix * Ix))
    Sxy = float(np.sum(Ix * Iy))
    Syy = float(np.sum(Iy * Iy))
    AtA = np.array([[Sxx, Sxy], [Sxy, Syy]])

    det = np.linalg.det(AtA)
    if abs(det) < 1e-6:
        # flat / featureless window - the aperture problem has no unique
        # answer here, so there is nothing trustworthy to return.
        return x0, y0, False, 0

    AtA_inv = np.linalg.inv(AtA)

    x, y = x0, y0
    for it in range(1, iterations + 1):
        I2 = bilinear.bilinear_sample_patch(next_gray, x, y, half)
        It = I2 - I1  # temporal derivative, sampled with the CURRENT warp

        b = np.array([-float(np.sum(Ix * It)), -float(np.sum(Iy * It))])
        duv = AtA_inv @ b

        x += duv[0]
        y += duv[1]

        if np.hypot(duv[0], duv[1]) < eps:
            return x, y, True, it

    return x, y, False, iterations


def track_points_manual(prev_gray, next_gray, points, window=15, iterations=40):
    """Run the from-scratch tracker over a list of (x, y) points."""
    prev_f = prev_gray.astype(np.float64)
    next_f = next_gray.astype(np.float64)
    results = []
    for (x0, y0) in points:
        x, y, converged, n_it = lucas_kanade_track_point(prev_f, next_f, x0, y0, window, iterations)
        results.append({
            "x0": x0, "y0": y0,
            "x_pred": x, "y_pred": y,
            "dx": x - x0, "dy": y - y0,
            "converged": converged, "n_iters": n_it,
        })
    return results


def track_points_reference(prev_gray, next_gray, points, window=15):
    """OpenCV's production pyramidal LK tracker, used here as the
    reference "actual" pixel location - i.e. the independently-implemented,
    heavily-optimized version of the exact same math (multi-scale, same
    brightness-constancy model) - to validate the from-scratch tracker
    against something other than itself.
    """
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, next_gray, pts, None,
        winSize=(window, window), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    out = []
    for i in range(len(points)):
        out.append({
            "x_actual": float(next_pts[i, 0, 0]),
            "y_actual": float(next_pts[i, 0, 1]),
            "found": bool(status[i, 0] == 1),
            "err": float(err[i, 0]),
        })
    return out


def validate_tracking(prev_bgr, next_bgr, points, window=15, iterations=40):
    """Run both trackers on the same points and report pixel-level
    agreement between the derived math (manual) and OpenCV's reference
    implementation (actual) - the "validate the theoretical result with
    actual pixel locations" step.
    """
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)

    manual = track_points_manual(prev_gray, next_gray, points, window, iterations)
    reference = track_points_reference(prev_gray, next_gray, points, window=max(window, 15))

    rows = []
    for m, r in zip(manual, reference):
        err_px = float(np.hypot(m["x_pred"] - r["x_actual"], m["y_pred"] - r["y_actual"]))
        rows.append({**m, **r, "agreement_err_px": err_px})
    return rows
