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
#
# Displacements bigger than about half the window break the linearization
# entirely, so the same iteration is run coarse-to-fine over an image
# pyramid: solve at 1/8 resolution, double the estimate, refine at 1/4,
# ... down to full resolution.

import cv2
import numpy as np

from . import bilinear


def image_gradients(gray):
    """Ix, Iy by central-difference Sobel. A 3x3 Sobel kernel's weights
    sum to 8 per unit step, so dividing by 8 gives true grey-levels/px -
    without it every LK update would come out 8x too small."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    return gx, gy


def lucas_kanade_track_point(prev_gray, next_gray, x0, y0, window=15, iterations=40, eps=0.01,
                             guess=(0.0, 0.0), grads=None, log=None):
    """From-scratch iterative Lucas-Kanade for ONE point, single scale.

    prev_gray/next_gray: float64 grayscale frames, same size.
    (x0, y0): point in prev_gray to track; `guess` = initial displacement.
    grads: optional precomputed (Ix, Iy) of prev_gray.
    log: optional dict that receives the normal-equation numbers and the
         per-iteration history (used for the worked example on the page).
    Returns (x_new, y_new, converged, n_iters).
    """
    half = window // 2
    gx, gy = grads if grads is not None else image_gradients(prev_gray)

    # Ix, Iy from the FIRST frame only - LK assumes the gradient near the
    # point doesn't change much over the small motion, so it's computed
    # once and reused every iteration (this is what makes it cheap).
    Ix = bilinear.bilinear_sample_patch(gx, x0, y0, half)
    Iy = bilinear.bilinear_sample_patch(gy, x0, y0, half)
    I1 = bilinear.bilinear_sample_patch(prev_gray, x0, y0, half)

    Sxx = float(np.sum(Ix * Ix))
    Sxy = float(np.sum(Ix * Iy))
    Syy = float(np.sum(Iy * Iy))
    AtA = np.array([[Sxx, Sxy], [Sxy, Syy]])
    eig = np.linalg.eigvalsh(AtA)
    if log is not None:
        log.update({"Sxx": Sxx, "Sxy": Sxy, "Syy": Syy, "eig_min": float(eig[0]), "eig_max": float(eig[1]),
                    "n_pixels": int(Ix.size), "iters": []})

    # flat / featureless window (or a single straight edge): the smaller
    # eigenvalue is ~0, the aperture problem has no unique answer, so
    # nothing trustworthy to return. Threshold: mean squared gradient
    # along the weakest direction below 0.5 (grey levels/px)^2.
    if eig[0] < 0.5 * Ix.size:
        return x0 + guess[0], y0 + guess[1], False, 0

    AtA_inv = np.linalg.inv(AtA)

    x, y = x0 + guess[0], y0 + guess[1]
    for it in range(1, iterations + 1):
        I2 = bilinear.bilinear_sample_patch(next_gray, x, y, half)
        It = I2 - I1  # temporal derivative, sampled with the CURRENT warp

        b = np.array([-float(np.sum(Ix * It)), -float(np.sum(Iy * It))])
        duv = AtA_inv @ b

        if log is not None:
            log["iters"].append({"it": it, "x": x, "y": y, "SxIt": -b[0], "SyIt": -b[1],
                                 "du": float(duv[0]), "dv": float(duv[1]),
                                 "rms_residual": float(np.sqrt(np.mean(It ** 2)))})
        x += duv[0]
        y += duv[1]

        if np.hypot(duv[0], duv[1]) < eps:
            return x, y, True, it

    return x, y, False, iterations


def build_pyramid(gray, levels):
    pyr = [gray]
    for _ in range(levels):
        if min(pyr[-1].shape) < 40:
            break
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def track_points_manual(prev_gray, next_gray, points, window=15, iterations=40, levels=3, logs=None):
    """Run the from-scratch tracker over a list of (x, y) points,
    coarse-to-fine over a `levels`-deep pyramid."""
    prev_pyr = build_pyramid(np.asarray(prev_gray, dtype=np.float64), levels)
    next_pyr = build_pyramid(np.asarray(next_gray, dtype=np.float64), levels)
    grads = [image_gradients(p) for p in prev_pyr]
    top = len(prev_pyr) - 1

    results = []
    for k, (x0, y0) in enumerate(points):
        g = np.zeros(2)
        level_log = []
        converged, n_it = False, 0
        for L in range(top, -1, -1):
            s = 2.0 ** L
            lg = {} if logs is not None else None
            xL, yL = x0 / s, y0 / s
            x, y, converged, n_it = lucas_kanade_track_point(prev_pyr[L], next_pyr[L], xL, yL, window, iterations,
                                                             guess=tuple(g), grads=grads[L], log=lg)
            d = np.array([x - xL, y - yL])
            if lg is not None:
                lg.update({"level": L, "scale": s, "guess_in": g.tolist(), "d_out": d.tolist(),
                           "converged": converged, "n_iters": n_it})
                level_log.append(lg)
            g = d * 2.0 if L > 0 else d
        if logs is not None:
            logs.append(level_log)
        results.append({
            "x0": x0, "y0": y0,
            "x_pred": x0 + g[0], "y_pred": y0 + g[1],
            "dx": float(g[0]), "dy": float(g[1]),
            "converged": converged, "n_iters": n_it,
        })
    return results


def track_points_reference(prev_gray, next_gray, points, window=15):
    """OpenCV's production pyramidal LK tracker - an independent,
    optimized implementation of the same math, as a second reference."""
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


def match_template_location(prev_gray, next_gray, x0, y0, half=10, search=48):
    """'Actual' pixel location measured WITHOUT any optical-flow math:
    cut a (2*half+1)^2 template around (x0, y0) in frame t, slide it over
    a +-search px region of frame t+1, take the best normalized
    cross-correlation, then refine to sub-pixel by fitting a parabola
    through the score peak and its neighbours in x and in y.
    Returns (x, y, ncc_score)."""
    H, W = next_gray.shape
    tmpl = bilinear.bilinear_sample_patch(prev_gray, x0, y0, half).astype(np.float32)
    cx, cy = int(round(x0)), int(round(y0))
    rx0 = max(0, cx - search - half)
    ry0 = max(0, cy - search - half)
    rx1 = min(W, cx + search + half + 1)
    ry1 = min(H, cy + search + half + 1)
    region = np.asarray(next_gray[ry0:ry1, rx0:rx1], dtype=np.float32)
    if region.shape[0] <= tmpl.shape[0] or region.shape[1] <= tmpl.shape[1]:
        return float(x0), float(y0), 0.0
    res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(res)

    def parabola(a, b, c):
        den = a - 2 * b + c
        return 0.0 if abs(den) < 1e-9 else 0.5 * (a - c) / den

    sx = parabola(res[my, mx - 1], res[my, mx], res[my, mx + 1]) if 0 < mx < res.shape[1] - 1 else 0.0
    sy = parabola(res[my - 1, mx], res[my, mx], res[my + 1, mx]) if 0 < my < res.shape[0] - 1 else 0.0
    return rx0 + mx + half + sx, ry0 + my + half + sy, float(score)


def brightness_residual(prev_gray, next_gray, x0, y0, x1, y1, half=10):
    """RMS of I2(window moved to (x1,y1)) - I1(window at (x0,y0)) - how
    well brightness constancy holds for that displacement."""
    a = bilinear.bilinear_sample_patch(prev_gray, x0, y0, half)
    b = bilinear.bilinear_sample_patch(next_gray, x1, y1, half)
    return float(np.sqrt(np.mean((b - a) ** 2)))


def auto_pick_points(prev_gray, next_gray, n=10):
    """Strong Shi-Tomasi corners, preferring ones that actually move
    between the two frames (half moving, half static where possible)."""
    h, w = prev_gray.shape
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01,
                                      minDistance=max(10, min(h, w) // 25))
    if corners is None:
        return []
    corners = corners.reshape(-1, 2)
    margin = 40
    keep = (corners[:, 0] > margin) & (corners[:, 0] < w - margin) & (corners[:, 1] > margin) & (corners[:, 1] < h - margin)
    corners = corners[keep]
    if len(corners) == 0:
        return []
    ref = track_points_reference(prev_gray, next_gray, [tuple(c) for c in corners], window=21)
    disp = np.array([np.hypot(r["x_actual"] - c[0], r["y_actual"] - c[1]) if r["found"] else -1
                     for r, c in zip(ref, corners)])
    moving = [i for i in np.argsort(-disp) if disp[i] > 0.5][: n // 2]
    rest = [i for i in range(len(corners)) if i not in moving and disp[i] >= 0][: n - len(moving)]
    return [(float(corners[i][0]), float(corners[i][1])) for i in moving + rest]


def validate_tracking(prev_bgr, next_bgr, points, window=21, iterations=40, levels=3, logs=None):
    """Run the from-scratch tracker and compare its predicted locations
    with (a) independently measured template-matching locations and
    (b) OpenCV's pyramidal LK - the "validate the theoretical result with
    actual pixel locations" step. Also reports how much the predicted
    displacement reduces the brightness-constancy residual."""
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    pf = prev_gray.astype(np.float64)
    nf = next_gray.astype(np.float64)

    manual = track_points_manual(pf, nf, points, window, iterations, levels, logs=logs)
    reference = track_points_reference(prev_gray, next_gray, points, window=window)

    rows = []
    for m, r in zip(manual, reference):
        xm, ym, ncc = match_template_location(pf, nf, m["x0"], m["y0"], half=window // 2)
        rows.append({
            **m, **r,
            "x_match": xm, "y_match": ym, "ncc": ncc,
            "err_vs_match_px": float(np.hypot(m["x_pred"] - xm, m["y_pred"] - ym)),
            "agreement_err_px": float(np.hypot(m["x_pred"] - r["x_actual"], m["y_pred"] - r["y_actual"])),
            "rms_no_motion": brightness_residual(pf, nf, m["x0"], m["y0"], m["x0"], m["y0"], window // 2),
            "rms_tracked": brightness_residual(pf, nf, m["x0"], m["y0"], m["x_pred"], m["y_pred"], window // 2),
        })
    return rows


def bilinear_worked_example(gray, x, y):
    """All the numbers of one bilinear lookup, for the page/report."""
    g = np.asarray(gray, dtype=np.float64)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    wx, wy = x - x0, y - y0
    Ia, Ib, Ic, Id = g[y0, x0], g[y0, x0 + 1], g[y0 + 1, x0], g[y0 + 1, x0 + 1]
    R0 = (1 - wx) * Ia + wx * Ib
    R1 = (1 - wx) * Ic + wx * Id
    val = (1 - wy) * R0 + wy * R1
    remap = float(cv2.remap(g.astype(np.float32), np.float32([[x]]), np.float32([[y]]), cv2.INTER_LINEAR)[0, 0])
    return {"x": x, "y": y, "x0": x0, "y0": y0, "wx": wx, "wy": wy, "Ia": Ia, "Ib": Ib, "Ic": Ic, "Id": Id,
            "R0": R0, "R1": R1, "value": val, "ours": float(bilinear.bilinear_sample(g, x, y)),
            "cv2_remap": remap, "nearest": float(g[int(round(y)), int(round(x))])}
