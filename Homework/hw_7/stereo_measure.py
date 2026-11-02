# HW7 part 1 - calibrated stereo object measurement.
#
# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
# Two photos of the same scene from cameras with intrinsics K, separated by
# a measured baseline B. After rectification both image planes are coplanar
# and epipolar lines are horizontal, so a 3D point lands on the same row in
# both images and differs only in column by the disparity d:
#     d = xL - xR - doff        doff = cxL - cxR after rectification
#     Z = f B / d
#     X = (x - cx) Z / f        Y = (y - cy) Z / f
# Sizes are distances between those 3D points, never pixel lengths.
#
# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
# 1. SIFT matches, kept if they shift sideways -> essential matrix -> (R, t
#    direction) between the views, unless the pair is declared rectified.
# 2. |t| is set to the measured baseline, which is what makes the result
#    metric: E alone only gives the direction of t.
# 3. cv2.stereoRectify + remap -> rectified pair; the disparity search range
#    comes from the matched features, not from a fixed guess.
# 4. StereoSGBM -> dense disparity -> depth map.
# 5. A clicked point is matched along its epipolar row by NCC with a parabola
#    fit. SGBM is smoothed and weakest on object edges, which is exactly
#    where object corners sit, so it is only the fallback.

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK_MAX_SIDE = 1400   # processing resolution; K is scaled to match
MIN_INLIERS = 30
PATCH_HALF = 10
SIDEWAYS_SLOPE = 0.15  # largest row change per column of shift for a sideways match
MIN_SHIFT_PX = 2.0
LOW_NCC = 0.6
GOOD_NCC = 0.8
OUTLIER_MIN_PX = 8.0
OUTLIER_RATIO = 0.15


class StereoError(ValueError):
    """A problem with the inputs the user can act on; shown as-is in the UI."""


def adapt_calibration(K_cal, cal_size, image_size):
    """Calibration K for images of another size from the same camera. Handles
    the other orientation (x/y swapped) and another aspect ratio, which on a
    phone is a centre crop (4:3 sensor -> 16:9 video), not a stretch."""
    K = np.asarray(K_cal, dtype=np.float64).copy()
    cw, ch = float(cal_size[0]), float(cal_size[1])
    w, h = float(image_size[0]), float(image_size[1])
    swapped = (cw > ch) != (w > h)
    if swapped:
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1.0]])
        cw, ch = ch, cw
    aspect = w / h
    crop_w, crop_h = (cw, cw / aspect) if cw / ch < aspect else (ch * aspect, ch)
    K[0, 2] -= (cw - crop_w) / 2.0
    K[1, 2] -= (ch - crop_h) / 2.0
    s = w / crop_w
    K[:2] *= s
    return K, swapped


def resize_pair(left, right, K):
    """Downscale both images to the working size, scaling K to match."""
    if left.shape[:2] != right.shape[:2]:
        raise StereoError(f"Left and right images must be the same size (got {left.shape[1]}x{left.shape[0]} "
                          f"and {right.shape[1]}x{right.shape[0]}). Use the same camera and orientation.")
    s = min(1.0, WORK_MAX_SIDE / max(left.shape[:2]))
    if s < 1.0:
        size = (round(left.shape[1] * s), round(left.shape[0] * s))
        K = K.copy()
        K[0] *= size[0] / left.shape[1]
        K[1] *= size[1] / left.shape[0]
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    return left, right, K


def match_features(gray1, gray2, ratio=0.75):
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(gray1, None)
    k2, d2 = sift.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    pts1 = np.float64([k1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float64([k2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts1, pts2


def sideways_matches(pts1, pts2):
    """Keep matches that shift along the row, right-to-left as a left/right
    pair must. Repeating patterns such as blinds produce many false matches
    one slat up or down, and those can outvote the real ones in RANSAC and
    yield a confident but wrong vertical camera motion."""
    flow = pts2 - pts1
    along_row = np.abs(flow[:, 1]) < SIDEWAYS_SLOPE * np.abs(flow[:, 0])
    leftward = along_row & (flow[:, 0] < -MIN_SHIFT_PX)
    rightward = along_row & (flow[:, 0] > MIN_SHIFT_PX)
    if rightward.sum() > leftward.sum():
        raise StereoError("Scene points move to the right from the first image to the second, so the images look "
                          "swapped. Upload the image from the left position as the left image.")
    if leftward.sum() < MIN_INLIERS:
        raise StereoError(f"Only {int(leftward.sum())} matches shift sideways between the images (need {MIN_INLIERS}). "
                          f"The camera has to move sideways, and the scene needs more texture.")
    return pts1[leftward], pts2[leftward]


def relative_pose(pts1, pts2, K, baseline_mm):
    """(R, T) taking left-camera points to right-camera points, with |T| set
    to the measured baseline, plus the inlier mask."""
    if len(pts1) < MIN_INLIERS:
        raise StereoError(f"Only {len(pts1)} feature matches between the images (need {MIN_INLIERS}). "
                          f"Check that both show the same scene with enough texture.")
    pts1, pts2 = sideways_matches(pts1, pts2)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        raise StereoError("Could not estimate the camera geometry from these images (no unique essential "
                          "matrix). Add more depth variation or texture, or use the rectified-pair option.")
    n, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    inliers = mask.ravel() > 0
    if inliers.sum() < MIN_INLIERS:
        raise StereoError(f"Only {int(inliers.sum())} matches agree on one camera geometry (need {MIN_INLIERS}). "
                          f"The images may show different scenes or too small a camera movement.")
    return R, t.ravel() * baseline_mm, inliers


def rectify(left, right, K, R, T):
    size = (left.shape[1], left.shape[0])
    d = np.zeros(5)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, d, K, d, size, R, T.reshape(3, 1), flags=0, alpha=0)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        raise StereoError("The camera moved mostly vertically between the images. Move it sideways, "
                          "left image from the left position, right image from the right position.")
    if P2[0, 3] > 0:
        raise StereoError("The camera moved to the left between these images, so they look swapped. "
                          "Upload the image from the left position as the left image.")
    m1 = cv2.initUndistortRectifyMap(K, d, R1, P1, size, cv2.CV_32FC2)
    m2 = cv2.initUndistortRectifyMap(K, d, R2, P2, size, cv2.CV_32FC2)
    rect_l = cv2.remap(left, m1[0], m1[1], cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, m2[0], m2[1], cv2.INTER_LINEAR)
    return rect_l, rect_r, {"f": float(P1[0, 0]), "cx": float(P1[0, 2]), "cy": float(P1[1, 2]),
                            "doff": float(P1[0, 2] - P2[0, 2]), "baseline": float(-P2[0, 3] / P2[0, 0])}, R1


def disparity_range(rect_l, rect_r):
    """Column-shift range of the scene, from features on the rectified pair.
    Returns (lo, hi, n matches on one row, rows spread)."""
    gl = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    p1, p2 = match_features(gl, gr)
    keep = np.abs(p1[:, 1] - p2[:, 1]) < 3.0
    if keep.sum() < MIN_INLIERS:
        raise StereoError(f"After rectification only {int(keep.sum())} matches share a row (need {MIN_INLIERS}), so "
                          f"the images are not a usable stereo pair. Check the calibration and the capture.")
    d = (p1 - p2)[keep, 0]
    return (float(np.percentile(d, 1)), float(np.percentile(d, 99)), int(keep.sum()),
            float(np.median(np.abs(p1[keep, 1] - p2[keep, 1]))))


def compute_disparity(rect_l, rect_r, lo, hi):
    """SGBM over the measured range. Disparity is returned as raw column
    shift (xL - xR) with NaN where SGBM found no match."""
    margin = max(16.0, 0.15 * (hi - lo))
    min_d = int(np.floor(lo - margin))
    num = int(np.ceil((hi + margin - min_d) / 16.0)) * 16
    num = min(max(num, 32), 512)
    block = 7
    sgbm = cv2.StereoSGBM_create(minDisparity=min_d, numDisparities=num, blockSize=block,
                                 P1=8 * 3 * block ** 2, P2=32 * 3 * block ** 2, disp12MaxDiff=2,
                                 uniquenessRatio=8, speckleWindowSize=120, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    raw = sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0
    raw[raw < min_d] = np.nan
    return raw


def depth_from_disparity(raw, geom):
    d = raw - geom["doff"]
    Z = np.full(raw.shape, np.nan, np.float32)
    ok = np.isfinite(d) & (d > 0.5)
    Z[ok] = geom["f"] * geom["baseline"] / d[ok]
    return Z


def match_on_row(rect_l, rect_r, x, y, seed, search):
    """NCC along row y of the right image around xR = x - seed. Returns the raw
    disparity (sub-pixel) and the NCC score, or (None, score) when the patch
    has no usable match."""
    gl = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    h, w = gl.shape
    xi, yi, p = int(round(x)), int(round(y)), PATCH_HALF
    if not (p <= xi < w - p and p <= yi < h - p):
        return None, 0.0
    tmpl = gl[yi - p:yi + p + 1, xi - p:xi + p + 1]
    if tmpl.std() < 2.0:
        return None, 0.0
    x_lo = max(p, int(np.floor(x - seed - search)))
    x_hi = min(w - p - 1, int(np.ceil(x - seed + search)))
    if x_hi - x_lo < 2:
        return None, 0.0
    strip = gr[yi - p:yi + p + 1, x_lo - p:x_hi + p + 1]
    if strip.std() < 2.0:
        return None, 0.0
    score = cv2.matchTemplate(strip, tmpl, cv2.TM_CCOEFF_NORMED)[0]
    k = int(np.argmax(score))
    sub = 0.0
    if 0 < k < len(score) - 1:
        a, b, c = score[k - 1], score[k], score[k + 1]
        if a - 2 * b + c < 0:
            sub = float(0.5 * (a - c) / (a - 2 * b + c))
    return float(x - (x_lo + k + sub)), float(score[k])


def point_3d(x, y, raw_d, geom):
    d = raw_d - geom["doff"]
    if d <= 0:
        raise StereoError(f"The point at ({x:.0f}, {y:.0f}) has no positive disparity, so its depth is undefined. "
                          f"Adjust its match on the right image.")
    Z = geom["f"] * geom["baseline"] / d
    return np.array([(x - geom["cx"]) * Z / geom["f"], (y - geom["cy"]) * Z / geom["f"], Z])


def disparity_outliers(disparities):
    """Indices far from the shape's median disparity. These are usually a
    click matched onto the background through a confident but wrong patch."""
    d = np.asarray(disparities, dtype=np.float64)
    med = float(np.median(d))
    return [i for i, v in enumerate(d) if abs(v - med) > max(OUTLIER_MIN_PX, OUTLIER_RATIO * abs(med))]


def fit_circle_3d(pts):
    """Best plane through the rim points, then a least-squares circle in that
    plane. Works for a tilted circle, which appears as an ellipse in the image.
    Returns centre, diameter and the RMS rim residual."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c)
    u, v = Vt[0], Vt[1]
    a = (pts - c) @ u
    b = (pts - c) @ v
    A = np.column_stack([2 * a, 2 * b, np.ones(len(a))])
    if np.linalg.matrix_rank(A) < 3:
        raise StereoError("The rim points are collinear in 3D, so no circle fits them. "
                          "Click points spread around the rim.")
    sol = np.linalg.lstsq(A, a ** 2 + b ** 2, rcond=None)[0]
    r = float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))
    residual = float(np.sqrt(np.mean((np.hypot(a - sol[0], b - sol[1]) - r) ** 2)))
    return c + sol[0] * u + sol[1] * v, 2.0 * r, residual


def measure_shape(kind, pts):
    """Real-world dimensions from a shape's 3D points (mm)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if kind == "rectangle":
        if n != 4:
            raise StereoError("A rectangle needs exactly 4 corners, clicked in order around it.")
        e = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        return {"edges": e, "items": [{"label": "Width (mean of edges 1-2 and 3-4)", "mm": (e[0] + e[2]) / 2},
                                      {"label": "Height (mean of edges 2-3 and 4-1)", "mm": (e[1] + e[3]) / 2}]}
    if kind == "circle":
        if n < 3:
            raise StereoError("A circle needs at least 3 points on its rim.")
        centre, diameter, residual = fit_circle_3d(pts)
        return {"edges": [], "centre": centre.tolist(), "items": [{"label": "Diameter", "mm": diameter}],
                "residual": residual}
    if kind == "polygon":
        if n < 3:
            raise StereoError("A polygon needs at least 3 vertices.")
        e = [float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n)]
        return {"edges": e, "items": [{"label": f"Edge {i + 1}-{(i + 1) % n + 1}", "mm": v} for i, v in enumerate(e)]
                                     + [{"label": "Perimeter", "mm": sum(e)}]}
    raise StereoError(f"Unknown shape type: {kind}")


def run_stereo(left, right, K, dist, baseline_mm, already_rectified):
    """Full pair -> rectified images, disparity and depth, in millimetres."""
    left, right, K = resize_pair(left, right, K)
    if dist is not None:
        left = cv2.undistort(left, K, dist)
        right = cv2.undistort(right, K, dist)
    n_matches = n_inliers = None
    if already_rectified:
        R, T = np.eye(3), np.array([-float(baseline_mm), 0.0, 0.0])
    else:
        p1, p2 = match_features(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
        R, T, inliers = relative_pose(p1, p2, K, baseline_mm)
        n_matches, n_inliers = len(p1), int(inliers.sum())
    rect_l, rect_r, geom, R1 = rectify(left, right, K, R, T)
    lo, hi, n_rows, row_err = disparity_range(rect_l, rect_r)
    raw = compute_disparity(rect_l, rect_r, lo, hi)
    if np.isfinite(raw).mean() < 0.02:
        raise StereoError("Dense stereo matched almost nothing. The images may lack texture, "
                          "or the camera geometry may be off.")
    rotation_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    return {"left": left, "right": right, "rect_l": rect_l, "rect_r": rect_r, "raw": raw, "geom": geom,
            "depth": depth_from_disparity(raw, geom), "K": K, "T": T, "R1": R1, "rotation_deg": rotation_deg,
            "lo": lo, "hi": hi, "n_matches": n_matches, "n_inliers": n_inliers, "n_rows": n_rows, "row_err": row_err}


def locate_point(rect_l, rect_r, raw, x, y, lo, hi):
    """Right-image match for a left-image click. Returns raw disparity, NCC
    and which method won; (None, ...) means the user must place it by hand.

    Object corners sit on a depth edge, where a centred patch is mostly
    background and matches the background. So patches shifted around the click
    are tried too, one of which lies wholly on the object. The click itself is
    on the outline, and an outline belongs to the nearer surface, so among
    confident matches the largest disparity wins."""
    margin = max(16.0, 0.15 * (hi - lo))
    centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0 + margin
    found = []
    for ox in (0, -PATCH_HALF, PATCH_HALF):
        for oy in (0, -PATCH_HALF, PATCH_HALF):
            d, score = match_on_row(rect_l, rect_r, x + ox, y + oy, centre, half)
            if d is not None:
                found.append((d, score))
    best = max((score for _, score in found), default=0.0)
    if best >= LOW_NCC:
        good = [f for f in found if f[1] >= min(best, GOOD_NCC)]
        d, score = max(good)
        return d, score, "ncc"

    xi, yi = int(round(x)), int(round(y))
    h, w = raw.shape
    patch = raw[max(0, yi - 3):min(h, yi + 4), max(0, xi - 3):min(w, xi + 4)]
    patch = patch[np.isfinite(patch)]
    if patch.size:
        return float(np.median(patch)), best, "sgbm"
    return None, best, "none"


def save_map_figure(values, title, unit, path, cmap="turbo"):
    ok = values[np.isfinite(values)]
    lo, hi = (np.percentile(ok, [2, 98]) if ok.size else (0.0, 1.0))
    h, w = values.shape
    fig, ax = plt.subplots(figsize=(7, 7 * h / w + 0.5), dpi=110)
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=unit)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
