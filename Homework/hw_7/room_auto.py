# HW7 part 3 - reading the room off the video with no clicks.
#
# Plain walls carry few features, but a room's edges are long straight lines,
# so this works from those (line segment detector):
#
# Focal length: phone video is cropped by stabilisation, so a photo calibration
#   runs short. Where a frame sees a corner, the front wall's floor and ceiling
#   lines meet at one vanishing point and a side-wall line reaches the horizon
#   at another. The walls are square, so (v1 - p).(v2 - p) + f^2 = 0.
#
# Height: the floor and ceiling lines in one frame are parallel in 3D, so their
#   back-projected planes give the wall's direction. Both lines lie on the wall
#   at distance Z, which puts each at y = -Z (n.w)/(n.u). Working inside one
#   frame keeps the slow pose drift of a plain-wall reconstruction out of it.
#
# Corners and width: where the floor line stops inside the image there may be a
#   corner. Stops that triangulate to one 3D point across frames are accepted.
#   With both ends found the width is measured; otherwise the wall was not seen
#   to its end and only the stretch that was seen can be reported.

import cv2
import numpy as np

from Homework.hw5_motion_sfm import sfm_planar as sfm

MIN_SEGMENT_PX = 30
MIN_LINE_PX = 50
LINE_TOL_PX = 4.0
BORDER_PX = 12
FLOOR_REGION = 0.6       # floor line is in the lower 40% of the image
CEILING_REGION = 0.25    # ceiling line is in the upper 25%
MIN_FLOOR_SPAN = 0.2     # of the image width
MIN_CEILING_SPAN = 0.1
BASEBOARD_SEARCH_PX = 70
MIN_FLOOR_STEP = 12.0
MIN_SIDE_LINE_PX = 60
FOCAL_RANGE = (0.7, 1.6)
MIN_FOCAL_FRAMES = 2
MAX_VP_WIDTHS = 6        # a vanishing point further out than this is just noise
CEILING_CLUSTER = 0.93   # window heads and the like sit well below the ceiling
MIN_HEIGHT_FRAMES = 3
CORNER_TOL_PX = 10.0
MIN_CORNER_VIEWS = 3
MIN_CORNER_ANGLE_DEG = 3.0
WALL_DEPTH_TOL = 0.1


def _segments(gray):
    """Line segments as x1, y1, x2, y2 with x1 <= x2."""
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if found is None:
        return np.empty((0, 4))
    seg = found.reshape(-1, 4).astype(np.float64)
    swap = seg[:, 0] > seg[:, 2]
    seg[swap] = seg[swap][:, [2, 3, 0, 1]]
    return seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= MIN_SEGMENT_PX]


def _line_y(line, x):
    return line["a"] * x + line["b"]


def _mid_y(line):
    return _line_y(line, (line["x0"] + line["x1"]) / 2.0)


def _long_lines(seg):
    """Near-horizontal segments joined into lines y = ax + b, longest first."""
    dx = seg[:, 2] - seg[:, 0]
    slope = (seg[:, 3] - seg[:, 1]) / np.maximum(dx, 1e-9)
    keep = np.abs(slope) < np.tan(np.radians(20))
    seg, slope, dx = seg[keep], slope[keep], dx[keep]
    joined = []
    for i in np.argsort(-dx):
        if dx[i] < MIN_LINE_PX:
            break
        b = seg[i, 1] - slope[i] * seg[i, 0]
        off = np.maximum(np.abs(seg[:, 1] - slope[i] * seg[:, 0] - b), np.abs(seg[:, 3] - slope[i] * seg[:, 2] - b))
        on = off < LINE_TOL_PX
        x = np.r_[seg[on, 0], seg[on, 2]]
        a, b = np.polyfit(x, np.r_[seg[on, 1], seg[on, 3]], 1)
        joined.append({"a": float(a), "b": float(b), "x0": float(x.min()), "x1": float(x.max()),
                       "span": float(dx[on].sum())})
    lines = []
    for line in sorted(joined, key=lambda l: -l["span"]):
        mid = (line["x0"] + line["x1"]) / 2.0
        if all(abs(_line_y(line, mid) - _line_y(q, mid)) > 2 * LINE_TOL_PX for q in lines):
            lines.append(line)
    return lines


def _floor_edge(gray, line):
    """The top of a baseboard is often the stronger line, but the floor starts at
    its foot. Moves the line to the sharpest bright-to-dark step just below it."""
    h = gray.shape[0]
    xs = np.arange(int(line["x0"]), int(line["x1"]), 3)
    offsets = np.arange(-8, BASEBOARD_SEARCH_PX)
    profile = np.full(len(offsets), np.nan)
    for k, o in enumerate(offsets):
        ys = np.round(_line_y(line, xs) + o).astype(int)
        ok = (ys >= 0) & (ys < h)
        if ok.any():
            profile[k] = gray[ys[ok], xs[ok]].mean()
    step = profile[6:] - profile[:-6]
    if not np.isfinite(step).any():
        return line
    k = int(np.nanargmin(step))
    if step[k] > -MIN_FLOOR_STEP:
        return line
    return {**line, "b": line["b"] + float(offsets[k] + 3)}


def read_frame(frame):
    """The floor and ceiling lines of one frame, plus its segments."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    seg = _segments(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray))
    lines = _long_lines(seg)
    low = [l for l in lines if FLOOR_REGION * h < _mid_y(l) < h - BORDER_PX and l["span"] > MIN_FLOOR_SPAN * w]
    high = [l for l in lines if BORDER_PX < _mid_y(l) < CEILING_REGION * h and l["span"] > MIN_CEILING_SPAN * w]
    floor = max(low, key=_mid_y) if low else None
    return {"size": (w, h), "floor": _floor_edge(gray, floor) if floor else None,
            "ceiling": min(high, key=_mid_y) if high else None, "segments": seg}


def _focal_from_frame(view, K, up_camera):
    """f from one frame's two wall vanishing points, or None without a corner.
    The horizon comes from the reconstruction: near a corner the image's own
    verticals lean with perspective, which can't be told from roll."""
    ceiling, floor = view["ceiling"], view["floor"]
    if ceiling is None or floor is None or abs(ceiling["a"] - floor["a"]) < 0.05:
        return None
    w, h = view["size"]
    p = K[:2, 2]
    x1 = (floor["b"] - ceiling["b"]) / (ceiling["a"] - floor["a"])
    v1 = np.array([x1, _line_y(floor, x1)])
    if 0 <= x1 <= w or abs(x1 - p[0]) > MAX_VP_WIDTHS * w:
        return None
    seg = view["segments"]
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    slant = np.degrees(np.arctan2(np.abs(d[:, 1]), d[:, 0]))
    side = seg[(length >= MIN_SIDE_LINE_PX) & (slant > 20) & (slant < 60)]
    if not len(side):
        return None
    side = side[np.argmax(np.hypot(side[:, 2] - side[:, 0], side[:, 3] - side[:, 1]))]
    normal = np.linalg.inv(K).T @ up_camera
    horizon = np.array([normal[1], -normal[0]])
    along = side[2:] - side[:2]
    # solve side[:2] + s * along = v1 + r * horizon
    A = np.column_stack([along, -horizon])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s = np.linalg.solve(A, v1 - side[:2])[0]
    v2 = side[:2] + s * along
    f2 = -float(np.dot(v1 - p, v2 - p))
    return np.sqrt(f2) if f2 > 0 else None


def estimate_focal(views, Ps, K):
    """Median focal length in working pixels over the corner frames, with how
    many frames gave one. None when the video shows too few corners, or they
    disagree with the calibration beyond any plausible crop."""
    f0 = K[0, 0]
    Kinv = np.linalg.inv(K)
    Rs = [Kinv @ P[:3, :3] for P in Ps]
    up = up_direction(views, Rs, Kinv)
    found = [f for f in (_focal_from_frame(v, K, R @ up) for v, R in zip(views, Rs)) if f is not None]
    found = [f for f in found if FOCAL_RANGE[0] <= f / f0 <= FOCAL_RANGE[1]]
    if len(found) < MIN_FOCAL_FRAMES:
        return None, len(found)
    return float(np.median(found)), len(found)


def _plane(line, Kinv, w):
    n = np.cross(Kinv @ [0.0, _line_y(line, 0.0), 1.0], Kinv @ [float(w), _line_y(line, float(w)), 1.0])
    return n / np.linalg.norm(n)


def up_direction(views, Rs, Kinv):
    """World up: the direction lying in the back-projected plane of every
    near-vertical segment. Falls back on the camera's own up axis."""
    fallback = np.median([-R[1] for R in Rs], axis=0)
    fallback = fallback / np.linalg.norm(fallback)
    normals, weights = [], []
    for view, R in zip(views, Rs):
        seg = view["segments"]
        d = seg[:, 2:] - seg[:, :2]
        upright = np.abs(d[:, 1]) > 3 * np.abs(d[:, 0])
        for x1, y1, x2, y2 in seg[upright]:
            n = np.cross(Kinv @ [x1, y1, 1.0], Kinv @ [x2, y2, 1.0])
            normals.append(R.T @ (n / np.linalg.norm(n)))
            weights.append(np.hypot(x2 - x1, y2 - y1))
    if len(normals) < 10:
        return fallback
    N, w0 = np.array(normals), np.array(weights)
    w = w0.copy()
    up = fallback
    for limit in (8, 8, 4, 2.5, 2.5):
        up = np.linalg.svd(N * w[:, None], full_matrices=False)[2][-1]
        w = w0 * (np.abs(N @ up) < np.sin(np.radians(limit)))
        if not w.any():
            return fallback
    return up if up @ fallback > 0 else -up


def _height_ratio(view, R, up, Kinv):
    """Floor-to-ceiling height over the camera-to-wall distance, in one frame."""
    w = view["size"][0]
    n_top, n_bottom = _plane(view["ceiling"], Kinv, w), _plane(view["floor"], Kinv, w)
    along = np.cross(n_top, n_bottom)
    along = along / np.linalg.norm(along)
    u = R @ up
    u = u - along * (u @ along)
    u = u / np.linalg.norm(u)
    out = np.cross(along, u)
    out = out * np.sign(out[2])
    return abs((n_top @ out) / (n_top @ u) - (n_bottom @ out) / (n_bottom @ u))


def height_over_distance(views, Rs, up, Kinv):
    """Median ratio over the frames that see the ceiling. Keeps the top cluster
    because a window head can pass for the ceiling line in some frames."""
    ratios = np.array([_height_ratio(v, R, up, Kinv) for v, R in zip(views, Rs)
                       if v["floor"] is not None and v["ceiling"] is not None])
    if len(ratios) < MIN_HEIGHT_FRAMES:
        return None
    top = ratios[ratios >= CEILING_CLUSTER * np.percentile(ratios, 90)]
    if len(top) < MIN_HEIGHT_FRAMES:
        return None
    return {"ratio": float(np.median(top)), "frames": len(top), "spread": float(top.max() - top.min())}


def _reproject(P, X):
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2] if h[2] > 0 else np.array([np.inf, np.inf])


def _agree(X, seen, Ps):
    return [f for f, pt in seen.items() if np.linalg.norm(_reproject(Ps[f], X) - pt) < CORNER_TOL_PX]


def consensus_point(seen, Ps, centres):
    """The 3D point most of {frame: pixel} agree on. Pairs far apart along the
    path start it, the agreeing frames refine it. None without enough support."""
    frames = sorted(seen)
    best = []
    for i, a in enumerate(frames):
        for b in frames[i + 1:]:
            X = sfm.triangulate_point([Ps[a], Ps[b]], [seen[a], seen[b]])
            r1, r2 = X - centres[a], X - centres[b]
            cos = r1 @ r2 / (np.linalg.norm(r1) * np.linalg.norm(r2) + 1e-12)
            if np.degrees(np.arccos(np.clip(cos, -1, 1))) < MIN_CORNER_ANGLE_DEG:
                continue
            agree = _agree(X, seen, Ps)
            if len(agree) > len(best):
                best = agree
    if len(best) < MIN_CORNER_VIEWS:
        return None
    X = sfm.triangulate_point([Ps[f] for f in best], [seen[f] for f in best])
    agree = _agree(X, seen, Ps)
    return (X, agree) if len(agree) >= MIN_CORNER_VIEWS else None


def _line_ends(views, frames, side):
    """Where the floor line stops inside the image, per frame."""
    seen = {}
    for f, view in zip(frames, views):
        line = view["floor"]
        if line is None:
            continue
        w = view["size"][0]
        x = line["x0"] if side == "left" else line["x1"]
        if BORDER_PX < x < w - BORDER_PX:
            seen[f] = np.array([x, _line_y(line, x)])
    return seen


def visible_end(frames, Ps, Kinv, view_size, wall_point, normal, along):
    """How far along the wall the frames reach, right-hand image edge."""
    w, h = view_size
    reach = []
    for f in frames:
        M = Kinv @ Ps[f][:3, :3]
        ray = M.T @ (Kinv @ [float(w), h / 2.0, 1.0])
        centre = -np.linalg.solve(Ps[f][:3, :3], Ps[f][:3, 3])
        t = (wall_point - centre) @ normal / (ray @ normal)
        if t > 0:
            reach.append(float((centre + t * ray - wall_point) @ along))
    return max(reach) if reach else None


def analyse(views, frames, Ps, K, centres):
    """Everything readable from the lines, in reconstruction units. views and
    frames are the registered frames only; Ps and centres are keyed by frame."""
    Kinv = np.linalg.inv(K)
    Rs = [Kinv @ Ps[f][:3, :3] for f in frames]
    up = up_direction(views, Rs, Kinv)
    forward = np.median([R[2] for R in Rs], axis=0)
    forward = forward - up * (forward @ up)
    forward = forward / np.linalg.norm(forward)
    along = np.cross(up, forward)
    along = along * np.sign(along @ np.median([R[0] for R in Rs], axis=0))

    out = {"height": height_over_distance(views, Rs, up, Kinv), "corners": {}, "reach": None}
    left = consensus_point(_line_ends(views, frames, "left"), Ps, centres)
    mean_c = np.mean([centres[f] for f in frames], axis=0)
    if left is not None and (left[0] - mean_c) @ forward > 0:
        X, agree = left
        out["corners"]["A"] = {str(f): _line_ends(views, frames, "left")[f].tolist() for f in agree}
        depth = (X - mean_c) @ forward
        right = consensus_point(_line_ends(views, frames, "right"), Ps, centres)
        if right is not None and (right[0] - X) @ along > 0 and \
                abs((right[0] - mean_c) @ forward - depth) < WALL_DEPTH_TOL * depth:
            out["corners"]["B"] = {str(f): _line_ends(views, frames, "right")[f].tolist() for f in right[1]}
        else:
            out["reach"] = visible_end(frames, Ps, Kinv, views[0]["size"], X, forward, along)
    return out
