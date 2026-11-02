# HW7 part 3 - room dimensions from video.
#
# A video on its own only fixes a room up to scale, and only its visible
# parts. So the work splits into three steps:
#
# 1. Structure from motion on key frames of one video: SIFT matches between nearby frames,
#    an essential-matrix start on the best pair, then PnP registers each
#    further frame and new tracks are triangulated. Every frame then has a
#    pose [R|t] in one shared, unitless coordinate frame. Resection and
#    intersection are alternated at the end to pull the poses together.
#
# 2. Room corners rarely carry features (a plain wall meeting a plain wall),
#    so they are found from the room's straight edges (room_auto.py), and can
#    also be marked by hand on two or more key frames. Each corner is
#    triangulated through the recovered cameras (DLT, as in HW5).
#
# 3. A real measurement fixes the scale: how far the camera travelled, the
#    camera-to-wall distance, or a known length marked in the room. Under the
#    rectangular-room assumption, length, width and height are the means of
#    the parallel room edges that are known.
#
# The other known measurements, when given, are used as independent checks.

import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from Homework.hw5_motion_sfm import sfm_planar as sfm
from Homework.hw5_motion_sfm.optical_flow import _write_status
from . import room_auto as auto

WORK_MAX_SIDE = 1280
MATCH_WINDOW = 3     # each key frame is matched against the next 3
MIN_PAIR_MATCHES = 40
ROW_SLOPE = 0.15
MIN_PNP_INLIERS = 15
MIN_POINTS = 30
MIN_PARALLAX_DEG = 1.0
MAX_REPROJ_PX = 2.5
BAD_CORNER_PX = 12.0
FOCAL_SCAN = (1.0, 1.1, 1.2)   # of the starting focal length: video crops, so it only ever runs short
MIN_FOCAL_SAMPLES = 3
MAX_REFINED_RMS = 1.0

CORNERS = {"A": "floor, front-left", "B": "floor, front-right", "C": "floor, back-right", "D": "floor, back-left",
           "E": "ceiling, front-left", "F": "ceiling, front-right", "G": "ceiling, back-right",
           "H": "ceiling, back-left"}
EDGES = {"width": ["AB", "DC", "EF", "HG"], "length": ["AD", "BC", "EH", "FG"], "height": ["AE", "BF", "CG", "DH"]}
FRONT_WALL = "ABEF"


class RoomError(ValueError):
    """A problem with the inputs the user can act on; shown as-is in the UI."""


def probe_video(path):
    """Size and length of a video, reading a frame to prove it decodes."""
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if not ok:
        raise RoomError("That video could not be decoded. Try an MP4 (H.264) file.")
    return {"width": frame.shape[1], "height": frame.shape[0], "frames": n, "fps": fps,
            "seconds": n / fps if fps else 0.0}


def extract_keyframes(path, n_keyframes, K, dist):
    """The very first and last frames, so the key frames span the whole camera
    path (the travel scale depends on it), and between them the sharpest frame
    of each equal stretch. Frames are shrunk to the working size and undistorted.
    Returns frames, their K, source indices."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_keyframes:
        total = 0
        while cap.grab():
            total += 1
        cap.release()
        cap = cv2.VideoCapture(path)
    if total < 2:
        cap.release()
        raise RoomError("The video has fewer than 2 frames.")
    n_keyframes = min(n_keyframes, total)
    bounds = np.linspace(0, total, n_keyframes + 1).astype(int)
    probes = {0, total - 3, total - 2, total - 1}   # the frame count in a container can run slightly high
    for a, b in zip(bounds[:-1], bounds[1:]):
        probes.update(np.linspace(a, b - 1, min(5, b - a)).astype(int).tolist())

    best = {}
    window = 0
    for i in range(total):
        if not cap.grab():
            break
        while window + 1 < n_keyframes and i >= bounds[window + 1]:
            window += 1
        if i not in probes:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if window == 0 and i > 0 and 0 in best:
            continue
        if window == n_keyframes - 1 or window not in best or sharp > best[window][0]:
            best[window] = (sharp, i, frame)
    cap.release()
    if len(best) < 2:
        raise RoomError("Could not read enough frames from the video.")

    frames, indices = [], []
    K_work = None
    for window in sorted(best):
        _, i, frame = best[window]
        if dist is not None:
            frame = cv2.undistort(frame, K, dist)
        h, w = frame.shape[:2]
        s = min(1.0, WORK_MAX_SIDE / max(h, w))
        if s < 1.0:
            size = (round(w * s), round(h * s))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if K_work is None:
            K_work = K.copy()
            K_work[0] *= frame.shape[1] / w
            K_work[1] *= frame.shape[0] / h
        frames.append(frame)
        indices.append(i)
    return frames, K_work, indices


def _detect(frames, on_progress):
    # painted walls have almost no contrast, so default SIFT finds next to
    # nothing on them; local contrast equalisation brings out what is there
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    features = []
    for i, f in enumerate(frames):
        features.append(sift.detectAndCompute(clahe.apply(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)), None))
        on_progress(5 + 25 * (i + 1) / len(frames), "Detecting SIFT features")
    return features


def _match_pairs(features, on_progress):
    """{(i, j): (kp indices in i, kp indices in j)} for nearby frames, kept
    only if they fit one fundamental matrix."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    n = len(features)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, min(i + 1 + MATCH_WINDOW, n))]
    out = {}
    for c, (i, j) in enumerate(pairs):
        on_progress(30 + 30 * (c + 1) / len(pairs), "Matching frames")
        (ki, di), (kj, dj) = features[i], features[j]
        if di is None or dj is None or len(ki) < 8 or len(kj) < 8:
            continue
        good = [p[0] for p in matcher.knnMatch(di, dj, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < MIN_PAIR_MATCHES:
            continue
        a = np.array([m.queryIdx for m in good])
        b = np.array([m.trainIdx for m in good])
        pa = np.float64([ki[m.queryIdx].pt for m in good])
        pb = np.float64([kj[m.trainIdx].pt for m in good])
        # repeating patterns (blinds, tiles) give false matches one period away
        # that fit a wrong motion; on a sideways sweep the true ones run along
        # the row, so use those whenever there are enough of them
        flow = pb - pa
        along_row = np.abs(flow[:, 1]) < ROW_SLOPE * np.abs(flow[:, 0])
        if along_row.sum() >= MIN_PAIR_MATCHES:
            a, b, pa, pb = a[along_row], b[along_row], pa[along_row], pb[along_row]
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.999)
        if mask is None:
            continue
        keep = mask.ravel() > 0
        if keep.sum() >= MIN_PAIR_MATCHES:
            out[(i, j)] = (a[keep], b[keep])
    return out


def _build_tracks(features, matches):
    """Union-find over (frame, keypoint) nodes. Each track is {frame: kp};
    tracks seen twice in one frame are ambiguous and dropped."""
    offset = np.cumsum([0] + [len(f[0]) for f in features])
    parent = list(range(offset[-1]))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), (a, b) in matches.items():
        for u, v in zip(a, b):
            parent[find(int(offset[i] + u))] = find(int(offset[j] + v))

    groups = {}
    nodes = {}
    for (i, j), (a, b) in matches.items():
        nodes.update({int(offset[i] + u): (i, int(u)) for u in a})
        nodes.update({int(offset[j] + v): (j, int(v)) for v in b})
    for node, (frame, kp) in nodes.items():
        groups.setdefault(find(node), []).append((frame, kp))
    tracks = []
    for g in groups.values():
        frames = [f for f, _ in g]
        if len(set(frames)) == len(frames):
            tracks.append(dict(g))
    return tracks


def _reproj_error(P, X, pt):
    h = P @ np.append(X, 1.0)
    if h[2] <= 0:
        return np.inf
    return float(np.hypot(h[0] / h[2] - pt[0], h[1] / h[2] - pt[1]))


def _ray_angle_deg(c1, c2, X):
    a, b = c1 - X, c2 - X
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _initial_pair(features, matches, K):
    """The pair with many inliers and real parallax. Rotation alone gives
    plenty of matches but no depth, so parallax has to be checked."""
    best, best_score = None, 0.0
    max_parallax = 0.0
    for (i, j), (a, b) in matches.items():
        pa = np.float64([features[i][0][int(u)].pt for u in a])
        pb = np.float64([features[j][0][int(v)].pt for v in b])
        E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            continue
        n, R, t, mask = cv2.recoverPose(E, pa, pb, K, mask=mask)
        keep = mask.ravel() > 0
        if keep.sum() < MIN_PAIR_MATCHES:
            continue
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        Xh = cv2.triangulatePoints(P1, P2, pa[keep].T, pb[keep].T)
        X = (Xh[:3] / Xh[3]).T
        c2 = -R.T @ t.ravel()
        ang = np.degrees(np.arccos(np.clip(np.sum(X * (X - c2), axis=1) /
                                           (np.linalg.norm(X, axis=1) * np.linalg.norm(X - c2, axis=1) + 1e-12),
                                           -1.0, 1.0)))
        parallax = float(np.median(ang))
        max_parallax = max(max_parallax, parallax)
        if parallax < MIN_PARALLAX_DEG:
            continue
        score = keep.sum() * min(parallax, 10.0)
        if score > best_score:
            best, best_score = (i, j, R, t.ravel()), score
    if best is None:
        raise RoomError(f"No pair of frames has enough parallax (best median {max_parallax:.2f} deg, need "
                        f"{MIN_PARALLAX_DEG} deg). The camera has to move sideways through the room, not just rotate "
                        f"on the spot, and the room needs visible texture.")
    return best


def reconstruct(frames, K, on_progress=lambda pct, label: None, cache=None):
    """Incremental SfM. Returns poses {frame: (R, t)}, points, colours and stats.
    Features and matches don't depend on K, so a cache dict keeps them for a
    second pass with a corrected focal length."""
    cv2.setRNGSeed(0)   # RANSAC here should give one answer per video
    cache = {} if cache is None else cache
    if "matches" not in cache:
        cache["features"] = _detect(frames, on_progress)
        cache["matches"] = _match_pairs(cache["features"], on_progress)
    features, matches = cache["features"], cache["matches"]
    if not matches:
        raise RoomError(f"No two nearby key frames share {MIN_PAIR_MATCHES} features. The video needs a slower "
                        f"sweep or more visible texture (posters, furniture, door frames).")
    tracks = _build_tracks(features, matches)
    on_progress(62, "Choosing the starting pair")
    a, b, R, t = _initial_pair(features, matches, K)

    def kp(frame, idx):
        return features[frame][0][idx].pt

    poses = {a: (np.eye(3), np.zeros(3)), b: (R, t)}
    Ps = {f: sfm.projection_matrix(K, *pose) for f, pose in poses.items()}
    points = {}

    def triangulate_new():
        centres = {f: sfm.camera_center(*poses[f]) for f in poses}
        for ti, track in enumerate(tracks):
            if ti in points:
                continue
            seen = [f for f in track if f in poses]
            if len(seen) < 2:
                continue
            X = sfm.triangulate_point([Ps[f] for f in seen], [kp(f, track[f]) for f in seen])
            if max(_reproj_error(Ps[f], X, kp(f, track[f])) for f in seen) > MAX_REPROJ_PX:
                continue
            if max(_ray_angle_deg(centres[seen[0]], centres[f], X) for f in seen[1:]) < MIN_PARALLAX_DEG / 2:
                continue
            points[ti] = X

    triangulate_new()
    pending = set(range(len(frames))) - set(poses)
    while pending:
        on_progress(65 + 25 * (1 - len(pending) / len(frames)), "Registering frames")
        counts = {f: sum(1 for ti in points if f in tracks[ti]) for f in pending}
        f = max(counts, key=counts.get)
        pending.discard(f)
        if counts[f] < MIN_PNP_INLIERS:
            continue
        ids = [ti for ti in points if f in tracks[ti]]
        obj = np.float64([points[ti] for ti in ids])
        img = np.float64([kp(f, tracks[ti][f]) for ti in ids])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=3.0, iterationsCount=300,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < MIN_PNP_INLIERS:
            continue
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
        poses[f] = (cv2.Rodrigues(rvec)[0], tvec.ravel())
        Ps[f] = sfm.projection_matrix(K, *poses[f])
        triangulate_new()

    # resection-intersection: the first camera stays fixed as the gauge
    on_progress(92, "Refining poses and points")
    for _ in range(3):
        for ti in list(points):
            seen = [f for f in tracks[ti] if f in poses]
            X = sfm.triangulate_point([Ps[f] for f in seen], [kp(f, tracks[ti][f]) for f in seen])
            good = [f for f in seen if _reproj_error(Ps[f], X, kp(f, tracks[ti][f])) < MAX_REPROJ_PX]
            if len(good) < 2:
                del points[ti]
                continue
            points[ti] = sfm.triangulate_point([Ps[f] for f in good], [kp(f, tracks[ti][f]) for f in good])
        for f in poses:
            if f == a:
                continue
            ids = [ti for ti in points if f in tracks[ti]]
            if len(ids) < MIN_PNP_INLIERS:
                continue
            obj = np.float64([points[ti] for ti in ids])
            img = np.float64([kp(f, tracks[ti][f]) for ti in ids])
            rvec = cv2.Rodrigues(poses[f][0])[0]
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, poses[f][1].copy().reshape(3, 1))
            poses[f] = (cv2.Rodrigues(rvec)[0], tvec.ravel())
            Ps[f] = sfm.projection_matrix(K, *poses[f])

    if len(poses) < 2 or len(points) < MIN_POINTS:
        raise RoomError(f"The reconstruction has only {len(points)} 3D points on {len(poses)} frames (need "
                        f"{MIN_POINTS} points). The room needs more texture or a slower sweep.")

    errs, ids, colours = [], [], []
    for ti, X in points.items():
        errs.extend(_reproj_error(Ps[f], X, kp(f, tracks[ti][f])) for f in tracks[ti] if f in poses)
        ids.append(ti)
        f = next(f for f in tracks[ti] if f in poses)
        x, y = kp(f, tracks[ti][f])
        colours.append(frames[f][int(y), int(x)][::-1] / 255.0)
    return {"poses": poses, "points": np.float64([points[ti] for ti in ids]), "colours": np.float64(colours),
            "reproj_rms": float(np.sqrt(np.mean(np.square(np.clip(errs, 0, 1e3))))),
            "n_frames": len(frames), "ref_frame": a, "pair": (a, b)}


def path_complete(registered, n_extracted):
    """True when the first and last video frames are both in the reconstruction,
    so the reconstructed path can stand for the camera's real travel."""
    return {0, n_extracted - 1} <= set(registered)


def camera_travel(centres):
    """Largest distance between any two camera positions on the path."""
    c = np.asarray(centres, dtype=np.float64)
    return float(np.max(np.linalg.norm(c[:, None] - c[None, :], axis=2)))


def triangulate_marks(marks, Ps, centres):
    """{label: {frame: [x, y]}} -> {label: {X, views, reproj, ray_angle}}."""
    out = {}
    for label, views in marks.items():
        frames = [int(f) for f in views if int(f) in Ps]
        if not views:
            continue
        if len(frames) < 2:
            raise RoomError(f"Point {label} is marked on {len(frames)} reconstructed frame(s). Every marked point "
                            f"needs at least 2 frames, ideally far apart along the camera path.")
        obs = [views[str(f)] for f in frames]
        X = sfm.triangulate_point([Ps[f] for f in frames], obs)
        err = max(_reproj_error(Ps[f], X, o) for f, o in zip(frames, obs))
        angle = max(_ray_angle_deg(centres[f1], centres[f2], X) for f1 in frames for f2 in frames)
        out[label] = {"X": X, "views": len(frames), "reproj": err, "ray_angle": angle}
    return out


def viewing_direction(Ps, K):
    """Median camera forward axis in reconstruction coordinates."""
    Kinv = np.linalg.inv(K)
    forward = np.median([(Kinv @ P[:3, :3])[2] for P in Ps], axis=0)
    return forward / np.linalg.norm(forward)


def front_wall_distance(points, centres, forward):
    """Distance from the mean camera position to the front wall. Three corners
    give the wall plane itself. A video may show only one end of the wall, so
    with fewer the wall is taken as square to the camera's viewing direction,
    which a sideways sweep facing the wall gives. None if no corner is marked."""
    wall = [points[c]["X"] for c in FRONT_WALL if c in points]
    if not wall:
        return None
    mean_camera = np.mean(centres, axis=0)
    if len(wall) < 3:
        return float(abs(np.dot(np.mean(wall, axis=0) - mean_camera, forward)))
    centroid, normal, _ = sfm.fit_plane(wall)
    if len(wall) == 3:
        normal = np.cross(wall[1] - wall[0], wall[2] - wall[0])
        if np.linalg.norm(normal) < 1e-9:
            return None
        normal = normal / np.linalg.norm(normal)
    return float(abs(np.dot(mean_camera - centroid, normal)))


def measure_room(points, centres, forward, scale_mode, travel_mm, wall_mm, ref_mm, back_to_wall=False):
    """Fix the scale from one known measurement, then read off the room."""
    travel = camera_travel(centres)
    wall = front_wall_distance(points, centres, forward)
    ref = None
    if "R1" in points and "R2" in points:
        ref = float(np.linalg.norm(points["R1"]["X"] - points["R2"]["X"]))

    if scale_mode == "travel":
        if not travel_mm or travel_mm <= 0:
            raise RoomError("Enter the camera's left-right travel in mm to use it as the scale.")
        if travel < 1e-9:
            raise RoomError("The camera path has no length, so it cannot set the scale.")
        scale, source = travel_mm / travel, f"camera travel = {travel_mm:g} mm"
    elif scale_mode == "wall":
        if not wall_mm or wall_mm <= 0:
            raise RoomError("Enter the camera-to-wall distance in mm to use it as the scale.")
        if wall is None or wall < 1e-9:
            raise RoomError("The camera-to-wall scale needs at least 1 front-wall corner marked "
                            "(A, B, E or F), on 2 or more frames.")
        scale, source = wall_mm / wall, f"camera to front wall = {wall_mm:g} mm"
    elif scale_mode == "reference":
        if not ref_mm or ref_mm <= 0:
            raise RoomError("Enter the real length between R1 and R2 in mm.")
        if ref is None or ref < 1e-9:
            raise RoomError("The known-length scale needs both R1 and R2 marked, each on at least 2 frames.")
        scale, source = ref_mm / ref, f"R1-R2 = {ref_mm:g} mm"
    else:
        raise RoomError("Choose a scale reference.")

    dims = {}
    for name, edges in EDGES.items():
        seen = [{"edge": f"{e[0]}-{e[1]}",
                 "mm": float(np.linalg.norm(points[e[0]]["X"] - points[e[1]]["X"]) * scale)}
                for e in edges if e[0] in points and e[1] in points]
        if seen:
            vals = [s["mm"] for s in seen]
            dims[name] = {"mm": float(np.mean(vals)), "edges": seen, "spread": float(max(vals) - min(vals)),
                          "method": f"mean of {len(vals)} edge(s)" if len(vals) > 1 else "1 edge"}
    if "length" not in dims and back_to_wall and wall is not None:
        dims["length"] = {"mm": wall * scale, "edges": [], "spread": 0.0,
                          "method": "camera to front wall (you stated the camera was at the opposite wall)"}

    checks = []
    if scale_mode != "travel" and travel_mm:
        checks.append({"name": "Camera left-right travel", "known": travel_mm, "measured": travel * scale})
    if wall is not None and scale_mode != "wall" and wall_mm:
        checks.append({"name": "Camera to front wall", "known": wall_mm, "measured": wall * scale})
    if ref is not None and scale_mode != "reference" and ref_mm:
        checks.append({"name": "R1-R2 known length", "known": ref_mm, "measured": ref * scale})
    for c in checks:
        c["error_pct"] = 100.0 * (c["measured"] - c["known"]) / c["known"]
    return {"scale": scale, "source": source, "dims": dims, "checks": checks,
            "wall_mm": None if wall is None else wall * scale, "travel_mm": travel * scale}


def add_auto_dimensions(res, found, points):
    """Fills in what the room's lines give, for dimensions the corners don't.
    Width is only a lower limit when the wall's far end was not seen."""
    wall_mm = res["wall_mm"]
    height = found.get("height")
    if "height" not in res["dims"] and height and wall_mm:
        res["dims"]["height"] = {
            "mm": height["ratio"] * wall_mm, "edges": [], "spread": height["spread"] * wall_mm,
            "method": f"floor and ceiling lines on the front wall, median of {height['frames']} frames"}
    if "width" not in res["dims"] and found.get("reach") and "A" in points:
        res["dims"]["width"] = {"mm": found["reach"] * res["scale"], "edges": [], "spread": 0.0, "lower_limit": True,
                                "method": "front wall from its left corner to the last point the video shows"}
    return res


def plot_scene(centres, cloud, colours, corners, unit, path):
    """Top-down view and 3D view, in the first camera's axes (X right, Z ahead)."""
    centres = np.asarray(centres, dtype=np.float64)
    keep = np.ones(len(cloud), bool)
    if len(cloud):
        lo, hi = np.percentile(cloud, [2, 98], axis=0)
        keep = np.all((cloud >= lo) & (cloud <= hi), axis=1)
    pts = cloud[keep]
    cols = colours[keep]
    fig = plt.figure(figsize=(11, 4.8), dpi=110)
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(pts[:, 0], pts[:, 2], c=cols, s=3)
    ax.plot(centres[:, 0], centres[:, 2], "-o", color="#f97316", ms=3, label="camera path")
    for label, X in corners.items():
        ax.plot(X[0], X[2], "s", color="#4f46e5", ms=5)
        ax.annotate(label, (X[0], X[2]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Z ahead ({unit})")
    ax.set_title("Top-down")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="best", fontsize=8)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=2)
    ax.plot(centres[:, 0], centres[:, 2], -centres[:, 1], "-o", color="#f97316", ms=3)
    for label, X in corners.items():
        ax.scatter([X[0]], [X[2]], [-X[1]], color="#4f46e5", marker="s", s=24)
        ax.text(X[0], X[2], -X[1], label, fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")
    ax.set_title("3D reconstruction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _refine_focal(frames, views, K, on_progress, cache):
    """The corner estimate depends a little on the poses it is read through, and
    the true focal length is the one that reproduces itself. So reconstruct at a
    few values and solve estimate(f) = f on the line through them. Returns
    result, K, frames used, or None if the video can't settle it."""
    tried = []
    for s in FOCAL_SCAN:
        K_s = K.copy()
        K_s[:2, :2] *= s
        try:
            res = reconstruct(frames, K_s, on_progress, cache)
        except RoomError:
            continue
        order = sorted(res["poses"])
        found, n = auto.estimate_focal([views[f] for f in order],
                                       [sfm.projection_matrix(K_s, *res["poses"][f]) for f in order], K_s)
        if found:
            tried.append((K_s[0, 0], found, n))
    if len(tried) < MIN_FOCAL_SAMPLES:
        return None
    start, found, used = np.array(tried).T
    slope, offset = np.polyfit(start, found, 1)
    if slope > 0.9:
        return None
    f = offset / (1 - slope)
    if not 0.9 * start.min() <= f <= 1.1 * start.max():
        return None
    K_new = K.copy()
    K_new[:2, :2] *= f / K[0, 0]
    res = reconstruct(frames, K_new, on_progress, cache)
    if res["reproj_rms"] > MAX_REFINED_RMS:
        return None
    return res, K_new, int(np.median(used))


def process_video(video_path, run_dir, K, dist, n_keyframes, meta, refine_focal=True):
    """Background job: key frames + SfM + reading the room's lines. Writes
    frames, recon.npz, meta.json."""
    def progress(pct, label):
        _write_status(run_dir, state="running", progress=round(pct), label=label)

    try:
        progress(1, "Extracting frames")
        frames, K_work, indices = extract_keyframes(video_path, n_keyframes, K, dist)
        names = []
        for i, f in enumerate(frames):
            name = f"frame_{i:02d}.jpg"
            cv2.imwrite(os.path.join(run_dir, name), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            names.append(name)
        cache = {}
        res = reconstruct(frames, K_work, progress, cache)

        views = [auto.read_frame(f) for f in frames]
        if refine_focal:
            refined = _refine_focal(frames, views, K_work, progress, cache)
            if refined:
                res, K_work, n = refined
                full = K_work[0, 0] * meta["video"]["width"] / frames[0].shape[1]
                meta["k_source"] += (f" Focal length corrected to {full:.0f} px from the room's right angles "
                                     f"on {n} frames.")

        order = sorted(res["poses"])
        Ps = np.stack([sfm.projection_matrix(K_work, *res["poses"][f]) for f in order])
        centres = np.stack([sfm.camera_center(*res["poses"][f]) for f in order])
        np.savez(os.path.join(run_dir, "recon.npz"), registered=np.array(order), Ps=Ps, centres=centres,
                 cloud=res["points"], colours=res["colours"], K=K_work)
        plot_scene(centres, res["points"], res["colours"], {}, "reconstruction units",
                   os.path.join(run_dir, "reconstruction.png"))
        meta["auto"] = auto.analyse([views[f] for f in order], order, dict(zip(order, Ps)), K_work,
                                    dict(zip(order, centres)))
        meta.update(names=names, source_frames=indices, registered=order, n_points=len(res["points"]),
                    reproj_rms=round(res["reproj_rms"], 3), frame_size=[frames[0].shape[1], frames[0].shape[0]],
                    fx_work=float(K_work[0, 0]), work_scale=frames[0].shape[1] / meta["video"]["width"],
                    pair=list(res["pair"]), n_extracted=len(frames), dropped=len(frames) - len(order))
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        _write_status(run_dir, state="done", progress=100, label="Done")
    except RoomError as e:
        _write_status(run_dir, state="error", message=str(e))
    except Exception as e:  # a background thread has nowhere else to report to
        _write_status(run_dir, state="error", message=f"Reconstruction failed: {e}")
