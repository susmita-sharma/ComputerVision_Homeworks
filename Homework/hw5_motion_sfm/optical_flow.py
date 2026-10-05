# HW5 part 1 - dense optical flow + live object/feature tracking over a
# real video clip.
#
# Per consecutive frame pair (t, t+1):
#   1. Farneback dense flow: one (u, v) vector per pixel.
#   2. KLT feature tracks (Shi-Tomasi corners + pyramidal LK, forward-
#      backward checked) - sparse points followed across many frames.
#   3. Camera ego-motion: a similarity transform (tx, ty, rotation, scale)
#      fit with RANSAC to the KLT matches. Most tracked corners sit on the
#      static background, so this is "how the whole image moved because
#      the camera moved".
#   4. Camera-compensated (residual) flow = dense flow - flow predicted by
#      the camera transform. What is left over is motion of things moving
#      *relative to the scene* - those connected regions are the "moving
#      objects", boxed and given persistent IDs by a centroid tracker.
#
# Every frame is rendered as a 2x2 panel (tracking overlay | HSV flow |
# flow vectors | residual-motion heatmap) into a browser-playable WebM at
# the clip's own (sub-sampled) frame rate, so it plays back in real time.
# Per-frame statistics are logged so the "what can be inferred" write-up
# is backed by numbers (stats.csv, stats.png, objects.csv, summary.md).
#
# Runs as a background job (process_video is called from a thread) and
# reports progress through files in its run directory - status.json and
# live.jpg - so any web worker can serve the live preview.

import csv
import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRIMARY = "#4f46e5"
ACCENT = "#f97316"
OBJ_COLORS = [(0, 200, 255), (255, 120, 0), (80, 220, 80), (255, 0, 200), (0, 120, 255), (200, 200, 0),
              (180, 80, 255), (60, 255, 200)]  # BGR

TARGET_FPS = 30.0    # 60fps phone clips are sub-sampled to ~30fps - still real-time playback
ANALYSIS_MAX_DIM = 400


# ------------------------------------------------------------ flow math --

def dense_flow(prev_gray, next_gray):
    """Farneback dense optical flow: one (dx, dy) vector per pixel."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def flow_to_color(flow, max_mag=None):
    """Classic HSV flow visualization: angle -> hue, magnitude -> value.

    Color tells you *which way* things moved, brightness *how fast*.
    Brightness is normalized against max(3px, 99th percentile) rather than
    the frame max, so a nearly static frame stays dark instead of
    amplifying sub-pixel noise into bright color.
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    if max_mag is None:
        max_mag = max(3.0, float(np.percentile(mag, 99)))
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_direction_wheel(size=90):
    """Legend: color <-> direction (image coords, y down), so the flow
    colors are decodable."""
    ys, xs = np.mgrid[0:size, 0:size]
    dx = xs - size / 2
    dy = ys - size / 2
    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (size / 2) * 255, 0, 255).astype(np.uint8)
    wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    wheel[mag > size / 2] = 0
    return wheel


def flow_stats(flow):
    """Numeric summary of one flow field."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    moving_mask = mag > 1.0
    dominant_dir = float(_circular_mean_deg(ang[moving_mask], mag[moving_mask])) if moving_mask.any() else 0.0
    # divergence du/dx + dv/dy: > 0 = expanding field (approaching / camera moving forward)
    div = np.gradient(flow[..., 0], axis=1) + np.gradient(flow[..., 1], axis=0)
    return {
        "mean_mag": float(mag.mean()),
        "max_mag": float(mag.max()),
        "moving_frac": float(moving_mask.mean()),
        "dominant_dir_deg": dominant_dir,
        "divergence": float(div.mean()),
    }


def _circular_mean_deg(ang_deg, weights=None):
    a = np.radians(np.asarray(ang_deg, dtype=np.float64))
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=np.float64)
    return (np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) + 360) % 360


def direction_name(deg):
    """Image-coordinate angle (0 = +x right, 90 = +y down) -> words."""
    names = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
    return names[int(((deg % 360) + 22.5) // 45) % 8]


# ------------------------------------------------------- camera motion --

def estimate_camera_motion(prev_pts, next_pts):
    """Similarity transform (tx, ty, rotation, scale) of the dominant image
    motion, fit with RANSAC to KLT matches. Returns the 2x3 matrix (or
    identity if there aren't enough matches) and its decomposition."""
    M = None
    if len(prev_pts) >= 8:
        M, inliers = cv2.estimateAffinePartial2D(prev_pts, next_pts, method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.0)
    if M is None:
        M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return M, {"tx": float(M[0, 2]), "ty": float(M[1, 2]), "rot_deg": rot, "scale": scale}


def camera_flow_field(M, h, w):
    """Flow each pixel would have if ONLY the camera moved (per M)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs
    v = M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys
    return np.dstack([u, v]).astype(np.float32)


def detect_moving_objects(residual_mag, min_area_frac=0.003, min_thr=1.0, max_objects=6):
    """Connected regions of camera-compensated motion = moving objects.

    Threshold adapts to the frame's noise floor (median + 4 robust sigma),
    never below `min_thr` px/frame."""
    smooth = cv2.GaussianBlur(residual_mag, (0, 0), 2)
    med = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - med)))
    thr = max(min_thr, med + 4 * 1.4826 * mad)

    mask = (smooth > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    h, w = residual_mag.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    objs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_frac * h * w:
            continue
        x, y, bw, bh = stats[i, :4]
        objs.append({"box": (int(x), int(y), int(bw), int(bh)), "area": int(area),
                     "centroid": (float(cents[i, 0]), float(cents[i, 1])), "label": i})
    objs.sort(key=lambda o: -o["area"])
    objs = objs[:max_objects]
    keep = np.zeros_like(mask)
    for o in objs:
        keep[labels == o["label"]] = 1
    return objs, keep, thr


class CentroidTracker:
    """Gives detected object regions persistent IDs across frames by
    greedy nearest-centroid matching."""

    def __init__(self, max_dist, max_missed=6):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self.next_id = 1
        self.active = {}   # id -> {"c", "missed", "trail"}
        self.history = {}  # id -> per-object log for objects.csv

    def update(self, detections, t_sec):
        unmatched = list(range(len(detections)))
        pairs = []
        for oid, o in self.active.items():
            for j in unmatched:
                d = np.hypot(o["c"][0] - detections[j]["centroid"][0], o["c"][1] - detections[j]["centroid"][1])
                if d < self.max_dist:
                    pairs.append((d, oid, j))
        pairs.sort()
        used_ids, used_dets = set(), set()
        for d, oid, j in pairs:
            if oid in used_ids or j in used_dets:
                continue
            used_ids.add(oid)
            used_dets.add(j)
            self._assign(oid, detections[j], t_sec)
        for j in range(len(detections)):
            if j not in used_dets:
                oid = self.next_id
                self.next_id += 1
                self.active[oid] = {"c": detections[j]["centroid"], "missed": 0, "trail": []}
                self.history[oid] = {"first_t": t_sec, "last_t": t_sec, "n_frames": 0, "path_px": 0.0,
                                     "speed_sum": 0.0, "dir_sin": 0.0, "dir_cos": 0.0, "max_area": 0}
                self._assign(oid, detections[j], t_sec)
        for oid in list(self.active):
            if oid not in used_ids and not any(d.get("id") == oid for d in detections):
                self.active[oid]["missed"] += 1
                if self.active[oid]["missed"] > self.max_missed:
                    del self.active[oid]

    def _assign(self, oid, det, t_sec):
        o = self.active[oid]
        prev_c = o["c"]
        o["c"] = det["centroid"]
        o["missed"] = 0
        o["trail"] = (o["trail"] + [det["centroid"]])[-30:]
        det["id"] = oid
        hst = self.history[oid]
        if hst["n_frames"] > 0:
            hst["path_px"] += float(np.hypot(det["centroid"][0] - prev_c[0], det["centroid"][1] - prev_c[1]))
        hst["n_frames"] += 1
        hst["last_t"] = t_sec
        hst["speed_sum"] += det.get("speed", 0.0)
        a = np.radians(det.get("dir_deg", 0.0))
        hst["dir_sin"] += np.sin(a) * det.get("speed", 0.0)
        hst["dir_cos"] += np.cos(a) * det.get("speed", 0.0)
        hst["max_area"] = max(hst["max_area"], det["area"])


# ------------------------------------------------------------ drawing --

def _label(img, text, org, scale=0.4, color=(255, 255, 255), bg=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _panel_title(img, text):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 18), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, text, (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_vectors(gray, flow, step=12, gain=3.0):
    vis = cv2.cvtColor((gray * 0.5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h, w = gray.shape
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = flow[y, x]
            m = np.hypot(u, v)
            if m < 0.3:
                continue
            hue = int(((np.degrees(np.arctan2(v, u)) + 360) % 360) / 2)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.arrowedLine(vis, (x, y), (int(x + u * gain), int(y + v * gain)), color, 1, cv2.LINE_AA,
                            tipLength=0.35)
    return vis


def render_panels(frame, gray, flow, residual_mag, obj_mask, objects, tracks, moving_flags, cam, thr,
                  t_sec, wheel, tracker):
    h, w = frame.shape[:2]

    # 1) tracking overlay
    p1 = frame.copy()
    for tr, moving in zip(tracks, moving_flags):
        pts = np.array(tr[-12:], dtype=np.int32)
        color = (0, 140, 255) if moving else (255, 200, 0)
        if len(pts) > 1:
            cv2.polylines(p1, [pts], False, color, 1, cv2.LINE_AA)
        cv2.circle(p1, tuple(pts[-1]), 2, color, -1)
    for o in objects:
        c = OBJ_COLORS[(o["id"] - 1) % len(OBJ_COLORS)]
        x, y, bw, bh = o["box"]
        cv2.rectangle(p1, (x, y), (x + bw, y + bh), c, 2)
        cx, cy = o["centroid"]
        a = np.radians(o["dir_deg"])
        cv2.arrowedLine(p1, (int(cx), int(cy)),
                        (int(cx + np.cos(a) * min(40, 6 * o["speed"] + 8)), int(cy + np.sin(a) * min(40, 6 * o["speed"] + 8))),
                        c, 2, cv2.LINE_AA, tipLength=0.3)
        trail = tracker.active.get(o["id"], {}).get("trail", [])
        if len(trail) > 1:
            cv2.polylines(p1, [np.array(trail, dtype=np.int32)], False, c, 1, cv2.LINE_AA)
        _label(p1, f"obj {o['id']}  {o['speed']:.1f}px/f", (x, max(12, y - 3)), 0.38, (0, 0, 0), c)
    _label(p1, f"t={t_sec:5.2f}s  objects:{len(objects)}  tracks:{len(tracks)}", (4, 13), 0.38)
    _label(p1, f"camera shift ({cam['tx']:+.1f},{cam['ty']:+.1f})px/f", (4, 28), 0.38)
    _panel_title(p1, "1 Objects + KLT feature tracks")

    # 2) HSV flow with direction legend
    p2 = flow_to_color(flow)
    wh, ww = wheel.shape[:2]
    p2[2:2 + wh, w - ww - 2:w - 2] = np.maximum(p2[2:2 + wh, w - ww - 2:w - 2], wheel)
    for o in objects:
        x, y, bw, bh = o["box"]
        cv2.rectangle(p2, (x, y), (x + bw, y + bh), (255, 255, 255), 1)
    _panel_title(p2, "2 Dense flow (hue=dir, bright=speed)")

    # 3) vector field
    p3 = draw_vectors(gray, flow)
    _panel_title(p3, "3 Flow vectors (x3 length)")

    # 4) camera-compensated motion
    heat = np.clip(residual_mag / max(3 * thr, 1e-6) * 255, 0, 255).astype(np.uint8)
    p4 = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p4, contours, -1, (80, 255, 80), 1)
    _label(p4, f"threshold {thr:.2f}px/f", (4, 13), 0.38)
    _panel_title(p4, "4 Camera-compensated motion")

    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


# ---------------------------------------------------------- the job --

def _write_status(run_dir, **kw):
    tmp = os.path.join(run_dir, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(kw, f)
    os.replace(tmp, os.path.join(run_dir, "status.json"))


def read_status(run_dir):
    try:
        with open(os.path.join(run_dir, "status.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_live(run_dir, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        tmp = os.path.join(run_dir, "live.jpg.tmp")
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, os.path.join(run_dir, "live.jpg"))


def process_video(video_path, run_dir, label, start_sec=0.0, max_seconds=30.0):
    """Full pipeline for one video -> files in run_dir. Safe to run in a
    background thread; progress goes to status.json / live.jpg."""
    try:
        _process(video_path, run_dir, label, start_sec, max_seconds)
    except Exception as e:  # surface any failure to the polling page
        _write_status(run_dir, state="error", message=str(e))


def _process(video_path, run_dir, label, start_sec, max_seconds):
    t_start = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open that video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if fps > 1 else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0.0

    stride = max(1, int(round(fps / TARGET_FPS)))
    out_fps = fps / stride
    start_idx = int(start_sec * fps)
    if total_frames and start_idx >= total_frames - 2:
        raise ValueError(f"Start time {start_sec}s is past the end of the clip ({duration:.1f}s).")
    end_idx = start_idx + int(max_seconds * fps)
    if total_frames:
        end_idx = min(end_idx, total_frames)
    n_expected = max(1, (end_idx - start_idx) // stride - 1)
    if start_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    ok, frame = cap.read()
    if not ok:
        raise ValueError("Could not read frames from that video.")
    src_h, src_w = frame.shape[:2]
    scale = min(1.0, ANALYSIS_MAX_DIM / max(src_h, src_w))
    w = int(src_w * scale) // 2 * 2
    h = int(src_h * scale) // 2 * 2

    def prep(f):
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)

    prev = prep(frame)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_idx = start_idx

    wheel = draw_direction_wheel(size=min(56, h // 5))
    video_path_out = os.path.join(run_dir, "flow_tracking.webm")
    writer = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*"VP80"), out_fps, (2 * w, 2 * h))
    if not writer.isOpened():
        raise ValueError("This OpenCV build cannot write WebM/VP8 video.")

    tracker = CentroidTracker(max_dist=0.15 * max(w, h))
    tracks = []  # list of point trails [(x, y), ...]
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    series = []
    keyframes = {}
    best = {"motion": (-1, None), "objects": (-1, None)}
    pair_choice = (-1.0, start_idx)
    i = 0

    while True:
        for _ in range(stride - 1):
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.read()
        frame_idx += 1
        if not ok or frame_idx >= end_idx:
            break
        cur = prep(frame)
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        t_sec = (frame_idx - start_idx) / fps

        # --- KLT feature tracks (forward-backward checked) ---
        if tracks:
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, **lk_params)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None, **lk_params)
            fb = np.abs(p0 - p0r).reshape(-1, 2).max(axis=1)
            good = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
            prev_pts = p0.reshape(-1, 2)[good]
            next_pts = p1.reshape(-1, 2)[good]
            tracks = [tr + [tuple(p)] for tr, p, g in zip(tracks, p1.reshape(-1, 2), good) if g]
            tracks = [tr[-20:] for tr in tracks]
        else:
            prev_pts = next_pts = np.zeros((0, 2), np.float32)

        if i % 5 == 0 and len(tracks) < 150:
            mask = np.full((h, w), 255, np.uint8)
            for tr in tracks:
                cv2.circle(mask, (int(tr[-1][0]), int(tr[-1][1])), 6, 0, -1)
            new = cv2.goodFeaturesToTrack(cur_gray, maxCorners=200 - len(tracks), qualityLevel=0.01,
                                          minDistance=7, mask=mask)
            if new is not None:
                tracks += [[tuple(p)] for p in new.reshape(-1, 2)]

        # --- dense flow, camera motion, residual motion, objects ---
        flow = dense_flow(prev_gray, cur_gray)
        M, cam = estimate_camera_motion(prev_pts, next_pts)
        residual = flow - camera_flow_field(M, h, w)
        res_mag = np.hypot(residual[..., 0], residual[..., 1])
        objects, obj_mask, thr = detect_moving_objects(res_mag)
        for o in objects:
            sel = obj_mask[o["box"][1]:o["box"][1] + o["box"][3], o["box"][0]:o["box"][0] + o["box"][2]] > 0
            ru = residual[o["box"][1]:o["box"][1] + o["box"][3], o["box"][0]:o["box"][0] + o["box"][2], 0][sel]
            rv = residual[o["box"][1]:o["box"][1] + o["box"][3], o["box"][0]:o["box"][0] + o["box"][2], 1][sel]
            mu, mv = float(np.mean(ru)), float(np.mean(rv))
            o["speed"] = float(np.hypot(mu, mv))
            o["dir_deg"] = float((np.degrees(np.arctan2(mv, mu)) + 360) % 360)
        tracker.update(objects, t_sec)

        moving_flags = []
        for tr in tracks:
            x, y = int(min(w - 1, max(0, tr[-1][0]))), int(min(h - 1, max(0, tr[-1][1])))
            moving_flags.append(bool(obj_mask[y, x]))

        stats = flow_stats(flow)
        gx = cv2.Sobel(cur_gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(cur_gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        stats.update({
            "frame": frame_idx, "t_sec": t_sec,
            "cam_tx": cam["tx"], "cam_ty": cam["ty"], "cam_rot_deg": cam["rot_deg"], "cam_scale": cam["scale"],
            "n_objects": len(objects), "object_area_frac": float(obj_mask.mean()),
            "residual_mean": float(res_mag.mean()), "n_tracks": len(tracks),
            "low_texture_frac": float((np.hypot(gx, gy) < 2.0).mean()),
            "object_speeds": [round(o["speed"], 3) for o in objects],
        })
        series.append(stats)

        panel = render_panels(cur, cur_gray, flow, res_mag, obj_mask, objects, tracks, moving_flags, cam, thr,
                              t_sec, wheel, tracker)
        writer.write(panel)

        if i == 0:
            keyframes["start"] = (t_sec, panel.copy())
        if i == n_expected // 2:
            keyframes["middle"] = (t_sec, panel.copy())
        if i > 0 and stats["moving_frac"] > best["motion"][0]:
            best["motion"] = (stats["moving_frac"], (t_sec, panel.copy()))
        if len(objects) > best["objects"][0]:
            best["objects"] = (len(objects), (t_sec, panel.copy()))
        activity = stats["object_area_frac"] + 0.01 * stats["mean_mag"]
        if activity > pair_choice[0]:
            pair_choice = (activity, frame_idx - stride)  # pair = (this frame - stride, next original frame)

        if i % 2 == 0:
            _write_live(run_dir, panel)
        if i % 10 == 0:
            _write_status(run_dir, state="running", progress=min(0.99, i / n_expected), frame=i,
                          n_total=n_expected, t_sec=round(t_sec, 2), label=label,
                          elapsed=round(time.time() - t_start, 1))

        prev_gray = cur_gray
        i += 1

    cap.release()
    writer.release()
    if i < 2:
        raise ValueError("Need at least 3 readable frames - clip too short or unreadable.")

    _write_status(run_dir, state="running", progress=0.99, frame=i, n_total=n_expected, label=label,
                  message="writing plots and summary")

    keyframes["peak_motion"] = best["motion"][1]
    keyframes["most_objects"] = best["objects"][1]
    keyframe_files = []
    for name in ("start", "middle", "peak_motion", "most_objects"):
        if keyframes.get(name) is None:
            continue
        t_kf, img = keyframes[name]
        fn = f"keyframe_{name}.png"
        cv2.imwrite(os.path.join(run_dir, fn), img)
        keyframe_files.append({"name": name.replace("_", " "), "file": fn, "t_sec": round(t_kf, 2)})

    _save_stats_csv(series, os.path.join(run_dir, "stats.csv"))
    objects_log = _save_objects_csv(tracker, os.path.join(run_dir, "objects.csv"), out_fps)
    _plot_stats(series, out_fps, os.path.join(run_dir, "stats.png"))

    # Consecutive ORIGINAL frames (full resolution) for the tracking page.
    pair_idx = pair_choice[1]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pair_idx)
    ok1, f1 = cap.read()
    ok2, f2 = cap.read()
    cap.release()
    has_pair = bool(ok1 and ok2)
    if has_pair:
        cv2.imwrite(os.path.join(run_dir, "pair_frame_t.png"), f1)
        cv2.imwrite(os.path.join(run_dir, "pair_frame_t1.png"), f2)

    overall = _overall(series, out_fps)
    inferences = infer(series, out_fps, objects_log)
    summary = {
        "label": label,
        "video_name": os.path.basename(video_path).split("_", 1)[-1],
        "src_fps": fps, "out_fps": out_fps, "stride": stride,
        "src_size": [src_w, src_h], "analysis_size": [w, h],
        "clip_duration_sec": round(duration, 2), "start_sec": start_sec,
        "processed_sec": round(series[-1]["t_sec"], 2), "n_pairs": len(series),
        "overall": overall, "inferences": inferences, "keyframes": keyframe_files,
        "objects": objects_log[:12], "has_pair": has_pair,
        "pair_frames": [pair_idx, pair_idx + 1], "pair_t_sec": round((pair_idx - start_idx) / fps + start_sec, 3),
        "processing_sec": round(time.time() - t_start, 1),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_summary_md(summary, os.path.join(run_dir, "summary.md"))
    _write_status(run_dir, state="done", progress=1.0, frame=i, n_total=n_expected, label=label)


# ------------------------------------------------------ evidence/report --

def _save_stats_csv(series, path):
    keys = ["frame", "t_sec", "mean_mag", "max_mag", "moving_frac", "dominant_dir_deg", "divergence",
            "cam_tx", "cam_ty", "cam_rot_deg", "cam_scale", "n_objects", "object_area_frac",
            "residual_mean", "n_tracks", "low_texture_frac"]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for s in series:
            wr.writerow([round(s[k], 5) if isinstance(s[k], float) else s[k] for k in keys])


def _save_objects_csv(tracker, path, out_fps):
    rows = []
    for oid, hst in tracker.history.items():
        n = hst["n_frames"]
        mean_speed = hst["speed_sum"] / n if n else 0.0
        direction = (np.degrees(np.arctan2(hst["dir_sin"], hst["dir_cos"])) + 360) % 360
        rows.append({
            "id": oid, "first_t": round(hst["first_t"], 2), "last_t": round(hst["last_t"], 2),
            "duration_s": round(hst["last_t"] - hst["first_t"], 2), "n_frames": n,
            "mean_speed_px_per_frame": round(mean_speed, 2), "mean_speed_px_per_s": round(mean_speed * out_fps, 1),
            "direction_deg": round(float(direction), 1), "direction": direction_name(direction),
            "path_px": round(hst["path_px"], 1), "max_area_px": hst["max_area"],
        })
    rows.sort(key=lambda r: -r["n_frames"])
    with open(path, "w", newline="") as f:
        keys = ["id", "first_t", "last_t", "duration_s", "n_frames", "mean_speed_px_per_frame",
                "mean_speed_px_per_s", "direction_deg", "direction", "path_px", "max_area_px"]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def _overall(series, out_fps):
    mm = np.array([s["mean_mag"] for s in series])
    mf = np.array([s["moving_frac"] for s in series])
    no = np.array([s["n_objects"] for s in series])
    cam = np.hypot([s["cam_tx"] for s in series], [s["cam_ty"] for s in series])
    return {
        "peak_mean_mag": float(mm.max()), "avg_mean_mag": float(mm.mean()),
        "peak_t": float(series[int(mf[1:].argmax()) + 1 if len(mf) > 1 else 0]["t_sec"]),
        "peak_moving_frac": float(mf.max() * 100), "avg_moving_frac": float(mf.mean() * 100),
        "max_objects": int(no.max()), "frames_with_objects_pct": float((no > 0).mean() * 100),
        "median_cam_shift": float(np.median(cam)), "out_fps": out_fps,
    }


def infer(series, out_fps, objects_log):
    """Turn the per-frame numbers into plain-language conclusions, each
    one quoting the measurement it rests on."""
    out = []
    t = np.array([s["t_sec"] for s in series])
    mm = np.array([s["mean_mag"] for s in series])
    no = np.array([s["n_objects"] for s in series])
    tx = np.array([s["cam_tx"] for s in series])
    ty = np.array([s["cam_ty"] for s in series])
    sc = np.array([s["cam_scale"] for s in series])
    rot = np.array([s["cam_rot_deg"] for s in series])
    div = np.array([s["divergence"] for s in series])
    cam = np.hypot(tx, ty)

    mf = np.array([s["moving_frac"] for s in series])
    active = (mf > 0.005) | (no > 0)
    pk = int(mf[1:].argmax()) + 1 if len(mf) > 1 else 0  # frame 0 skipped: clip starts often glitch
    out.append(f"When: something was moving (>0.5% of pixels moving more than 1 px/frame, or a moving object "
               f"detected) in {active.mean() * 100:.0f}% of the {t[-1]:.1f}s analysed. Peak activity at "
               f"t={t[pk]:.2f}s, when {mf[pk] * 100:.1f}% of the frame moved; the fastest pixels then moved "
               f"{series[pk]['max_mag']:.1f} px/frame (~{series[pk]['max_mag'] * out_fps:.0f} px/s at analysis resolution).")
    quiet = t[~active]
    if len(quiet):
        out.append(f"Static periods: {len(quiet) / len(t) * 100:.0f}% of frames are essentially still "
                   f"(no object, <0.5% of pixels moving), e.g. around t={quiet[len(quiet) // 2]:.1f}s.")

    moving_cam = cam > 0.5
    if moving_cam.mean() > 0.3:
        mtx, mty = float(np.median(tx[moving_cam])), float(np.median(ty[moving_cam]))
        content_dir = (np.degrees(np.arctan2(mty, mtx)) + 360) % 360
        cam_dir = direction_name(content_dir + 180)
        out.append(f"Camera ego-motion: the whole image shifted in {moving_cam.mean() * 100:.0f}% of frames "
                   f"(median shift ({mtx:+.2f}, {mty:+.2f}) px/frame, image content moving {direction_name(content_dir)}) "
                   f"- i.e. the camera itself was panning/moving {cam_dir}. This is global, smooth flow over the "
                   f"whole frame, unlike an object's localized flow.")
    else:
        out.append(f"Camera ego-motion: essentially static (median global shift {np.median(cam):.2f} px/frame) - "
                   f"so the flow that does appear comes from objects moving in the scene, not the camera.")
    zoom = (sc - 1) * 100
    if np.abs(zoom).mean() > 0.15:
        out.append(f"Zoom / depth motion: average frame-to-frame scale change {zoom.mean():+.2f}% "
                   f"({'expanding - camera moving forward or zooming in' if zoom.mean() > 0 else 'contracting - camera moving back or zooming out'}).")
    if np.abs(rot).mean() > 0.1:
        out.append(f"Rotation: the camera rolled on average {rot.mean():+.2f} deg/frame "
                   f"({np.abs(rot).sum():.1f} deg of total roll over the clip).")

    if no.max() > 0:
        lasting = [o for o in objects_log if o["duration_s"] >= 0.5]
        out.append(f"What moved: independent moving regions (motion left after removing camera motion) were "
                   f"present in {(no > 0).mean() * 100:.0f}% of frames, up to {no.max()} at once; "
                   f"{len(lasting)} object track(s) persisted for at least 0.5s.")
        for o in lasting[:3]:
            out.append(f"Object {o['id']}: visible t={o['first_t']}-{o['last_t']}s, moving {o['direction']} at "
                       f"~{o['mean_speed_px_per_frame']} px/frame ({o['mean_speed_px_per_s']} px/s), "
                       f"centroid travelled {o['path_px']} px.")
    else:
        out.append("What moved: no region moved independently of the camera above the noise threshold.")

    moving_dirs = [s["dominant_dir_deg"] for s in series if s["moving_frac"] > 0.01]
    if moving_dirs:
        d = _circular_mean_deg(moving_dirs)
        out.append(f"Direction: the dominant flow direction over moving frames was {direction_name(d)} "
                   f"({d:.0f} deg in image coords, 0 = right, 90 = down).")

    if active.any():
        md = float(div[active].mean())
        if abs(md) > 0.003:
            out.append(f"Divergence (du/dx + dv/dy) averaged {md:+.4f} on active frames - "
                       f"{'positive: the field expands, i.e. things approach the camera (time-to-contact cue)' if md > 0 else 'negative: the field contracts, i.e. things recede from the camera'}.")

    ltf = float(np.mean([s["low_texture_frac"] for s in series]))
    out.append(f"Reliability: {ltf * 100:.0f}% of pixels have almost no texture (|gradient| < 2 grey levels/px). "
               f"There brightness constancy gives no constraint (aperture problem), so flow in those areas is "
               f"filled in by smoothing rather than measured.")
    return out


def _plot_stats(series, out_fps, path):
    t = [s["t_sec"] for s in series]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    ax.plot(t, [s["mean_mag"] for s in series], color=PRIMARY, label="mean |flow| (px/frame)")
    ax.plot(t, np.hypot([s["cam_tx"] for s in series], [s["cam_ty"] for s in series]), color="#64748b",
            label="camera shift (px/frame)")
    ax.plot(t, [s["residual_mean"] for s in series], color=ACCENT, label="mean residual (object) motion")
    ax.set_xlabel("time (s)")
    ax.set_title("Speed over time", fontsize=11, fontweight="700")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t, [s["moving_frac"] * 100 for s in series], color=PRIMARY, label="% pixels with |flow| > 1px")
    ax.plot(t, [s["object_area_frac"] * 100 for s in series], color=ACCENT, label="% area of moving objects")
    ax2 = ax.twinx()
    ax2.step(t, [s["n_objects"] for s in series], color="#16a34a", where="post", label="# objects", alpha=0.7)
    ax2.set_ylabel("# objects", color="#16a34a")
    ax.set_xlabel("time (s)")
    ax.set_title("How much of the frame moved", fontsize=11, fontweight="700")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    ax.plot(t, [s["cam_tx"] for s in series], color=PRIMARY, label="camera tx")
    ax.plot(t, [s["cam_ty"] for s in series], color=ACCENT, label="camera ty")
    ax.plot(t, [s["divergence"] * 100 for s in series], color="#16a34a", alpha=0.7, label="divergence x100")
    ax.axhline(0, color="#cbd5e1", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_title("Camera motion and divergence", fontsize=11, fontweight="700")
    ax.legend(fontsize=8)

    fig.delaxes(axes[1, 1])
    ax = fig.add_subplot(2, 2, 4, projection="polar")
    dirs = np.radians([s["dominant_dir_deg"] for s in series if s["moving_frac"] > 0.01])
    if len(dirs):
        ax.hist(dirs, bins=24, color=PRIMARY)
    ax.set_theta_direction(-1)  # image coords: 90 deg = down
    ax.set_title("Dominant flow direction (0=right, 90=down)", fontsize=11, fontweight="700")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_summary_md(s, path):
    o = s["overall"]
    lines = [
        f"# Optical flow summary - {s['label']} ({s['video_name']})", "",
        f"- Source: {s['src_size'][0]}x{s['src_size'][1]} @ {s['src_fps']:.2f} fps, clip length {s['clip_duration_sec']} s",
        f"- Analysed: {s['processed_sec']} s from t={s['start_sec']} s, every {s['stride']} frame(s) "
        f"-> {s['out_fps']:.2f} fps, at {s['analysis_size'][0]}x{s['analysis_size'][1]} ({s['n_pairs']} frame pairs)",
        f"- Peak mean |flow| {o['peak_mean_mag']:.2f} px/frame at t={o['peak_t']:.2f} s; average {o['avg_mean_mag']:.2f}",
        f"- Moving area: peak {o['peak_moving_frac']:.1f}%, average {o['avg_moving_frac']:.1f}%",
        f"- Moving objects: up to {o['max_objects']} at once, present in {o['frames_with_objects_pct']:.0f}% of frames",
        "", "## What the optical flow tells us", "",
    ]
    lines += [f"- {x}" for x in s["inferences"]]
    lines += ["", "## Object tracks (longest first)", "",
              "| id | t start | t end | mean speed (px/frame) | direction | path (px) |", "|---|---|---|---|---|---|"]
    for ob in s["objects"]:
        lines.append(f"| {ob['id']} | {ob['first_t']} | {ob['last_t']} | {ob['mean_speed_px_per_frame']} | "
                     f"{ob['direction']} | {ob['path_px']} |")
    lines += ["", "## Files", "",
              "- flow_tracking.webm - 2x2 visualization video (plays at the analysed frame rate = real time)",
              "- stats.csv - per-frame numbers; stats.png - plots", "- objects.csv - every object track",
              "- keyframe_*.png - snapshots", "- pair_frame_t.png / pair_frame_t1.png - consecutive full-res frames for tracking validation"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
