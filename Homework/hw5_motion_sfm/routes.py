# Web glue for HW5 - three sub-pages sharing one blueprint:
#   /hw5/flow      dense optical flow on an uploaded video clip
#   /hw5/tracking  from-scratch Lucas-Kanade validated against OpenCV, on
#                  a frame pair (handed off from /flow, or uploaded directly)
#   /hw5/sfm       4-viewpoint planar structure-from-motion

import json
import os
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash

from Homework.hw2_calibration.routes import get_current_calibration
from . import optical_flow as of
from . import tracking as trk
from . import sfm_planar as sfm

hw5 = Blueprint("hw5", __name__, url_prefix="/hw5", template_folder="templates", static_folder="static")

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(MODULE_ROOT, "static")
VIDEO_UPLOAD_DIR = os.path.join(STATIC_ROOT, "video_uploads")
SFM_UPLOAD_DIR = os.path.join(STATIC_ROOT, "sfm_uploads")
OUTPUT_DIR = os.path.join(STATIC_ROOT, "outputs")
for d in (VIDEO_UPLOAD_DIR, SFM_UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

PRIMARY = "#4f46e5"
ACCENT = "#f97316"


def _stamp():
    return str(int(time.time() * 1000))


def _save_png(image_bgr, tag, stamp):
    filename = f"{stamp}_{tag}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), image_bgr)
    return url_for("hw5.static", filename=f"outputs/{filename}")


def _static_url(abs_path):
    rel = os.path.relpath(abs_path, STATIC_ROOT)
    return url_for("hw5.static", filename=rel.replace(os.sep, "/"))


def _static_abs(url_path):
    """Reverse of url_for('hw5.static', ...): turn '/hw5/static/...' back
    into an absolute filesystem path so a route can re-read an image it
    (or another route) already saved."""
    rel = url_path.split("/static/", 1)[-1]
    return os.path.join(STATIC_ROOT, rel)


@hw5.route("/")
def home():
    return render_template("hw5/overview.html")


# ---------------------------------------------------------------- flow --

@hw5.route("/flow", methods=["GET", "POST"])
def flow_view():
    if request.method == "GET":
        return render_template("hw5/optical_flow.html", result=None)

    f = request.files.get("video")
    if not f or not f.filename:
        flash("Choose a video file first.")
        return redirect(url_for("hw5.flow_view"))

    stamp = _stamp()
    in_path = os.path.join(VIDEO_UPLOAD_DIR, f"{stamp}_{f.filename}")
    f.save(in_path)

    max_seconds = float(request.form.get("max_seconds", 30))

    try:
        out = of.process_video(in_path, OUTPUT_DIR, stamp, max_seconds=max_seconds)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for("hw5.flow_view"))

    flow_video_url = _static_url(out["flow_video_path"])

    series = out["series"]
    t = [s["t_sec"] for s in series]
    mean_mag = [s["mean_mag"] for s in series]
    moving_frac = [s["moving_frac"] * 100 for s in series]
    directions = [s["dominant_dir_deg"] for s in series if s["moving_frac"] > 0.01]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("white")
    axes[0].plot(t, mean_mag, color=PRIMARY, label="mean flow magnitude (px/frame)")
    axes[0].plot(t, moving_frac, color=ACCENT, label="% pixels moving (>1px)")
    axes[0].set_xlabel("time (s)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Motion over time", fontsize=11, fontweight="700")

    axes[1].hist(directions, bins=24, range=(0, 360), color=PRIMARY)
    axes[1].set_xlabel("flow direction (deg, 0=+x)")
    axes[1].set_title("Dominant motion direction histogram", fontsize=11, fontweight="700")
    fig.tight_layout()
    stats_plot_path = os.path.join(OUTPUT_DIR, f"{stamp}_flow_stats.png")
    fig.savefig(stats_plot_path, dpi=140)
    plt.close(fig)

    sample_urls = []
    for idx in sorted(out["sample_frames"].keys()):
        sf = out["sample_frames"][idx]
        sample_urls.append({
            "t_sec": sf["stats"]["t_sec"],
            "frame_url": _save_png(sf["frame_bgr"], f"sample{idx}_frame", stamp),
            "flow_url": _save_png(sf["flow_color"], f"sample{idx}_flowcolor", stamp),
            "stats": sf["stats"],
        })

    overall = {
        "peak_mean_mag": float(np.max(mean_mag)),
        "avg_mean_mag": float(np.mean(mean_mag)),
        "peak_moving_frac": float(np.max(moving_frac)),
        "avg_moving_frac": float(np.mean(moving_frac)),
    }

    # hand the middle consecutive frame pair to the tracking page for the
    # two-frame validation step, without making the user re-upload
    mid_idx = sorted(out["sample_frames"].keys())[len(out["sample_frames"]) // 2]
    cap = cv2.VideoCapture(in_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ok1, prev_frame = cap.read()
    ok2, next_frame = cap.read()
    cap.release()
    prev_next_urls = None
    if ok1 and ok2:
        prev_next_urls = (
            _save_png(prev_frame, "trackpair_prev", stamp),
            _save_png(next_frame, "trackpair_next", stamp),
        )

    result = {
        "flow_video_url": flow_video_url,
        "stats_plot_url": _static_url(stats_plot_path),
        "samples": sample_urls,
        "overall": overall,
        "fps": out["fps"],
        "n_frames_used": out["n_frames_used"],
        "prev_next_urls": prev_next_urls,
    }
    return render_template("hw5/optical_flow.html", result=result)


# ------------------------------------------------------------- tracking --

@hw5.route("/tracking", methods=["GET"])
def tracking_view():
    prev_url = request.args.get("prev")
    next_url = request.args.get("next")
    return render_template("hw5/tracking.html", prev_url=prev_url, next_url=next_url, result=None)


@hw5.route("/tracking/upload", methods=["POST"])
def tracking_upload():
    prev_f = request.files.get("prev_frame")
    next_f = request.files.get("next_frame")
    if not prev_f or not next_f or not prev_f.filename or not next_f.filename:
        flash("Choose both frame images.")
        return redirect(url_for("hw5.tracking_view"))

    stamp = _stamp()
    prev_path = os.path.join(SFM_UPLOAD_DIR, f"{stamp}_prev_{prev_f.filename}")
    next_path = os.path.join(SFM_UPLOAD_DIR, f"{stamp}_next_{next_f.filename}")
    prev_f.save(prev_path)
    next_f.save(next_path)

    return redirect(url_for("hw5.tracking_view", prev=_static_url(prev_path), next=_static_url(next_path)))


@hw5.route("/tracking/run", methods=["POST"])
def tracking_run():
    prev_url = request.form.get("prev_url")
    next_url = request.form.get("next_url")
    points_json = request.form.get("points")
    window = int(request.form.get("window", 21))

    try:
        points = json.loads(points_json)
        points = [(float(p[0]), float(p[1])) for p in points]
    except (TypeError, ValueError, json.JSONDecodeError):
        flash("Click at least one point on the image before running.")
        return redirect(url_for("hw5.tracking_view", prev=prev_url, next=next_url))

    if not points:
        flash("Click at least one point on the image before running.")
        return redirect(url_for("hw5.tracking_view", prev=prev_url, next=next_url))

    prev_bgr = cv2.imread(_static_abs(prev_url))
    next_bgr = cv2.imread(_static_abs(next_url))
    if prev_bgr is None or next_bgr is None:
        flash("Could not re-read the uploaded frames.")
        return redirect(url_for("hw5.tracking_view"))

    rows = trk.validate_tracking(prev_bgr, next_bgr, points, window=window)

    annotated = next_bgr.copy()
    for r in rows:
        p_manual = (int(round(r["x_pred"])), int(round(r["y_pred"])))
        p_actual = (int(round(r["x_actual"])), int(round(r["y_actual"])))
        cv2.circle(annotated, p_manual, 5, (0, 0, 255), 2)     # red = manual LK prediction
        cv2.circle(annotated, p_actual, 5, (0, 200, 0), 2)     # green = OpenCV reference "actual"
        cv2.line(annotated, p_manual, p_actual, (0, 200, 255), 1)

    stamp = _stamp()
    annotated_url = _save_png(annotated, "tracking_result", stamp)

    mean_err = float(np.mean([r["agreement_err_px"] for r in rows]))
    max_err = float(np.max([r["agreement_err_px"] for r in rows]))

    result = {
        "rows": rows,
        "annotated_url": annotated_url,
        "mean_err": mean_err,
        "max_err": max_err,
        "window": window,
    }
    return render_template("hw5/tracking.html", prev_url=prev_url, next_url=next_url, result=result)


# ------------------------------------------------------------------ sfm --

@hw5.route("/sfm", methods=["GET"])
def sfm_view():
    return render_template("hw5/sfm.html", stage="upload", views=None, result=None)


@hw5.route("/sfm/upload", methods=["POST"])
def sfm_upload():
    files = request.files.getlist("images")
    if len(files) != 4 or any(not f.filename for f in files):
        flash("Upload exactly 4 images, one per viewpoint.")
        return redirect(url_for("hw5.sfm_view"))

    width_mm = float(request.form.get("width_mm", 210))
    height_mm = float(request.form.get("height_mm", 297))
    n_boundary = int(request.form.get("n_boundary", 4))
    use_hw2_calib = request.form.get("use_hw2_calib") == "on"

    stamp = _stamp()
    image_urls = []
    for i, f in enumerate(files):
        path = os.path.join(SFM_UPLOAD_DIR, f"{stamp}_view{i}_{f.filename}")
        f.save(path)
        image_urls.append(_static_url(path))

    views_meta = {
        "stamp": stamp,
        "image_urls": image_urls,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "n_boundary": n_boundary,
        "use_hw2_calib": use_hw2_calib,
        "n_clicks": 4 + n_boundary,
    }
    return render_template("hw5/sfm.html", stage="click", views=views_meta, result=None)


@hw5.route("/sfm/run", methods=["POST"])
def sfm_run():
    try:
        width_mm = float(request.form["width_mm"])
        height_mm = float(request.form["height_mm"])
        n_boundary = int(request.form["n_boundary"])
        use_hw2_calib = request.form.get("use_hw2_calib") == "on"
        image_urls = json.loads(request.form["image_urls"])
        points_per_view = [json.loads(request.form[f"points_{i}"]) for i in range(4)]
    except (KeyError, ValueError, json.JSONDecodeError):
        flash("Something was missing from the point-marking step - please redo it.")
        return redirect(url_for("hw5.sfm_view"))

    n_needed = 4 + n_boundary
    if any(len(p) != n_needed for p in points_per_view):
        flash(f"Each image needs exactly {n_needed} clicks (4 corners + {n_boundary} boundary points).")
        return redirect(url_for("hw5.sfm_view"))

    world_corners = sfm.rectangle_world_corners(width_mm, height_mm)

    views = []
    image_size = None
    for url, pts in zip(image_urls, points_per_view):
        img = cv2.imread(_static_abs(url))
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        pts = [(float(x), float(y)) for x, y in pts]
        views.append({"corners": pts[:4], "boundary": pts[4:]})

    calib = get_current_calibration() if use_hw2_calib else None
    if calib is not None:
        K = sfm.scale_intrinsics(calib["camera_matrix"], calib["image_size"], image_size)
        dist = calib["dist_coeffs"]
        k_source = "HW2 calibration (rescaled to this image size)"
    else:
        K = sfm.default_intrinsics(image_size[0], image_size[1])
        dist = None
        k_source = "approximate (60 deg horizontal FOV assumption - no HW2 calibration on file)"

    result_data = sfm.reconstruct(views, world_corners, K, dist)

    stamp = _stamp()
    scene_path = os.path.join(OUTPUT_DIR, f"{stamp}_sfm_scene.png")
    topdown_path = os.path.join(OUTPUT_DIR, f"{stamp}_sfm_topdown.png")
    sfm.plot_scene(result_data, world_corners, scene_path)
    sfm.plot_topdown(result_data, world_corners, topdown_path)

    cameras = []
    for i, pose in enumerate(result_data["poses"]):
        cameras.append({
            "id": i + 1,
            "R": pose["R"].round(4).tolist(),
            "t": pose["t"].round(2).tolist(),
            "C": pose["C"].round(2).tolist(),
            "ok": pose["ok"],
        })

    result = {
        "cameras": cameras,
        "K": K.round(2).tolist(),
        "k_source": k_source,
        "image_size": image_size,
        "corner_recovery_error": [round(e, 3) for e in result_data["corner_recovery_error"]],
        "planarity_rms": round(result_data["planarity_rms"], 3),
        "boundary_3d": [list(np.round(p, 2)) for p in result_data["boundary_3d"]],
        "scene_url": _static_url(scene_path),
        "topdown_url": _static_url(topdown_path),
        "image_urls": image_urls,
        "width_mm": width_mm,
        "height_mm": height_mm,
    }
    return render_template("hw5/sfm.html", stage="result", views=None, result=result)
