# Web glue for HW7 - two pages sharing one blueprint:
#   /hw7/stereo    calibrated stereo object measurement (part 1)
#   /hw7/room      room dimensions from video (part 3)
#
# Part 2 is a handwritten derivation, so it has no page. As in HW5, every run
# writes into its own folder outputs/<run_id>/, and the video reconstruction
# runs as a background job with its status in a file.

import io
import json
import os
import re
import shutil
import threading
import time
import zipfile

import cv2
import numpy as np
from flask import (Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, session,
                   url_for)
from werkzeug.utils import secure_filename

from Homework import samples
from Homework.hw2_calibration.routes import calibration_summary, get_current_calibration
from Homework.hw5_motion_sfm import sfm_planar as sfm
from Homework.hw5_motion_sfm.optical_flow import _write_status, read_status
from . import room_video as room
from . import stereo_measure as sm

hw7 = Blueprint("hw7", __name__, url_prefix="/hw7", template_folder="templates", static_folder="static")

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(MODULE_ROOT, "static")
STEREO_UPLOAD_DIR = os.path.join(STATIC_ROOT, "stereo_uploads")
ROOM_UPLOAD_DIR = os.path.join(STATIC_ROOT, "room_uploads")
OUTPUT_DIR = os.path.join(STATIC_ROOT, "outputs")
for d in (STEREO_UPLOAD_DIR, ROOM_UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

RUN_ID_RE = re.compile(r"^(stereo|room)_[0-9]+$")
SHAPE_KINDS = ("rectangle", "circle", "polygon")

# The measurements taken for this homework's captures; form defaults only,
# every one of them can be changed on the page.
STEREO_BASELINE_MM = 297.0
STEREO_OBJECT_DISTANCE_MM = 1188.0
ROOM_TRAVEL_MM = 1260.0
ROOM_WALL_DISTANCE_MM = 2673.0

_room_jobs = set()      # runs with a live thread in this process
_room_jobs_lock = threading.Lock()


class InputError(ValueError):
    """A form problem the user can fix; shown as-is in the UI."""


def _new_run(kind):
    run_id = f"{kind}_{int(time.time() * 1000)}"
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir


def _run_dir(run_id):
    if not run_id or not RUN_ID_RE.match(run_id):
        return None
    d = os.path.join(OUTPUT_DIR, run_id)
    return d if os.path.isdir(d) else None


def _run_url(run_id, filename):
    return url_for("hw7.static", filename=f"outputs/{run_id}/{filename}")


def _read_json(run_dir, name):
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(run_dir, name, data):
    with open(os.path.join(run_dir, name), "w") as f:
        json.dump(data, f, indent=1)


def _number(form, name, label, required=False):
    """A positive form number, or None when optional and blank."""
    raw = (form.get(name) or "").strip()
    if not raw:
        if required:
            raise InputError(f"{label} is required.")
        return None
    try:
        v = float(raw)
    except ValueError:
        raise InputError(f"{label} must be a number.")
    if not np.isfinite(v) or v <= 0:
        raise InputError(f"{label} must be greater than zero.")
    return v


def _intrinsics(form, image_size, exif_path=None):
    """K, distortion and a description, from the source chosen on the form.
    Anything missing is an error: a guessed K would give invented sizes."""
    w, h = image_size
    source = form.get("calib_source")
    if source == "hw2":
        calib = get_current_calibration()
        if calib is None:
            raise InputError("No saved HW2 calibration. Run HW2 calibration first, or enter the parameters by hand.")
        K, swapped = sm.adapt_calibration(calib["camera_matrix"], calib["image_size"], image_size)
        dist = np.asarray(calib["dist_coeffs"], dtype=np.float64).ravel().copy()
        if swapped:  # tangential terms don't survive the x/y swap; radial ones do
            dist[2:4] = 0.0
        return K, dist, ("HW2 chessboard calibration adapted to this image size. Only valid if these images came "
                         "from the same camera.")
    if source == "exif":
        K, f35 = sfm.exif_intrinsics(exif_path, w, h) if exif_path else (None, None)
        if K is None:
            raise InputError("This image has no EXIF focal length. Upload the original photo file, "
                             "or use HW2 calibration or manual parameters.")
        return K, None, f"Photo EXIF: {f35:.0f} mm 35mm-equivalent focal length, principal point at the image centre"
    if source == "manual":
        fx = _number(form, "fx", "fx", required=True)
        fy = _number(form, "fy", "fy") or fx
        cx = _number(form, "cx", "cx") or w / 2.0
        cy = _number(form, "cy", "cy") or h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        dist = None
        text = (form.get("dist") or "").strip()
        if text:
            try:
                dist = np.array([float(v) for v in text.replace(",", " ").split()], dtype=np.float64)
            except ValueError:
                raise InputError("Distortion must be numbers: k1, k2, p1, p2, k3.")
            if len(dist) not in (4, 5):
                raise InputError("Distortion needs 4 or 5 numbers: k1, k2, p1, p2[, k3].")
        return K, dist, "Parameters entered by hand"
    raise InputError("Choose where the camera calibration comes from.")


def _dist_text(dist):
    return ", ".join(f"{v:.4f}" for v in dist) if dist is not None else "none applied"


@hw7.route("/")
def home():
    return render_template("hw7/overview.html")


@hw7.route("/download/<run_id>.zip")
def download_run(run_id):
    run_dir = _run_dir(run_id)
    if run_dir is None:
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(run_dir)):
            if fn in ("status.json", "stereo.npz", "recon.npz") or fn.endswith(".tmp"):
                continue
            zf.write(os.path.join(run_dir, fn), os.path.join(run_id, fn))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"hw7_{run_id}.zip")


# --------------------------------------------------------------- stereo --

def _stereo_view(run_id):
    run_dir = _run_dir(run_id)
    meta = _read_json(run_dir, "meta.json") if run_dir else None
    if meta is None:
        return None
    view = dict(meta)
    view["run_id"] = run_id
    view["urls"] = {k: _run_url(run_id, fn) for k, fn in (
        ("left", "left.jpg"), ("right", "right.jpg"), ("rect_left", "rect_left.jpg"), ("rect_right", "rect_right.jpg"),
        ("disparity", "disparity.png"), ("depth", "depth.png"))}
    view["zip_url"] = url_for("hw7.download_run", run_id=run_id)
    result = _read_json(run_dir, "result.json")
    if result:
        view["result"] = result
        view["annotated_url"] = _run_url(run_id, "measured.jpg")
    return view


@hw7.route("/stereo", methods=["GET"])
def stereo_view():
    run_id = request.args.get("run")
    run = _stereo_view(run_id) if run_id else None
    saved_calib = calibration_summary(get_current_calibration())
    defaults = {"baseline": STEREO_BASELINE_MM, "distance": STEREO_OBJECT_DISTANCE_MM}
    if run is not None:
        offer = samples.offer("hw7", "stereo", run) if run.get("result") else None
        return render_template("hw7/stereo.html", run=run, saved_calib=saved_calib, defaults=defaults,
                               sample_offer=offer, last_run=None)
    last = session.get("hw7_stereo")
    return render_template("hw7/stereo.html", run=None, saved_calib=saved_calib, defaults=defaults,
                           last_run=url_for("hw7.stereo_view", run=last) if last and _run_dir(last) else None)


@hw7.route("/stereo/upload", methods=["POST"])
def stereo_upload():
    back = redirect(url_for("hw7.stereo_view"))
    lf, rf = request.files.get("left"), request.files.get("right")
    if not lf or not rf or not lf.filename or not rf.filename:
        flash("Choose both the left and the right image.")
        return back
    stamp = int(time.time() * 1000)
    left_path = os.path.join(STEREO_UPLOAD_DIR, f"{stamp}_left_{secure_filename(lf.filename) or 'left.jpg'}")
    right_path = os.path.join(STEREO_UPLOAD_DIR, f"{stamp}_right_{secure_filename(rf.filename) or 'right.jpg'}")
    lf.save(left_path)
    rf.save(right_path)
    left, right = cv2.imread(left_path), cv2.imread(right_path)
    if left is None or right is None:
        flash(f"The {'left' if left is None else 'right'} file is not a readable image.")
        return back
    try:
        baseline = _number(request.form, "baseline_mm", "Stereo baseline", required=True)
        distance = _number(request.form, "object_distance_mm", "Camera-to-object distance")
        K, dist, k_source = _intrinsics(request.form, (left.shape[1], left.shape[0]), left_path)
        res = sm.run_stereo(left, right, K, dist, baseline, request.form.get("already_rectified") == "on")
    except (InputError, sm.StereoError) as e:
        flash(str(e))
        return back

    run_id, run_dir = _new_run("stereo")
    jpg = [cv2.IMWRITE_JPEG_QUALITY, 92]
    cv2.imwrite(os.path.join(run_dir, "left.jpg"), res["left"], jpg)
    cv2.imwrite(os.path.join(run_dir, "right.jpg"), res["right"], jpg)
    cv2.imwrite(os.path.join(run_dir, "rect_left.jpg"), res["rect_l"], jpg)
    cv2.imwrite(os.path.join(run_dir, "rect_right.jpg"), res["rect_r"], jpg)
    geom = res["geom"]
    sm.save_map_figure(res["raw"] - geom["doff"], "Disparity", "pixels", os.path.join(run_dir, "disparity.png"))
    sm.save_map_figure(res["depth"], "Depth Z", "mm", os.path.join(run_dir, "depth.png"), cmap="turbo_r")
    np.savez_compressed(os.path.join(run_dir, "stereo.npz"), raw=res["raw"], depth=res["depth"])

    Kw = res["K"]
    meta = {
        "geom": geom, "lo": res["lo"], "hi": res["hi"], "n_matches": res["n_matches"], "n_inliers": res["n_inliers"],
        "n_rows": res["n_rows"], "row_err": round(res["row_err"], 2), "rotation_deg": round(res["rotation_deg"], 2),
        "T": np.round(res["T"], 1).tolist(), "baseline_mm": baseline, "object_distance_mm": distance,
        "already_rectified": request.form.get("already_rectified") == "on", "k_source": k_source,
        "dist_text": _dist_text(dist), "image_size": [res["left"].shape[1], res["left"].shape[0]],
        "K_work": np.round(Kw, 2).tolist(),
        "expected_disparity": None if not distance else round(geom["f"] * baseline / distance, 1),
        "disparity_range": [round(res["lo"] - geom["doff"], 1), round(res["hi"] - geom["doff"], 1)],
        "depth_range": _finite_range(res["depth"]),
    }
    _write_json(run_dir, "meta.json", meta)
    session["hw7_stereo"] = run_id
    return redirect(url_for("hw7.stereo_view", run=run_id) + "#measure")


def _finite_range(values):
    ok = values[np.isfinite(values)]
    if not ok.size:
        return None
    return [round(float(np.percentile(ok, 2)), 1), round(float(np.percentile(ok, 98)), 1)]


def _load_rectified(run_dir):
    left = cv2.imread(os.path.join(run_dir, "rect_left.jpg"))
    right = cv2.imread(os.path.join(run_dir, "rect_right.jpg"))
    raw = np.load(os.path.join(run_dir, "stereo.npz"))["raw"]
    return left, right, raw


@hw7.route("/stereo/match/<run_id>", methods=["POST"])
def stereo_match(run_id):
    """Epipolar match for one click on the rectified left image."""
    run_dir = _run_dir(run_id)
    meta = _read_json(run_dir, "meta.json") if run_dir else None
    if meta is None:
        abort(404)
    data = request.get_json(silent=True) or {}
    try:
        x, y = float(data["x"]), float(data["y"])
    except (KeyError, TypeError, ValueError):
        abort(400)
    left, right, raw = _load_rectified(run_dir)
    d, score, method = sm.locate_point(left, right, raw, x, y, meta["lo"], meta["hi"])
    return jsonify({"d": d, "xr": None if d is None else x - d, "ncc": round(score, 3), "method": method})


def _parse_shapes(text):
    try:
        shapes = json.loads(text or "[]")
        parsed = []
        for s in shapes:
            if s["kind"] not in SHAPE_KINDS:
                raise ValueError
            parsed.append({"kind": s["kind"], "points": [
                {"x": float(p["x"]), "y": float(p["y"]), "xr": None if p.get("xr") is None else float(p["xr"]),
                 "ncc": float(p.get("ncc") or 0.0), "method": str(p.get("method") or "manual")}
                for p in s["points"]]})
    except (TypeError, ValueError, KeyError):
        raise InputError("The selected points could not be read. Please select them again.")
    if not parsed:
        raise InputError("Finish at least one shape before measuring.")
    return parsed


def _draw_shapes(image, shapes):
    out = image.copy()
    s = max(1, round(max(out.shape[:2]) / 1000))
    colours = [(0, 0, 255), (255, 120, 0), (0, 170, 0), (200, 0, 200), (0, 170, 220)]
    for n, shape in enumerate(shapes):
        col = colours[n % len(colours)]
        pts = np.array([[p["x"], p["y"]] for p in shape["points"]], np.int32)
        if shape["kind"] != "circle":
            cv2.polylines(out, [pts], True, col, 2 * s, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts, 1):
            cv2.circle(out, (int(x), int(y)), 5 * s, col, 2 * s, cv2.LINE_AA)
            cv2.putText(out, str(i), (int(x) + 6 * s, int(y) - 6 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, col, s,
                        cv2.LINE_AA)
        cx, cy = pts.mean(axis=0).astype(int)
        text = f"S{n + 1}: {shape['summary']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, s)
        cv2.rectangle(out, (cx - tw // 2 - 4, cy - th - 6), (cx + tw // 2 + 4, cy + 6), (0, 0, 0), -1)
        cv2.putText(out, text, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255), s, cv2.LINE_AA)
    return out


@hw7.route("/stereo/measure/<run_id>", methods=["POST"])
def stereo_measure(run_id):
    run_dir = _run_dir(run_id)
    meta = _read_json(run_dir, "meta.json") if run_dir else None
    if meta is None:
        flash("That stereo run no longer exists. Upload the images again.")
        return redirect(url_for("hw7.stereo_view"))
    back = redirect(url_for("hw7.stereo_view", run=run_id) + "#measure")
    geom = meta["geom"]
    try:
        shapes = _parse_shapes(request.form.get("shapes"))
        all_pts = []
        for shape in shapes:
            for p in shape["points"]:
                if p["xr"] is None:
                    raise InputError(f"The point at ({p['x']:.0f}, {p['y']:.0f}) has no match. Click its position "
                                     f"on the right image.")
                p["d"] = p["x"] - p["xr"] - geom["doff"]
                p["xyz"] = sm.point_3d(p["x"], p["y"], p["x"] - p["xr"], geom)
                all_pts.append(p)

        mean_z = float(np.mean([p["xyz"][2] for p in all_pts]))
        known = meta.get("object_distance_mm")
        corrected = request.form.get("use_known_distance") == "on"
        if corrected and not known:
            raise InputError("Enter the camera-to-object distance at upload to use it as a correction.")
        # A small error in the recovered yaw shifts every disparity by the same
        # amount, which rescales X, Y and Z together. Pose from matches can't pin
        # that down, a measured distance can. The median resists a bad click.
        offset = None
        if known:
            offset = geom["f"] * geom["baseline"] / known - float(np.median([p["d"] for p in all_pts]))
        if corrected:
            for p in all_pts:
                p["d"] += offset
                p["xyz"] = sm.point_3d(p["x"], p["y"], p["x"] - p["xr"] + offset, geom)
        for n, shape in enumerate(shapes, 1):
            for i in sm.disparity_outliers([p["d"] for p in shape["points"]]):
                shape["points"][i]["outlier"] = True
            m = sm.measure_shape(shape["kind"], [p["xyz"] for p in shape["points"]])
            shape.update(m)
            shape["summary"] = ", ".join(f"{i['label'].split(' (')[0]} {i['mm']:.0f} mm" for i in m["items"]
                                         if shape["kind"] != "polygon" or i["label"] == "Perimeter")
            shape["points"] = [{**p, "xyz": np.round(p["xyz"], 1).tolist(), "d": round(p["d"], 2)}
                               for p in shape["points"]]
    except (InputError, sm.StereoError) as e:
        flash(str(e))
        return back

    weak = sum(1 for s in shapes for p in s["points"] if p["method"] != "manual" and p["ncc"] < sm.LOW_NCC)
    result = {
        "shapes": shapes, "mean_z": mean_z, "known_z": known, "corrected": corrected,
        "z_error_pct": None if not known else 100.0 * (mean_z - known) / known,
        "disparity_offset": offset, "weak_points": weak,
        "outliers": [f"shape {n} point {i}" for n, s in enumerate(shapes, 1)
                     for i, p in enumerate(s["points"], 1) if p.get("outlier")],
    }
    left = cv2.imread(os.path.join(run_dir, "rect_left.jpg"))
    cv2.imwrite(os.path.join(run_dir, "measured.jpg"), _draw_shapes(left, shapes), [cv2.IMWRITE_JPEG_QUALITY, 92])
    _write_json(run_dir, "result.json", result)
    return redirect(url_for("hw7.stereo_view", run=run_id) + "#result")


# ----------------------------------------------------------------- room --

def _room_run_view(run_id):
    run_dir = _run_dir(run_id)
    if run_dir is None:
        return None
    _resume_room_job(run_id, run_dir)
    meta = _read_json(run_dir, "request.json") or {}
    status = read_status(run_dir) or {"state": "running", "progress": 0, "label": "Starting"}
    view = {"run_id": run_id, "status": status, "video_url": meta.get("video_url"), "video": meta.get("video"),
            "k_source": meta.get("k_source")}
    if status.get("state") != "done":
        return view
    done = _read_json(run_dir, "meta.json")
    if done is None:
        return view
    view.update(done)
    view["frames"] = [{"index": i, "url": _run_url(run_id, name), "source": done["source_frames"][i],
                       "registered": i in done["registered"]} for i, name in enumerate(done["names"])]
    view["whole_path"] = room.path_complete(done["registered"], done["n_extracted"])
    view["recon_url"] = _run_url(run_id, "reconstruction.png")
    view["zip_url"] = url_for("hw7.download_run", run_id=run_id)
    view["marks"] = _read_json(run_dir, "marks.json") or {}
    view["auto_found"] = bool(done.get("auto", {}).get("corners"))
    if not os.path.exists(os.path.join(run_dir, "result.json")) and view["auto_found"] and meta.get("wall_mm"):
        try:
            _measure_room(run_dir, done, {}, "wall", ROOM_TRAVEL_MM, meta["wall_mm"], None, True)
        except room.RoomError as e:
            view["auto_error"] = str(e)
    view["result"] = _read_json(run_dir, "result.json")
    if view["result"]:
        view["measured_url"] = _run_url(run_id, "reconstruction_mm.png")
    return view


@hw7.route("/room", methods=["GET"])
def room_view():
    run = _room_run_view(session.get("hw7_room_run"))
    offer = None
    if run and run.get("result"):
        offer = samples.offer("hw7", "room", {k: run[k] for k in (
            "status", "frames", "recon_url", "measured_url", "result", "marks", "zip_url", "n_points", "reproj_rms",
            "registered", "dropped", "video", "k_source", "frame_size", "source_frames", "whole_path")})
    defaults = {"travel": ROOM_TRAVEL_MM, "wall": ROOM_WALL_DISTANCE_MM}
    return render_template("hw7/room.html", run=run, sample_offer=offer, corners=room.CORNERS,
                           saved_calib=calibration_summary(get_current_calibration()), defaults=defaults)


def _launch_room_job(run_id, run_dir, req):
    """Runs the reconstruction in a thread, tracked so a run left behind by a
    server restart can be told from one still working."""
    def job():
        try:
            room.process_video(req["video_path"], run_dir, np.array(req["K"], dtype=np.float64),
                               None if req["dist"] is None else np.array(req["dist"], dtype=np.float64),
                               req["n_keyframes"], {"video": req["video"], "k_source": req["k_source"],
                                                    "dist_text": _dist_text(req["dist"])},
                               req.get("refine_focal", True))
        finally:
            _room_jobs.discard(run_id)

    _room_jobs.add(run_id)
    threading.Thread(target=job, daemon=True).start()


def _resume_room_job(run_id, run_dir):
    """A restart (the debug server restarts on any code change) kills the job's
    thread and leaves its status at 'running', so the page would wait forever."""
    status = read_status(run_dir)
    req = _read_json(run_dir, "request.json")
    with _room_jobs_lock:
        if status and status.get("state") == "running" and run_id not in _room_jobs and req:
            _launch_room_job(run_id, run_dir, req)


def _start_room_job(video_path, video_info, K, dist, k_source, n_keyframes, wall_mm, marks=None, refine_focal=True):
    run_id, run_dir = _new_run("room")
    req = {
        "video_path": video_path, "video_url": url_for("hw7.static", filename="room_uploads/"
                                                       + os.path.basename(video_path)),
        "video": video_info, "k_source": k_source, "n_keyframes": n_keyframes, "wall_mm": wall_mm,
        "refine_focal": refine_focal, "K": K.tolist(), "dist": None if dist is None else dist.tolist()}
    _write_json(run_dir, "request.json", req)
    if marks:
        _write_json(run_dir, "marks.json", marks)
    _write_status(run_dir, state="running", progress=0, label="Starting")
    with _room_jobs_lock:
        _launch_room_job(run_id, run_dir, req)
    session["hw7_room_run"] = run_id


@hw7.route("/room/start", methods=["POST"])
def room_start():
    back = redirect(url_for("hw7.room_view") + "#video")
    f = request.files.get("video")
    if not f or not f.filename:
        flash("Choose a video file first.")
        return back
    path = os.path.join(ROOM_UPLOAD_DIR, f"{int(time.time() * 1000)}_{secure_filename(f.filename) or 'video.mp4'}")
    f.save(path)
    try:
        info = room.probe_video(path)
        n_keyframes = int(_number(request.form, "n_keyframes", "Frames to extract", required=True))
        if not 4 <= n_keyframes <= 40:
            raise InputError("Frames to extract must be between 4 and 40.")
        wall = _number(request.form, "wall_mm", "Camera-to-front-wall distance", required=True)
        K, dist, k_source = _intrinsics(request.form, (info["width"], info["height"]))
    except (InputError, room.RoomError) as e:
        flash(str(e))
        return back
    _start_room_job(path, info, K, dist, k_source, n_keyframes, wall,
                    refine_focal=request.form.get("calib_source") != "manual")
    return back


@hw7.route("/room/status/<run_id>")
def room_status(run_id):
    run_dir = _run_dir(run_id)
    if run_dir is None:
        abort(404)
    _resume_room_job(run_id, run_dir)
    return jsonify(read_status(run_dir) or {"state": "running", "progress": 0})


@hw7.route("/room/clear", methods=["POST"])
def room_clear():
    session.pop("hw7_room_run", None)
    return redirect(url_for("hw7.room_view") + "#video")


@hw7.route("/room/rerun/<run_id>", methods=["POST"])
def room_rerun(run_id):
    """Same video and marks with a corrected focal length."""
    run_dir = _run_dir(run_id)
    req = _read_json(run_dir, "request.json") if run_dir else None
    if req is None:
        abort(404)
    try:
        fx = _number(request.form, "fx", "Focal length", required=True)
    except InputError as e:
        flash(str(e))
        return redirect(url_for("hw7.room_view") + "#video")
    K = np.array(req["K"], dtype=np.float64)
    K[1, 1] *= fx / K[0, 0]
    K[0, 0] = fx
    dist = None if req["dist"] is None else np.array(req["dist"], dtype=np.float64)
    source = f"Focal length {fx:.1f} px, corrected from the known wall distance. Previously: {req['k_source']}"
    _start_room_job(req["video_path"], req["video"], K, dist, source, req["n_keyframes"], req.get("wall_mm"),
                    _read_json(run_dir, "marks.json"), refine_focal=False)
    return redirect(url_for("hw7.room_view") + "#video")


def _measure_room(run_dir, meta, marks, scale_mode, travel, wall, ref, back_to_wall):
    """Dimensions from the corners found in the video and any marked by hand
    (a mark wins over a found corner). Writes result.json and the plot."""
    found = meta.get("auto") or {}
    all_marks = {**found.get("corners", {}), **marks}
    if not any(all_marks.values()):
        raise InputError("No room corner was found in this video. Mark at least one, on two or more frames.")
    data = np.load(os.path.join(run_dir, "recon.npz"))
    order = data["registered"].tolist()
    Ps = {f: data["Ps"][i] for i, f in enumerate(order)}
    centres = {f: data["centres"][i] for i, f in enumerate(order)}
    points = room.triangulate_marks(all_marks, Ps, centres)
    whole_path = room.path_complete(order, meta["n_extracted"])
    if scale_mode == "travel" and not whole_path:
        raise InputError("The first or last video frame could not be registered, so the reconstructed path is "
                         "shorter than the real travel. Use the wall distance or a known length for the scale.")
    res = room.measure_room(points, data["centres"], room.viewing_direction(data["Ps"], data["K"]), scale_mode,
                            travel if whole_path else None, wall, ref, back_to_wall)
    room.add_auto_dimensions(res, found, points)
    s = res["scale"]
    res["points"] = [{"label": k, "desc": room.CORNERS.get(k, "known-length reference"),
                      "xyz": np.round(p["X"] * s, 1).tolist(), "views": p["views"], "reproj": round(p["reproj"], 2),
                      "ray_angle": round(p["ray_angle"], 2), "found": k not in marks,
                      "suspect": p["reproj"] > room.BAD_CORNER_PX or p["ray_angle"] < room.MIN_PARALLAX_DEG}
                     for k, p in points.items()]
    res["missing"] = [n for n in ("length", "width", "height") if n not in res["dims"]]
    res.update(scale_mode=scale_mode, travel_mm_in=travel, wall_mm_in=wall, ref_mm_in=ref,
               whole_path=whole_path, back_to_wall=back_to_wall)
    wall_check = next((c for c in res["checks"] if c["name"] == "Camera to front wall"), None)
    if wall_check and res["scale_mode"] == "travel":
        res["suggested_fx"] = round(meta["fx_work"] / meta["work_scale"] * wall_check["known"] / wall_check["measured"], 1)
    room.plot_scene(data["centres"] * s, data["cloud"] * s, data["colours"], {k: p["X"] * s for k, p in points.items()},
                    "mm", os.path.join(run_dir, "reconstruction_mm.png"))
    _write_json(run_dir, "result.json", res)


@hw7.route("/room/measure/<run_id>", methods=["POST"])
def room_measure(run_id):
    run_dir = _run_dir(run_id)
    meta = _read_json(run_dir, "meta.json") if run_dir else None
    if meta is None:
        flash("That reconstruction no longer exists. Upload the video again.")
        return redirect(url_for("hw7.room_view"))
    back = redirect(url_for("hw7.room_view") + "#video")
    try:
        marks = json.loads(request.form.get("marks") or "{}")
        marks = {str(label): {str(f): [float(p[0]), float(p[1])] for f, p in views.items()}
                 for label, views in marks.items()
                 if label in room.CORNERS or label in ("R1", "R2")}
    except (TypeError, ValueError, AttributeError, IndexError):
        flash("The marked corners could not be read. Please mark them again.")
        return back
    _write_json(run_dir, "marks.json", marks)
    try:
        _measure_room(run_dir, meta, marks, request.form.get("scale_mode"),
                      _number(request.form, "travel_mm", "Camera travel"),
                      _number(request.form, "wall_mm", "Camera-to-wall distance"),
                      _number(request.form, "ref_mm", "R1-R2 length"), request.form.get("back_to_wall") == "on")
    except (InputError, room.RoomError) as e:
        flash(str(e))
    return back


SAMPLE_PARTS = [("stereo", "1. Stereo Measurement"), ("room", "3. Room Dimensions")]


@hw7.route("/sample-output")
def sample_output():
    part = samples.chosen_part(SAMPLE_PARTS)
    sample = samples.load("hw7", part)
    info = sample and sample["_sample"]
    tab = samples.tab("HW7", "hw7.sample_output", SAMPLE_PARTS, part, url_for(f"hw7.{part}_view"),
                      "the stereo page" if part == "stereo" else "the room page")
    if part == "stereo":
        return render_template("hw7/stereo.html", run=sample, sample_info=info, sample_tab=tab, last_run=None,
                               saved_calib=None, defaults={})
    return render_template("hw7/room.html", run=sample, sample_info=info, sample_tab=tab, corners=room.CORNERS,
                           saved_calib=None, defaults={})
