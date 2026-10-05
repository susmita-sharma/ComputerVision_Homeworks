# Web glue for HW5 - three sub-pages sharing one blueprint:
#   /hw5/flow      dense optical flow + object/feature tracking on two
#                  uploaded videos (background jobs with a live preview)
#   /hw5/tracking  from-scratch Lucas-Kanade validated against measured
#                  pixel locations, on a frame pair from each video
#   /hw5/sfm       4-viewpoint planar structure-from-motion
#
# Every run writes into its own folder outputs/<run_id>/ (flow_..., track_...,
# sfm_...), so each module's sample output can be downloaded as one zip
# from /hw5/download/<run_id>.zip.

import csv
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
from Homework.hw2_calibration.routes import get_current_calibration
from . import optical_flow as of
from . import reports
from . import sfm_planar as sfm
from . import tracking as trk

hw5 = Blueprint("hw5", __name__, url_prefix="/hw5", template_folder="templates", static_folder="static")

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_ROOT = os.path.join(MODULE_ROOT, "static")
VIDEO_UPLOAD_DIR = os.path.join(STATIC_ROOT, "video_uploads")
SFM_UPLOAD_DIR = os.path.join(STATIC_ROOT, "sfm_uploads")
OUTPUT_DIR = os.path.join(STATIC_ROOT, "outputs")
for d in (VIDEO_UPLOAD_DIR, SFM_UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

RUN_ID_RE = re.compile(r"^(flow|track|sfm)_[A-Za-z0-9_]+$")
SLOTS = ("1", "2")


def _stamp():
    return str(int(time.time() * 1000))


def _new_run(kind):
    run_id = f"{kind}_{_stamp()}"
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir


def _run_dir(run_id):
    if not run_id or not RUN_ID_RE.match(run_id):
        return None
    d = os.path.join(OUTPUT_DIR, run_id)
    return d if os.path.isdir(d) else None


def _run_url(run_id, filename):
    return url_for("hw5.static", filename=f"outputs/{run_id}/{filename}")


def _static_url(abs_path):
    rel = os.path.relpath(abs_path, STATIC_ROOT)
    return url_for("hw5.static", filename=rel.replace(os.sep, "/"))


def _static_abs(url_path):
    """Reverse of url_for('hw5.static', ...): '/hw5/static/...' -> absolute
    path, refusing anything that escapes the static folder."""
    rel = (url_path or "").split("/static/", 1)[-1].split("?", 1)[0]
    path = os.path.realpath(os.path.join(STATIC_ROOT, rel))
    if not path.startswith(os.path.realpath(STATIC_ROOT) + os.sep):
        return None
    return path


def _remember(key, sub, run_id):
    d = dict(session.get(key, {}))
    d[sub] = run_id
    session[key] = d


@hw5.route("/")
def home():
    return render_template("hw5/overview.html")


@hw5.route("/download/<run_id>.zip")
def download_run(run_id):
    """Every file a run produced, zipped - the 'save sample output' button."""
    run_dir = _run_dir(run_id)
    if run_dir is None:
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for fn in sorted(files):
                if fn in ("live.jpg", "status.json") or fn.endswith(".tmp"):
                    continue
                full = os.path.join(root, fn)
                zf.write(full, os.path.join(run_id, os.path.relpath(full, run_dir)))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"hw5_{run_id}.zip")


# ---------------------------------------------------------------- flow --

def _flow_slot_view(slot, run_id):
    run_dir = _run_dir(run_id)
    if run_dir is None:
        return None
    status = of.read_status(run_dir) or {"state": "running", "progress": 0}
    view = {"slot": slot, "run_id": run_id, "status": status, "summary": None}
    summary_path = os.path.join(run_dir, "summary.json")
    if status.get("state") == "done" and os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        view["summary"] = s
        view["status"] = {"state": "done"}
        view["zip_url"] = url_for("hw5.download_run", run_id=run_id)
        view["files"] = {fn: _run_url(run_id, fn) for fn in os.listdir(run_dir)
                         if fn not in ("live.jpg", "status.json") and not fn.endswith(".tmp")}
        if s.get("has_pair"):
            view["pair"] = (_run_url(run_id, "pair_frame_t.png"), _run_url(run_id, "pair_frame_t1.png"))
    return view


@hw5.route("/flow", methods=["GET"])
def flow_view():
    runs = session.get("hw5_flow", {})
    slots = []
    for s in SLOTS:
        key = f"flow_video_{s}"
        run = _flow_slot_view(s, runs.get(s))
        offer = info = None
        if run is None:
            run = samples.load("hw5", key)   # nothing of yours in this slot -> the saved sample
            info = run and run["_sample"]
        elif run.get("summary"):
            offer = samples.offer("hw5", key, {k: run[k] for k in ("slot", "status", "summary", "files", "pair",
                                                                     "zip_url") if k in run})
        slots.append({"slot": s, "run": run, "sample_info": info, "sample_offer": offer})
    return render_template("hw5/optical_flow.html", slots=slots)


@hw5.route("/flow/start/<slot>", methods=["POST"])
def flow_start(slot):
    if slot not in SLOTS:
        abort(404)
    f = request.files.get("video")
    if not f or not f.filename:
        flash("Choose a video file first.")
        return redirect(url_for("hw5.flow_view"))
    try:
        start_sec = max(0.0, float(request.form.get("start_sec", 0) or 0))
        max_seconds = min(60.0, max(2.0, float(request.form.get("max_seconds", 30) or 30)))
    except ValueError:
        flash("Start time and duration must be numbers.")
        return redirect(url_for("hw5.flow_view"))

    run_id, run_dir = _new_run(f"flow_v{slot}")
    in_path = os.path.join(VIDEO_UPLOAD_DIR, f"{run_id}_{secure_filename(f.filename) or 'video.mp4'}")
    f.save(in_path)
    of._write_status(run_dir, state="running", progress=0, frame=0, n_total=0, label=f"Video {slot}")
    threading.Thread(target=of.process_video, args=(in_path, run_dir, f"Video {slot}", start_sec, max_seconds),
                     daemon=True).start()
    _remember("hw5_flow", slot, run_id)
    return redirect(url_for("hw5.flow_view") + f"#video{slot}")


@hw5.route("/flow/status/<run_id>")
def flow_status(run_id):
    run_dir = _run_dir(run_id)
    if run_dir is None:
        abort(404)
    status = of.read_status(run_dir) or {"state": "running", "progress": 0}
    status["live_url"] = _run_url(run_id, "live.jpg") if os.path.exists(os.path.join(run_dir, "live.jpg")) else None
    return jsonify(status)


@hw5.route("/flow/clear/<slot>", methods=["POST"])
def flow_clear(slot):
    d = dict(session.get("hw5_flow", {}))
    d.pop(slot, None)
    session["hw5_flow"] = d
    return redirect(url_for("hw5.flow_view") + f"#video{slot}")


# ------------------------------------------------------------- tracking --

def _draw_tracking(next_bgr, rows):
    out = next_bgr.copy()
    s = max(1, int(round(max(out.shape[:2]) / 800)))
    for k, r in enumerate(rows, 1):
        p0 = (int(round(r["x0"])), int(round(r["y0"])))
        pm = (int(round(r["x_pred"])), int(round(r["y_pred"])))
        pn = (int(round(r["x_match"])), int(round(r["y_match"])))
        cv2.drawMarker(out, p0, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 10 * s, 1 * s)
        cv2.arrowedLine(out, p0, pm, (0, 220, 255), 1 * s, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(out, pn, 7 * s, (0, 200, 0), 1 * s)
        cv2.circle(out, pm, 4 * s, (0, 0, 255), 1 * s)
        cv2.putText(out, str(k), (pm[0] + 8 * s, pm[1] - 8 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 255, 255),
                    1 * s, cv2.LINE_AA)
    return out


def _zoom_tiles(prev_bgr, next_bgr, rows, tile=220):
    """Per point: [frame t crop | frame t+1 crop], magnified so sub-pixel
    disagreements between the trackers are visible."""
    def crop(img, cx, cy, half):
        pad = half + 2
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        x0, y0 = int(round(cx)) - half + pad, int(round(cy)) - half + pad
        return big[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1], int(round(cx)) - half, int(round(cy)) - half

    pairs = []
    for k, r in enumerate(rows[:12], 1):
        half = int(max(16, min(80, abs(r["dx"]) / 2 + abs(r["dy"]) / 2 + 14)))
        mx, my = (r["x0"] + r["x_pred"]) / 2, (r["y0"] + r["y_pred"]) / 2
        tiles = []
        for img, is_next in ((prev_bgr, False), (next_bgr, True)):
            c, ox, oy = crop(img, mx, my, half)
            z = tile / c.shape[0]
            c = cv2.resize(c, (tile, tile), interpolation=cv2.INTER_NEAREST)

            def P(x, y):
                return int((x - ox + 0.5) * z), int((y - oy + 0.5) * z)

            cv2.drawMarker(c, P(r["x0"], r["y0"]), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
            if is_next:
                cv2.circle(c, P(r["x_match"], r["y_match"]), 11, (0, 200, 0), 2)
                cv2.circle(c, P(r["x_actual"], r["y_actual"]), 3, (255, 120, 0), -1)
                cv2.circle(c, P(r["x_pred"], r["y_pred"]), 6, (0, 0, 255), 2)
            cv2.rectangle(c, (0, 0), (tile, 18), (0, 0, 0), -1)
            cv2.putText(c, f"P{k} frame {'t+1' if is_next else 't'}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(c)
        pairs.append(np.hstack([tiles[0], np.full((tile, 4, 3), 255, np.uint8), tiles[1]]))
    cols = 2
    while len(pairs) % cols:
        pairs.append(np.full_like(pairs[0], 255))
    grid_rows = [np.hstack([p if j == 0 else np.hstack([np.full((tile, 12, 3), 255, np.uint8), p])
                            for j, p in enumerate(pairs[i:i + cols])]) for i in range(0, len(pairs), cols)]
    sep = np.full((12, grid_rows[0].shape[1], 3), 255, np.uint8)
    out = grid_rows[0]
    for g in grid_rows[1:]:
        out = np.vstack([out, sep, g])
    return out


def _tracking_runs():
    out = []
    for label, run_id in session.get("hw5_track", {}).items():
        if _run_dir(run_id):
            out.append({"label": label, "run_id": run_id,
                        "url": url_for("hw5.tracking_view", run=run_id),
                        "zip": url_for("hw5.download_run", run_id=run_id)})
    return out


def _tracking_samples():
    out = []
    for key in samples.list_keys("hw5", prefix="track_"):
        s = samples.load("hw5", key)
        if s:
            out.append({"key": key, "label": s.get("label", key),
                        "url": url_for("hw5.tracking_view", sample=key)})
    return out


@hw5.route("/tracking", methods=["GET"])
def tracking_view():
    run_id = request.args.get("run")
    sample_key = request.args.get("sample")
    result = offer = info = None
    sample_list = _tracking_samples()
    if run_id:
        run_dir = _run_dir(run_id)
        if run_dir and os.path.exists(os.path.join(run_dir, "result.json")):
            with open(os.path.join(run_dir, "result.json")) as f:
                result = json.load(f)
            offer = samples.offer("hw5", "track_" + samples.slug(result["label"]), result)
    elif not request.args.get("prev") and not request.args.get("upload") and (sample_key or sample_list):
        # default view for visitors: a saved sample, no upload needed
        result = samples.load("hw5", sample_key or sample_list[0]["key"])
        info = result and result["_sample"]
    prev_url = request.args.get("prev") or (result or {}).get("prev_url")
    next_url = request.args.get("next") or (result or {}).get("next_url")
    label = request.args.get("label") or (result or {}).get("label") or "Uploaded pair"
    return render_template("hw5/tracking.html", prev_url=prev_url, next_url=next_url, label=label,
                           result=result, saved_runs=_tracking_runs(), sample_list=sample_list,
                           sample_info=info, sample_offer=offer)


@hw5.route("/tracking/upload", methods=["POST"])
def tracking_upload():
    prev_f = request.files.get("prev_frame")
    next_f = request.files.get("next_frame")
    if not prev_f or not next_f or not prev_f.filename or not next_f.filename:
        flash("Choose both frame images.")
        return redirect(url_for("hw5.tracking_view"))

    stamp = _stamp()
    prev_path = os.path.join(SFM_UPLOAD_DIR, f"{stamp}_prev_{secure_filename(prev_f.filename)}")
    next_path = os.path.join(SFM_UPLOAD_DIR, f"{stamp}_next_{secure_filename(next_f.filename)}")
    prev_f.save(prev_path)
    next_f.save(next_path)
    label = request.form.get("label") or "Uploaded pair"
    return redirect(url_for("hw5.tracking_view", prev=_static_url(prev_path), next=_static_url(next_path),
                            label=label))


@hw5.route("/tracking/run", methods=["POST"])
def tracking_run():
    prev_url = request.form.get("prev_url")
    next_url = request.form.get("next_url")
    label = request.form.get("label") or "Uploaded pair"
    window = int(request.form.get("window", 21)) // 2 * 2 + 1
    back = redirect(url_for("hw5.tracking_view", prev=prev_url, next=next_url, label=label))

    prev_path, next_path = _static_abs(prev_url), _static_abs(next_url)
    prev_bgr = cv2.imread(prev_path) if prev_path else None
    next_bgr = cv2.imread(next_path) if next_path else None
    if prev_bgr is None or next_bgr is None:
        flash("Could not re-read the frames - please upload them again.")
        return redirect(url_for("hw5.tracking_view"))
    if prev_bgr.shape != next_bgr.shape:
        flash("The two frames must have the same size.")
        return back

    try:
        points = [(float(p[0]), float(p[1])) for p in json.loads(request.form.get("points") or "[]")]
    except (TypeError, ValueError, IndexError):
        points = []
    if request.form.get("auto"):
        n_auto = int(request.form.get("n_auto", 10))
        points += trk.auto_pick_points(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY),
                                       cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY), n=n_auto)
    if not points:
        flash("Click at least one point on frame t (or tick auto-pick) before running.")
        return back

    logs = []
    rows = trk.validate_tracking(prev_bgr, next_bgr, points, window=window, logs=logs)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    # worked example on the converged point that moved the most (most informative)
    conv = [k for k, r in enumerate(rows) if r["converged"]] or list(range(len(rows)))
    wk = max(conv, key=lambda k: np.hypot(rows[k]["dx"], rows[k]["dy"]))
    bil = trk.bilinear_worked_example(next_gray, rows[wk]["x_pred"], rows[wk]["y_pred"])

    run_id, run_dir = _new_run("track")
    cv2.imwrite(os.path.join(run_dir, "frame_t.png"), prev_bgr)
    cv2.imwrite(os.path.join(run_dir, "frame_t1.png"), next_bgr)
    cv2.imwrite(os.path.join(run_dir, "tracking_annotated.jpg"), _draw_tracking(next_bgr, rows),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(os.path.join(run_dir, "tracking_zoom.png"), _zoom_tiles(prev_bgr, next_bgr, rows))

    keys = ["x0", "y0", "x_pred", "y_pred", "dx", "dy", "x_match", "y_match", "ncc", "x_actual", "y_actual",
            "err_vs_match_px", "agreement_err_px", "rms_no_motion", "rms_tracked", "converged", "n_iters"]
    with open(os.path.join(run_dir, "tracking_table.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["point"] + keys)
        for k, r in enumerate(rows, 1):
            wr.writerow([k] + [round(r[c], 4) if isinstance(r[c], float) else r[c] for c in keys])

    summary = {
        "mean_err_match": float(np.mean([r["err_vs_match_px"] for r in rows])),
        "max_err_match": float(np.max([r["err_vs_match_px"] for r in rows])),
        "mean_err_cv": float(np.mean([r["agreement_err_px"] for r in rows])),
        "mean_rms_before": float(np.mean([r["rms_no_motion"] for r in rows])),
        "mean_rms_after": float(np.mean([r["rms_tracked"] for r in rows])),
        "n_converged": sum(1 for r in rows if r["converged"]),
    }
    with open(os.path.join(run_dir, "tracking_report.md"), "w") as f:
        f.write(reports.tracking_report_md(label, rows, logs, window, bil, summary, wk))

    def clean(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    result = clean({
        "run_id": run_id, "label": label, "window": window, "rows": rows, "summary": summary,
        "levels": logs[wk] if logs else [], "bil": bil, "wk": wk,
        "prev_url": prev_url, "next_url": next_url,
        "annotated_url": _run_url(run_id, "tracking_annotated.jpg"),
        "zoom_url": _run_url(run_id, "tracking_zoom.png"),
        "csv_url": _run_url(run_id, "tracking_table.csv"),
        "report_url": _run_url(run_id, "tracking_report.md"),
        "zip_url": url_for("hw5.download_run", run_id=run_id),
        "image_size": [int(prev_bgr.shape[1]), int(prev_bgr.shape[0])],
    })
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f)
    _remember("hw5_track", label, run_id)
    return redirect(url_for("hw5.tracking_view", run=run_id) + "#result")


# ------------------------------------------------------------------ sfm --

@hw5.route("/sfm", methods=["GET"])
def sfm_view():
    run_id = request.args.get("run")
    run_dir = _run_dir(run_id) if run_id else None
    if run_dir and os.path.exists(os.path.join(run_dir, "result.json")):
        with open(os.path.join(run_dir, "result.json")) as f:
            result = json.load(f)
        return render_template("hw5/sfm.html", stage="result", views=None, result=result,
                               sample_offer=samples.offer("hw5", "sfm", result))
    last = session.get("hw5_sfm")
    sample = samples.load("hw5", "sfm")
    return render_template("hw5/sfm.html", stage="upload", views=None, result=sample,
                           sample_info=sample and sample["_sample"],
                           last_run=url_for("hw5.sfm_view", run=last) if last and _run_dir(last) else None)


@hw5.route("/sfm/upload", methods=["POST"])
def sfm_upload():
    files = request.files.getlist("images")
    if len(files) != 4 or any(not f.filename for f in files):
        flash("Upload exactly 4 images, one per viewpoint.")
        return redirect(url_for("hw5.sfm_view"))
    try:
        width_mm = float(request.form.get("width_mm", 210))
        height_mm = float(request.form.get("height_mm", 297))
        n_boundary = int(request.form.get("n_boundary", 4))
    except ValueError:
        flash("Width, height and point count must be numbers.")
        return redirect(url_for("hw5.sfm_view"))
    stamp = _stamp()
    image_urls = []
    sizes = set()
    for i, f in enumerate(files):
        path = os.path.join(SFM_UPLOAD_DIR, f"{stamp}_view{i + 1}_{secure_filename(f.filename)}")
        f.save(path)
        img = cv2.imread(path)
        if img is None:
            flash(f"Viewpoint {i + 1} is not a readable image.")
            return redirect(url_for("hw5.sfm_view"))
        # portrait vs landscape doesn't matter (the phone picks that per shot
        # from its tilt sensor); only a different pixel count means a
        # different camera/lens/resolution setting
        sizes.add(tuple(sorted(img.shape[:2])))
        image_urls.append(_static_url(path))
    if len(sizes) > 1:
        dims = ", ".join(f"{b}x{a}" for a, b in sorted(sizes))
        flash(f"The 4 photos have different resolutions ({dims}), so they can't share one camera model. "
              f"Use photos from the same camera, lens (1x/0.5x/2x) and resolution setting.")
        return redirect(url_for("hw5.sfm_view"))

    views_meta = {
        "image_urls": image_urls, "width_mm": width_mm, "height_mm": height_mm, "n_boundary": n_boundary,
        "use_hw2_calib": request.form.get("use_hw2_calib") == "on", "n_clicks": 4 + n_boundary,
        "notes": request.form.get("notes", ""),
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
    notes = request.form.get("notes", "")

    n_needed = 4 + n_boundary
    if any(len(p) != n_needed for p in points_per_view):
        flash(f"Each image needs exactly {n_needed} clicks (4 corners + {n_boundary} boundary points).")
        return redirect(url_for("hw5.sfm_view"))

    world_corners = sfm.rectangle_world_corners(width_mm, height_mm)
    run_id, run_dir = _new_run("sfm")
    os.makedirs(os.path.join(run_dir, "inputs"), exist_ok=True)

    views, imgs, image_names = [], [], []
    for i, (url, pts) in enumerate(zip(image_urls, points_per_view)):
        path = _static_abs(url)
        img = cv2.imread(path) if path else None
        if img is None:
            flash("Could not re-read the uploaded photos - please upload them again.")
            return redirect(url_for("hw5.sfm_view"))
        name = os.path.basename(path).split("_", 1)[-1]
        shutil.copy(path, os.path.join(run_dir, "inputs", name))
        image_names.append(name)
        imgs.append(img)
        pts = [(float(x), float(y)) for x, y in pts]
        views.append({"corners": pts[:4], "boundary": pts[4:]})
    image_sizes = [(img.shape[1], img.shape[0]) for img in imgs]
    image_size = image_sizes[0]
    mixed = len(set(image_sizes)) > 1

    # One physical camera, but each photo gets its own K: a photo the phone
    # saved in portrait has its x/y axes swapped relative to a landscape one,
    # so fx/fy and the principal point follow that photo's displayed size.
    calib = get_current_calibration() if use_hw2_calib else None
    f35 = next((f for f in (sfm.exif_intrinsics(_static_abs(u), *sz)[1] for u, sz in zip(image_urls, image_sizes))
                if f), None)
    Ks, dists, dist = [], [], None
    if calib is not None:
        dist = np.asarray(calib["dist_coeffs"], dtype=np.float64).reshape(-1)
        any_swapped = False
        for sz in image_sizes:
            Ki, swapped = sfm.calibration_intrinsics(calib["camera_matrix"], calib["image_size"], sz)
            di = dist.copy()
            if swapped:  # radial terms don't care about rotation; tangential ones do - dropped for those views
                di[2:4] = 0.0
                any_swapped = True
            Ks.append(Ki)
            dists.append(di)
        k_source = ("HW2 chessboard calibration, rescaled to this resolution"
                    + (" (x/y axes swapped for photos taken in the other orientation)" if any_swapped else "")
                    + ". Only valid if these photos came from the same camera as the HW2 chessboard photos.")
    else:
        if f35:
            f = f35 / 43.27 * float(np.hypot(*image_size))
            k_source = (f"photo EXIF: 35mm-equivalent focal length {f35:.0f} mm -> f = {f35:.0f}/43.27 x image "
                        f"diagonal = {f:.1f} px; principal point at the image centre")
        else:
            f = max(image_size) / (2.0 * np.tan(np.radians(30.0)))
            k_source = ("approximate: assumed 60 deg field of view across the long side of the image "
                        "(no HW2 calibration used, no EXIF focal length)")
        Ks = [np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]]) for w, h in image_sizes]
        dists = [None] * 4
    if mixed:
        k_source += (". The phone saved these photos in different orientations ("
                     + ", ".join(f"view {i + 1}: {'portrait' if h > w else 'landscape'}"
                                 for i, (w, h) in enumerate(image_sizes))
                     + "), so each view uses the same focal length with its principal point at its own image centre")
    K = Ks[0]
    dist_text = (", ".join(f"{v:.4f}" for v in dist) + " (k1, k2, p1, p2, k3)") if dist is not None else \
        "assumed zero (not calibrated)"

    res = sfm.reconstruct(views, world_corners, Ks, dists)

    sfm.plot_scene(res, world_corners, os.path.join(run_dir, "sfm_scene_3d.png"))
    sfm.plot_topdown(res, world_corners, os.path.join(run_dir, "sfm_boundary_topdown.png"))

    # each photo with its clicks (red corners / blue boundary) and the
    # reconstructed points re-projected through that view's camera (green)
    annotated_urls = []
    for i, (img, v, pose) in enumerate(zip(imgs, views, res["poses"]), 1):
        a = img.copy()
        s = max(1, int(round(max(a.shape[:2]) / 900)))
        poly = np.array([sfm.project(pose["P"], X) for X in res["boundary_estimate"]["polygon_3d"]], np.int32)
        cv2.polylines(a, [poly], True, (0, 200, 0), 2 * s, cv2.LINE_AA)
        for j, (x, y) in enumerate(v["corners"] + v["boundary"]):
            color = (0, 0, 255) if j < 4 else (255, 120, 0)
            cv2.circle(a, (int(x), int(y)), 7 * s, color, 2 * s)
            cv2.putText(a, f"C{j + 1}" if j < 4 else f"B{j - 3}", (int(x) + 9 * s, int(y) - 9 * s),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, color, 2 * s, cv2.LINE_AA)
        for X in list(world_corners) + list(res["boundary_3d"]):
            px, py = sfm.project(pose["P"], X)
            cv2.drawMarker(a, (int(px), int(py)), (0, 220, 0), cv2.MARKER_CROSS, 12 * s, 2 * s)
        cv2.putText(a, f"view {i}: reprojection RMS {pose['reproj_rms']:.2f}px", (12 * s, 32 * s),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 255, 255), 2 * s, cv2.LINE_AA)
        fn = f"view{i}_annotated.jpg"
        cv2.imwrite(os.path.join(run_dir, fn), a, [cv2.IMWRITE_JPEG_QUALITY, 88])
        annotated_urls.append(_run_url(run_id, fn))

    with open(os.path.join(run_dir, "cameras.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["camera", "image", "Cx_mm", "Cy_mm", "Cz_mm", "distance_mm", "tilt_deg", "rvec_x_deg",
                     "rvec_y_deg", "rvec_z_deg", "tx_mm", "ty_mm", "tz_mm", "reproj_rms_px", "fx", "fy", "cx", "cy"])
        for i, (p, nm) in enumerate(zip(res["poses"], image_names), 1):
            wr.writerow([i, nm, *np.round(p["C"], 2), round(p["distance"], 2), round(p["tilt_deg"], 2),
                         *np.round(p["rvec_deg"], 2), *np.round(p["t"], 2), round(p["reproj_rms"], 3),
                         *np.round([p["K"][0, 0], p["K"][1, 1], p["K"][0, 2], p["K"][1, 2]], 2)])
    with open(os.path.join(run_dir, "points.json"), "w") as f:
        json.dump({"width_mm": width_mm, "height_mm": height_mm, "images": image_names, "image_sizes": image_sizes,
                   "clicked_points_px": points_per_view, "K_per_view": [k.tolist() for k in Ks], "notes": notes,
                   "corners_3d_mm": np.round(res["corner_3d"], 3).tolist(),
                   "boundary_3d_mm": np.round(res["boundary_3d"], 3).tolist()}, f, indent=2)

    meta = {"width_mm": width_mm, "height_mm": height_mm, "image_size": image_size, "image_sizes": image_sizes,
            "image_names": image_names,
            "notes": notes, "k_source": k_source, "dist_text": dist_text, "views": views}
    with open(os.path.join(run_dir, "sfm_report.md"), "w") as f:
        f.write(reports.sfm_report_md(res, meta))

    bm = reports.bmatrix
    cameras = []
    for i, (p, v) in enumerate(zip(res["poses"], views), 1):
        hp = p["hpose"]
        cameras.append({
            "id": i, "image": image_names[i - 1], "ok": p["ok"],
            "size": f"{image_sizes[i - 1][0]}x{image_sizes[i - 1][1]}",
            "orientation": "portrait" if image_sizes[i - 1][1] > image_sizes[i - 1][0] else "landscape",
            "tex_Ki": bm(p["K"], "{:.1f}"),
            "corners": [[round(x, 1), round(y, 1)] for x, y in v["corners"]],
            "C": np.round(p["C"], 1).tolist(), "distance": round(p["distance"], 1), "tilt": round(p["tilt_deg"], 1),
            "rvec": np.round(p["rvec_deg"], 1).tolist(), "t": np.round(p["t"], 1).tolist(),
            "reproj_rms": round(p["reproj_rms"], 2), "reproj_max": round(p["reproj_max"], 2),
            "R_diff_deg": round(p["R_diff_deg"], 3), "t_diff": round(p["t_diff"], 2),
            "tex_A": bm(p["A_h"], "{:.0f}"), "tex_H": bm(p["H"], "{:.4g}"), "tex_B": bm(hp["B"], "{:.4g}"),
            "lambda": f"{hp['lambda']:.4g}", "tex_Rraw": bm(hp["R_raw"], "{:.4f}"), "tex_Rh": bm(hp["R"], "{:.4f}"),
            "tex_th": bm(hp["t"].reshape(3, 1), "{:.1f}"), "tex_R": bm(p["R"], "{:.4f}"),
            "tex_t": bm(p["t"].reshape(3, 1), "{:.1f}"), "tex_C": bm(p["C"].reshape(3, 1), "{:.1f}"),
            "tex_P": bm(p["P"], "{:.4g}"),
        })
    wt = res["worked_triangulation"]
    be = res["boundary_estimate"]
    result = {
        "run_id": run_id, "cameras": cameras, "tex_K": bm(K, "{:.1f}"), "k_source": k_source,
        "dist_text": dist_text, "image_size": image_size, "mixed_orientation": mixed,
        "fx": round(float(K[0, 0]), 1), "fy": round(float(K[1, 1]), 1),
        "cx": round(float(K[0, 2]), 1), "cy": round(float(K[1, 2]), 1),
        "hfov": round(float(np.degrees(2 * np.arctan(max(image_size) / (2 * K[0, 0])))), 1),
        "vfov": round(float(np.degrees(2 * np.arctan(min(image_size) / (2 * K[1, 1])))), 1),
        "corner_recovery_error": [round(e, 3) for e in res["corner_recovery_error"]],
        "planarity_rms": round(res["planarity_rms"], 3),
        "corners_3d": np.round(res["corner_3d"], 2).tolist(),
        "boundary_3d": np.round(res["boundary_3d"], 2).tolist(),
        "edges": [round(e, 1) for e in res["edge_lengths"]],
        "area": round(be["area"], 0), "perimeter": round(be["perimeter"], 1),
        "worked": None if wt is None else {
            "obs": [[round(x, 1), round(y, 1)] for x, y in wt["obs"]], "tex_A": bm(wt["A"], "{:.4g}"),
            "S": [float(f"{s:.4g}") for s in wt["S"]], "tex_Xh": bm(wt["X_h"].reshape(4, 1), "{:.5g}"),
            "X": np.round(wt["X"], 2).tolist()},
        "scene_url": _run_url(run_id, "sfm_scene_3d.png"),
        "topdown_url": _run_url(run_id, "sfm_boundary_topdown.png"),
        "annotated_urls": annotated_urls, "image_urls": image_urls, "image_names": image_names,
        "report_url": _run_url(run_id, "sfm_report.md"), "cameras_csv_url": _run_url(run_id, "cameras.csv"),
        "zip_url": url_for("hw5.download_run", run_id=run_id),
        "width_mm": width_mm, "height_mm": height_mm, "notes": notes,
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f)
    session["hw5_sfm"] = run_id
    return redirect(url_for("hw5.sfm_view", run=run_id))
