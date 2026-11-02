# Offline samples - shared by every homework.
#
# After a run, a page can offer "Save as offline sample". Saving copies every
# file the result refers to (any /<hw>/static/... URL inside the result data)
# into Homework/<hw>/static/samples/<key>/ together with a sample.json holding
# the result data, so the page can show that run to anyone later without an
# upload. The samples folder is NOT gitignored - commit it and the hosted site
# shows the same samples.
#
# Saving/deleting is only allowed when app.config["ALLOW_SAMPLE_SAVE"] is on
# (python main.py turns it on; gunicorn deployments leave it off unless the
# ALLOW_SAMPLE_SAVE=1 env var is set), so visitors to the hosted site can't
# overwrite your samples.

import io
import json
import os
import re
import shutil
import time
import uuid
import zipfile

from flask import (Blueprint, abort, current_app, flash, redirect, request, send_file, url_for)

HOMEWORK_ROOT = os.path.dirname(os.path.abspath(__file__))
PENDING_DIR = os.path.join(HOMEWORK_ROOT, "_pending_samples")
KEY_RE = re.compile(r"^[a-z0-9_]{1,60}$")
FILE_TAG = "sample://"
ZIP_TAG = "sample-zip://"
SKIP_FILES = {"live.jpg", "status.json"}

# Module pages that get the Run / Sample Output tabs:
# page endpoint -> (its GET page, that page's Sample Output sub-tab, first tab's label).
# POST-only endpoints that render a page point back at the GET page.
MODULE_PAGES = {
    "hw2.calibrate_view": ("hw2.calibrate_view", "calibration", "Run"),
    "hw2.measure_view": ("hw2.measure_view", "measurement", "Run"),
    "hw2.validate_view": ("hw2.validate_view", "validation", "Run"),
    "hw3.blur_view": ("hw3.blur_view", None, "Run"),
    "hw4.home": ("hw4.home", None, "Run"),
    "hw4.process_rgb": ("hw4.home", None, "Run"),
    "hw4.process_thermal": ("hw4.home", None, "Run"),
    "hw5.home": ("hw5.home", None, "Overview"),
    "hw5.flow_view": ("hw5.flow_view", "flow", "Run"),
    "hw5.tracking_view": ("hw5.tracking_view", "tracking", "Run"),
    "hw5.sfm_view": ("hw5.sfm_view", "sfm", "Run"),
    "hw5.sfm_upload": ("hw5.sfm_view", "sfm", "Run"),
    "hw7.home": ("hw7.home", None, "Overview"),
    "hw7.stereo_view": ("hw7.stereo_view", "stereo", "Run"),
    "hw7.room_view": ("hw7.room_view", "room", "Run"),
}

samples_bp = Blueprint("samples", __name__, url_prefix="/samples")


def can_save():
    return bool(current_app.config.get("ALLOW_SAMPLE_SAVE"))


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "sample"


def _static_folder(bp):
    blueprint = current_app.blueprints.get(bp)
    if blueprint is None or not blueprint.static_folder:
        abort(404)
    return os.path.realpath(blueprint.static_folder)


def _sample_dir(bp, key):
    if not KEY_RE.match(key or ""):
        abort(404)
    return os.path.join(_static_folder(bp), "samples", key)


def offer(bp, key, data):
    """Stash a just-computed result so the page can show a save button.
    Returns what the template needs, or None when saving is disabled."""
    if not can_save():
        return None
    os.makedirs(PENDING_DIR, exist_ok=True)
    token = uuid.uuid4().hex
    with open(os.path.join(PENDING_DIR, token + ".json"), "w") as f:
        json.dump({"bp": bp, "key": key, "data": data}, f)
    return {"token": token, "key": key,
            "save_url": url_for("samples.save", bp=bp, key=key),
            "exists": os.path.exists(os.path.join(_sample_dir(bp, key), "sample.json"))}


def _copy_files(obj, prefix, static_root, dest, copied):
    """Walk the result data; copy each referenced static file into dest and
    swap its URL for a sample:// placeholder."""
    if isinstance(obj, dict):
        return {k: (ZIP_TAG if k == "zip_url" else _copy_files(v, prefix, static_root, dest, copied))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_copy_files(v, prefix, static_root, dest, copied) for v in obj]
    if isinstance(obj, str) and obj.startswith(prefix):
        rel = obj[len(prefix):].split("?", 1)[0].split("#", 1)[0]
        src = os.path.realpath(os.path.join(static_root, rel))
        if not src.startswith(static_root + os.sep) or not os.path.isfile(src):
            return obj
        if os.path.basename(src) in SKIP_FILES:
            return None
        if src not in copied:
            name = os.path.basename(src)
            base, ext = os.path.splitext(name)
            n = 1
            while name in copied.values():
                n += 1
                name = f"{base}_{n}{ext}"
            shutil.copy2(src, os.path.join(dest, name))
            copied[src] = name
        return FILE_TAG + copied[src]
    return obj


def _resolve(obj, bp, key):
    if isinstance(obj, dict):
        return {k: _resolve(v, bp, key) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, bp, key) for v in obj]
    if isinstance(obj, str):
        if obj.startswith(FILE_TAG):
            return url_for(f"{bp}.static", filename=f"samples/{key}/{obj[len(FILE_TAG):]}")
        if obj == ZIP_TAG:
            return url_for("samples.download", bp=bp, key=key)
    return obj


def load(bp, key):
    """The saved sample's data with real URLs, plus a `_sample` info dict;
    None if there is no sample under that key."""
    path = os.path.join(_sample_dir(bp, key), "sample.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    data = _resolve(saved["data"], bp, key)
    data["_sample"] = {
        "key": key, "saved_at": saved.get("saved_at", ""),
        "zip_url": url_for("samples.download", bp=bp, key=key),
        "delete_url": url_for("samples.delete", bp=bp, key=key) if can_save() else None,
    }
    return data


def list_keys(bp, prefix=""):
    root = os.path.join(_static_folder(bp), "samples")
    if not os.path.isdir(root):
        return []
    return sorted(k for k in os.listdir(root)
                  if k.startswith(prefix) and os.path.exists(os.path.join(root, k, "sample.json")))


def tab(title, endpoint, parts, active, run_url, run_label):
    """What a Sample Output page's header needs. parts is [(key, label)]."""
    return {"title": title, "run_url": run_url, "run_label": run_label,
            "parts": [{"label": label, "url": url_for(endpoint, part=key), "active": key == active}
                      for key, label in parts]}


def chosen_part(parts):
    """The ?part= sub-tab asked for, or the first one."""
    asked = request.args.get("part")
    return asked if asked in dict(parts) else parts[0][0]


def module_tabs(sample_tab=None):
    """The Run / Sample Output tabs for the page being shown, or None off a
    module page. sample_tab is set when a Sample Output page is rendering."""
    if sample_tab:
        return {"run_url": sample_tab["run_url"], "run_label": "Run",
                "sample_url": request.full_path, "on_sample": True}
    page = MODULE_PAGES.get(request.endpoint)
    if page is None:
        return None
    run_endpoint, part, label = page
    return {"run_url": url_for(run_endpoint), "run_label": label, "on_sample": False,
            "sample_url": url_for(run_endpoint.split(".")[0] + ".sample_output", part=part)}


@samples_bp.app_context_processor
def inject_module_tabs():
    return {"module_tabs": module_tabs}


def _back():
    nxt = request.form.get("next") or "/"
    return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else "/")


@samples_bp.route("/save/<bp>/<key>", methods=["POST"])
def save(bp, key):
    if not can_save():
        abort(403)
    token = request.form.get("token", "")
    pending = os.path.join(PENDING_DIR, token + ".json")
    if not re.match(r"^[0-9a-f]{32}$", token) or not os.path.exists(pending):
        flash("That result has expired - run it again, then save.")
        return _back()
    with open(pending) as f:
        stash = json.load(f)
    if stash["bp"] != bp or stash["key"] != key:
        abort(400)

    dest = _sample_dir(bp, key)
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    static_root = _static_folder(bp)
    prefix = url_for(f"{bp}.static", filename="")
    data = _copy_files(stash["data"], prefix, static_root, tmp, {})
    with open(os.path.join(tmp, "sample.json"), "w") as f:
        json.dump({"key": key, "saved_at": time.strftime("%Y-%m-%d %H:%M"), "data": data}, f, indent=1)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    os.remove(pending)
    flash(f"Saved as the offline sample \"{key}\" in {os.path.relpath(dest, os.path.dirname(HOMEWORK_ROOT))} - "
          f"commit that folder so the hosted site shows it too.")
    return _back()


@samples_bp.route("/delete/<bp>/<key>", methods=["POST"])
def delete(bp, key):
    if not can_save():
        abort(403)
    shutil.rmtree(_sample_dir(bp, key), ignore_errors=True)
    flash(f"Deleted the offline sample \"{key}\".")
    return _back()


@samples_bp.route("/<bp>/<key>.zip")
def download(bp, key):
    d = _sample_dir(bp, key)
    if not os.path.isdir(d):
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(d)):
            zf.write(os.path.join(d, fn), os.path.join(f"{bp}_{key}", fn))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{bp}_sample_{key}.zip")
