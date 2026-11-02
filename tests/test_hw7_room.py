import json
import os
import re
import time

import cv2
import numpy as np
import pytest

from Homework.hw_7 import room_auto as auto
from Homework.hw_7 import room_video as room, routes
from tests.hw7_scene import Plane, project, render, yaw

F = 620.0
SIZE = (1280, 720)
K = np.array([[F, 0, 640.0], [0, F, 360.0], [0, 0, 1.0]])
WIDTH, LENGTH, HEIGHT = 3600.0, 4200.0, 2600.0
TRAVEL = 1260.0
WALL_DISTANCE = 2673.0
CAM_Z = LENGTH - WALL_DISTANCE
N_FRAMES = 48
FORWARD = np.array([0.0, 0.0, 1.0])   # the first camera looks along +Z at the front wall


def room_planes():
    W, L, H = WIDTH, LENGTH, HEIGHT
    return [
        Plane([0, 0, L], [1, 0, 0], [0, 1, 0], W, H, 11),       # front wall
        Plane([0, 0, 0], [0, 0, 1], [0, 1, 0], L, H, 12),       # left wall
        Plane([W, 0, 0], [0, 0, 1], [0, 1, 0], L, H, 13),       # right wall
        Plane([0, H, 0], [1, 0, 0], [0, 0, 1], W, L, 14),       # floor
        Plane([0, 0, 0], [1, 0, 0], [0, 0, 1], W, L, 15),       # ceiling
    ]


def camera(i):
    u = i / (N_FRAMES - 1)
    return yaw(2.0 * np.sin(u * 6)), np.array([WIDTH / 2 - TRAVEL / 2 + TRAVEL * u, HEIGHT / 2, CAM_Z])


CORNERS_3D = {"A": (0, HEIGHT, LENGTH), "B": (WIDTH, HEIGHT, LENGTH), "E": (0, 0, LENGTH), "F": (WIDTH, 0, LENGTH)}


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("room") / "room.mp4")
    planes = room_planes()
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24, SIZE)
    assert w.isOpened()
    for i in range(N_FRAMES):
        w.write(render(planes, K, *camera(i), SIZE))
    w.release()
    return path


def true_marks(frames, source_frames):
    marks = {}
    for label, X in CORNERS_3D.items():
        for f in frames:
            R, C = camera(source_frames[f])
            marks.setdefault(label, {})[str(f)] = project(K, R, C, X).tolist()
    return marks


@pytest.fixture(scope="module")
def recon(video):
    frames, K_work, indices = room.extract_keyframes(video, 10, K, None)
    res = room.reconstruct(frames, K_work)
    order = sorted(res["poses"])
    from Homework.hw5_motion_sfm import sfm_planar as sfm
    Ps = {f: sfm.projection_matrix(K_work, *res["poses"][f]) for f in order}
    centres = {f: sfm.camera_center(*res["poses"][f]) for f in order}
    return res, Ps, centres, indices, order


def test_reconstruction_registers_frames_with_low_error(recon):
    res, _, _, _, order = recon
    assert len(order) >= 8
    assert res["reproj_rms"] < 1.5


def test_room_width_and_height_from_camera_travel(recon):
    res, Ps, centres, indices, order = recon
    picks = [order[0], order[len(order) // 2], order[-1]]
    points = room.triangulate_marks(true_marks(picks, indices), Ps, centres)
    assert (indices[0], indices[-1]) == (0, N_FRAMES - 1)
    out = room.measure_room(points, np.array(list(centres.values())), FORWARD, "travel", TRAVEL, WALL_DISTANCE, None)
    assert out["dims"]["width"]["mm"] == pytest.approx(WIDTH, rel=0.04)
    assert out["dims"]["height"]["mm"] == pytest.approx(HEIGHT, rel=0.04)
    assert "length" not in out["dims"]
    check = next(c for c in out["checks"] if c["name"] == "Camera to front wall")
    assert abs(check["error_pct"]) < 5


def test_wall_distance_can_set_the_scale(recon):
    res, Ps, centres, indices, order = recon
    points = room.triangulate_marks(true_marks([order[0], order[-1]], indices), Ps, centres)
    out = room.measure_room(points, np.array(list(centres.values())), FORWARD, "wall", TRAVEL, WALL_DISTANCE, None,
                            back_to_wall=True)
    assert out["dims"]["width"]["mm"] == pytest.approx(WIDTH, rel=0.04)
    assert out["dims"]["length"]["mm"] == pytest.approx(WALL_DISTANCE)
    assert "opposite wall" in out["dims"]["length"]["method"]


def test_scale_errors_are_clear(recon):
    res, Ps, centres, indices, order = recon
    points = room.triangulate_marks(true_marks([order[0], order[-1]], indices), Ps, centres)
    path = np.array(list(centres.values()))
    with pytest.raises(room.RoomError, match="R1 and R2"):
        room.measure_room(points, path, FORWARD, "reference", None, None, 900)
    with pytest.raises(room.RoomError, match="travel"):
        room.measure_room(points, path, FORWARD, "travel", None, None, None)
    with pytest.raises(room.RoomError, match="1 front-wall"):
        room.measure_room({}, path, FORWARD, "wall", None, 2673, None)


def test_one_end_of_the_wall_is_enough_for_scale_and_height(recon):
    res, Ps, centres, indices, order = recon
    points = room.triangulate_marks(true_marks([order[0], order[-1]], indices), Ps, centres)
    one_end = {k: points[k] for k in "AE"}
    out = room.measure_room(one_end, np.array(list(centres.values())), FORWARD, "wall", None, WALL_DISTANCE, None,
                            back_to_wall=True)
    assert out["dims"]["height"]["mm"] == pytest.approx(HEIGHT, rel=0.06)
    assert out["dims"]["length"]["mm"] == pytest.approx(WALL_DISTANCE)
    assert "width" not in out["dims"]


def test_viewing_direction_is_the_camera_z_axis():
    P = K @ np.hstack([yaw(0.0), np.zeros((3, 1))])
    assert room.viewing_direction([P, P], K) == pytest.approx(FORWARD)


def test_path_with_unregistered_ends_is_not_the_full_travel():
    assert room.path_complete([0, 1, 2, 3], 4)
    assert not room.path_complete([3, 4, 5, 11], 12)
    assert not room.path_complete([0, 1, 2], 4)


def test_single_view_mark_is_rejected(recon):
    _, Ps, centres, _, order = recon
    with pytest.raises(room.RoomError, match="at least 2 frames"):
        room.triangulate_marks({"A": {str(order[0]): [10.0, 10.0]}}, Ps, centres)


def test_rotation_only_video_is_rejected(tmp_path):
    planes = room_planes()
    frames = [render(planes, K, yaw(a), np.array([WIDTH / 2, HEIGHT / 2, CAM_Z]), SIZE) for a in np.linspace(-6, 6, 6)]
    with pytest.raises(room.RoomError, match="parallax"):
        room.reconstruct(frames, K)


def test_web_flow_from_upload_to_room_size(video, tmp_path, monkeypatch):
    import main
    for name in ("OUTPUT_DIR", "ROOM_UPLOAD_DIR"):
        monkeypatch.setattr(routes, name, str(tmp_path))
    main.app.config["TESTING"] = True
    client = main.app.test_client()

    with open(video, "rb") as f:
        r = client.post("/hw7/room/start", data={"video": (f, "room.mp4"), "n_keyframes": "10",
                                                   "wall_mm": str(WALL_DISTANCE), "calib_source": "manual",
                                                   "fx": str(F)},
                        content_type="multipart/form-data")
    assert r.status_code == 302
    with client.session_transaction() as s:
        run_id = s["hw7_room_run"]
    for _ in range(240):
        st = client.get(f"/hw7/room/status/{run_id}").get_json()
        if st["state"] != "running":
            break
        time.sleep(0.5)
    assert st["state"] == "done", st

    meta = json.load(open(os.path.join(str(tmp_path), run_id, "meta.json")))
    marks = true_marks([meta["registered"][0], meta["registered"][-1]], meta["source_frames"])
    page = client.get("/hw7/room").data.decode()
    assert "Extracted frames" in page and "Optional: mark corners by hand" in page

    r = client.post(f"/hw7/room/measure/{run_id}", data={
        "marks": json.dumps(marks), "scale_mode": "wall", "travel_mm": "1260", "wall_mm": str(WALL_DISTANCE)},
        follow_redirects=True)
    page = r.data.decode()
    assert "Estimated room dimensions" in page
    assert "Length could not be determined" in page or "camera-to-front-wall distance, because" in page
    assert "Video 2" not in page and "/hw7/room/start/" not in page
    result = json.load(open(os.path.join(str(tmp_path), run_id, "result.json")))
    assert result["dims"]["width"]["mm"] == pytest.approx(WIDTH, rel=0.04)
    assert result["dims"]["height"]["mm"] == pytest.approx(HEIGHT, rel=0.04)

    # travel scale checks the wall distance, suggests a focal length, and a re-run keeps the marks
    client.post(f"/hw7/room/measure/{run_id}", data={
        "marks": json.dumps(marks), "scale_mode": "travel", "travel_mm": str(TRAVEL), "wall_mm": str(WALL_DISTANCE)})
    result = json.load(open(os.path.join(str(tmp_path), run_id, "result.json")))
    assert result["suggested_fx"] == pytest.approx(F, rel=0.05)
    time.sleep(0.01)
    client.post(f"/hw7/room/rerun/{run_id}", data={"fx": str(result["suggested_fx"])})
    with client.session_transaction() as s:
        new_id = s["hw7_room_run"]
    assert new_id != run_id
    for _ in range(240):
        st = client.get(f"/hw7/room/status/{new_id}").get_json()
        if st["state"] != "running":
            break
        time.sleep(0.5)
    assert st["state"] == "done", st
    assert json.load(open(os.path.join(str(tmp_path), new_id, "marks.json"))) == marks


def test_floor_and_ceiling_lines_are_found_with_the_foot_of_the_baseboard():
    img = np.full((1280, 720, 3), 170, np.uint8)
    img[:100] = 215            # ceiling
    img[1100:1140] = 240       # baseboard
    img[1140:] = 60            # carpet
    view = auto.read_frame(img)
    assert view["ceiling"]["b"] == pytest.approx(100, abs=4)
    assert view["floor"]["b"] == pytest.approx(1140, abs=6)


def test_corner_is_the_point_the_line_ends_agree_on():
    K_c = np.array([[900.0, 0, 360], [0, 900.0, 640], [0, 0, 1]])
    X = np.array([-1.0, 1.0, 5.0])
    frames = list(range(6))
    centres = {f: np.array([f * 0.5, 0.0, 0.0]) for f in frames}
    Ps = {f: K_c @ np.c_[np.eye(3), -centres[f]] for f in frames}
    seen = {f: (Ps[f] @ np.append(X, 1))[:2] / (Ps[f] @ np.append(X, 1))[2] for f in frames}
    seen[3] = seen[3] + 80      # a stray line end
    found, agree = auto.consensus_point(seen, Ps, centres)
    assert found == pytest.approx(X, abs=0.02)
    assert 3 not in agree


def test_a_line_end_with_no_second_view_is_no_corner():
    frames = list(range(4))
    centres = {f: np.array([f * 0.5, 0.0, 0.0]) for f in frames}
    K_c = np.array([[900.0, 0, 360], [0, 900.0, 640], [0, 0, 1]])
    Ps = {f: K_c @ np.c_[np.eye(3), -centres[f]] for f in frames}
    seen = {f: np.array([100.0 + 300 * f, 600.0 + 150 * (f % 2)]) for f in frames}
    assert auto.consensus_point(seen, Ps, centres) is None


def test_focal_length_comes_back_from_the_walls_being_square():
    f, w, h = 1100.0, 720, 1280
    K_true = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    v1 = np.array([w / 2 + 2400.0, h / 2])
    v2 = np.array([w / 2 - f * f / 2400.0, h / 2])

    def line(pt):
        a = (pt[1] - v1[1]) / (pt[0] - v1[0])
        return {"a": a, "b": v1[1] - a * v1[0], "x0": 0.0, "x1": float(w), "span": float(w)}

    foot = np.array([300.0, 1050.0])
    start = foot + 0.3 * (v2 - foot)
    view = {"size": (w, h), "floor": line(foot), "ceiling": line([300.0, 230.0]), "segments": np.array([[*start, *foot]])}
    assert auto._focal_from_frame(view, K_true, np.array([0, -1.0, 0])) == pytest.approx(f, rel=0.01)


def test_window_head_does_not_pass_for_the_ceiling():
    Kinv = np.linalg.inv(K)
    def view(top):
        return {"size": SIZE, "ceiling": {"a": 0.0, "b": float(top), "x0": 0, "x1": SIZE[0], "span": SIZE[0]},
                "floor": {"a": 0.0, "b": SIZE[1] - 40.0, "x0": 0, "x1": SIZE[0], "span": SIZE[0]}}
    views = [view(40)] * 5 + [view(160)] * 3
    out = auto.height_over_distance(views, [np.eye(3)] * 8, np.array([0, -1.0, 0]), Kinv)
    assert out["frames"] == 5
    assert out["ratio"] == pytest.approx((SIZE[1] - 80.0) / K[1, 1], rel=0.01)


def test_width_is_a_lower_limit_when_the_far_end_is_never_seen():
    res = {"dims": {}, "wall_mm": 2000.0, "scale": 100.0}
    room.add_auto_dimensions(res, {"height": {"ratio": 1.2, "frames": 5, "spread": 0.01}, "reach": 25.0}, {"A": {}})
    assert res["dims"]["height"]["mm"] == pytest.approx(2400.0)
    assert res["dims"]["width"]["mm"] == pytest.approx(2500.0)
    assert res["dims"]["width"]["lower_limit"] is True


def test_run_left_running_by_a_restart_is_picked_up_again(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "OUTPUT_DIR", str(tmp_path))
    run_id, run_dir = routes._new_run("room")
    routes._write_json(run_dir, "request.json", {
        "video_path": "x.mp4", "video": {}, "k_source": "", "n_keyframes": 4, "K": K.tolist(), "dist": None})
    routes._write_status(run_dir, state="running", progress=19, label="Detecting SIFT features")
    resumed = []
    monkeypatch.setattr(routes.room, "process_video", lambda *a: resumed.append(a[0]))
    routes._resume_room_job(run_id, run_dir)
    for _ in range(100):
        if resumed:
            break
        time.sleep(0.01)
    assert resumed == ["x.mp4"]
