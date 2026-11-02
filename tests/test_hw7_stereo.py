import io
import json
import os
import re

import cv2
import numpy as np
import pytest

from Homework.hw_7 import routes, stereo_measure as sm
from tests.hw7_scene import Plane, project, render, yaw

F = 1500.0
SIZE = (1600, 1200)
K = np.array([[F, 0, 800.0], [0, F, 600.0], [0, 0, 1.0]])
BASELINE = 297.0
DISTANCE = 1188.0
RECT = (-260.0, -150.0, 300.0, 200.0)   # x, y, width, height at DISTANCE
RIGHT_R = yaw(-3.0)
RIGHT_C = np.array([BASELINE, 0.0, 0.0])


def scene():
    x, y, w, h = RECT
    cx, cy, r = 260.0, -60.0, 110.0
    return [
        Plane([-3000, -3000, 2600], [1, 0, 0], [0, 1, 0], 6000, 6000, 1),
        Plane([x, y, DISTANCE], [1, 0, 0], [0, 1, 0], w, h, 2),
        Plane([cx - r, cy - r, DISTANCE], [1, 0, 0], [0, 1, 0], 2 * r, 2 * r, 3,
              mask=lambda a, b: np.hypot(a - r, b - r) <= r),
    ]


@pytest.fixture(scope="module")
def pair():
    planes = scene()
    return render(planes, K, np.eye(3), np.zeros(3), SIZE), render(planes, K, RIGHT_R, RIGHT_C, SIZE)


def rectified_px(res, X):
    """True position of a world point on the rectified left image."""
    g = res["geom"]
    h = res["R1"] @ np.asarray(X, float)
    return np.array([g["f"] * h[0] / h[2] + g["cx"], g["f"] * h[1] / h[2] + g["cy"]])


def test_adapt_calibration_treats_video_as_centre_crop():
    Kc = np.array([[3000.0, 0, 2016], [0, 3000.0, 1512], [0, 0, 1]])
    Kv, swapped = sm.adapt_calibration(Kc, (4032, 3024), (1920, 1080))
    assert not swapped
    assert Kv[0, 0] == pytest.approx(Kv[1, 1])
    assert Kv[0, 0] == pytest.approx(3000 * 1920 / 4032)
    assert (Kv[0, 2], Kv[1, 2]) == pytest.approx((960, 540))


def test_circle_fit_recovers_tilted_diameter():
    t = np.linspace(0, 2 * np.pi, 9)[:-1]
    u, v = np.array([1.0, 0, 0.5]) / np.sqrt(1.25), np.array([0, 1.0, 0])
    pts = np.array([100, 40, 900.0]) + 75 * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    _, diameter, residual = sm.fit_circle_3d(pts)
    assert diameter == pytest.approx(150.0)
    assert residual < 1e-6


def test_polygon_reports_every_edge():
    m = sm.measure_shape("polygon", [[0, 0, 0], [30, 0, 0], [30, 40, 0]])
    assert m["edges"] == pytest.approx([30, 40, 50])
    assert m["items"][-1]["mm"] == pytest.approx(120)


def test_measures_rectangle_and_tilted_cameras(pair):
    res = sm.run_stereo(pair[0], pair[1], K, None, BASELINE, False)
    g = res["geom"]
    assert g["baseline"] == pytest.approx(BASELINE, rel=1e-6)
    assert res["row_err"] < 1.5

    x, y, w, h = RECT
    corners = [[x, y, DISTANCE], [x + w, y, DISTANCE], [x + w, y + h, DISTANCE], [x, y + h, DISTANCE]]
    pts = []
    for X in corners:
        px = rectified_px(res, X)
        d, score, _ = sm.locate_point(res["rect_l"], res["rect_r"], res["raw"], px[0], px[1], res["lo"], res["hi"])
        pts.append(sm.point_3d(px[0], px[1], d, g))
    assert np.mean([p[2] for p in pts]) == pytest.approx(DISTANCE, rel=0.02)
    m = sm.measure_shape("rectangle", pts)
    assert m["items"][0]["mm"] == pytest.approx(w, rel=0.02)
    assert m["items"][1]["mm"] == pytest.approx(h, rel=0.02)


def test_measures_circle_diameter(pair):
    res = sm.run_stereo(pair[0], pair[1], K, None, BASELINE, False)
    pts = []
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        X = [260 + 110 * np.cos(a), -60 + 110 * np.sin(a), DISTANCE]
        px = rectified_px(res, X)
        d, _, _ = sm.locate_point(res["rect_l"], res["rect_r"], res["raw"], px[0], px[1], res["lo"], res["hi"])
        pts.append(sm.point_3d(px[0], px[1], d, res["geom"]))
    assert sm.measure_shape("circle", pts)["items"][0]["mm"] == pytest.approx(220, rel=0.03)


def test_swapped_images_are_rejected(pair):
    with pytest.raises(sm.StereoError, match="swapped"):
        sm.run_stereo(pair[1], pair[0], K, None, BASELINE, False)


def test_repeating_pattern_cannot_outvote_the_sideways_matches():
    # like blinds: most raw matches land one slat up or down from the true one
    rng = np.random.default_rng(1)
    pts1 = rng.uniform(100, 900, (300, 2))
    pts2 = pts1 + [-140.0, 0.0]
    pts2[:200, 1] += rng.choice([-60.0, 60.0], 200) * 2
    p1, p2 = sm.sideways_matches(pts1, pts2)
    assert len(p1) == 100
    assert np.allclose(p1[:, 1], p2[:, 1])


def test_corner_matched_onto_the_background_is_flagged():
    assert sm.disparity_outliers([260.0, 256.4, 253.5, 77.8]) == [3]
    assert sm.disparity_outliers([260.0, 256.4, 253.5, 251.0]) == []


def test_unrelated_images_are_rejected(pair):
    noise = np.random.default_rng(0).integers(0, 255, pair[0].shape, np.uint8)
    with pytest.raises(sm.StereoError):
        sm.run_stereo(pair[0], noise, K, None, BASELINE, False)


def test_different_sizes_are_rejected(pair):
    with pytest.raises(sm.StereoError, match="same size"):
        sm.run_stereo(pair[0], pair[1][:600], K, None, BASELINE, False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    import main
    for name in ("OUTPUT_DIR", "STEREO_UPLOAD_DIR"):
        monkeypatch.setattr(routes, name, str(tmp_path))
    monkeypatch.setattr(routes, "get_current_calibration", lambda: None)
    main.app.config["TESTING"] = True
    return main.app.test_client()


def jpg(img):
    return io.BytesIO(cv2.imencode(".png", img)[1].tobytes())


def form(pair, **over):
    data = {"left": (jpg(pair[0]), "l.png"), "right": (jpg(pair[1]), "r.png"), "baseline_mm": "297",
            "object_distance_mm": "1188", "calib_source": "manual", "fx": str(F), "cx": "800", "cy": "600"}
    data.update(over)
    return data


def test_web_flow_from_upload_to_measurement(client, pair):
    r = client.post("/hw7/stereo/upload", data=form(pair), content_type="multipart/form-data")
    assert r.status_code == 302 and "run=" in r.location
    run_id = re.search(r"run=(stereo_\d+)", r.location).group(1)
    page = client.get(f"/hw7/stereo?run={run_id}")
    assert b"disparity map" in page.data and b"rectified left" in page.data

    m = client.post(f"/hw7/stereo/match/{run_id}", json={"x": 500, "y": 500}).get_json()
    assert m["xr"] is not None and m["ncc"] > 0.6

    shape = [{"kind": "polygon", "points": [{"x": 500 + a, "y": 500 + b, "xr": 500 + a - m["d"], "ncc": m["ncc"],
                                             "method": "ncc"} for a, b in ((0, 0), (100, 0), (100, 80))]}]
    r = client.post(f"/hw7/stereo/measure/{run_id}", data={"shapes": json.dumps(shape)})
    assert r.status_code == 302
    page = client.get(f"/hw7/stereo?run={run_id}").data.decode()
    assert "Edge 1-2" in page and "Perimeter" in page and "Stereo vs stated" in page

    # a pose error adds a constant to every disparity; the stated distance removes it
    for p in shape[0]["points"]:
        p["xr"] -= 40.0
    client.post(f"/hw7/stereo/measure/{run_id}", data={"shapes": json.dumps(shape), "use_known_distance": "on"})
    with open(os.path.join(str(routes.OUTPUT_DIR), run_id, "result.json")) as f:
        result = json.load(f)
    assert result["corrected"] and result["z_error_pct"] < -5
    assert np.mean([p["xyz"][2] for p in result["shapes"][0]["points"]]) == pytest.approx(DISTANCE, rel=0.005)


@pytest.mark.parametrize("over, message", [
    ({"baseline_mm": ""}, "baseline is required"),
    ({"baseline_mm": "0"}, "greater than zero"),
    ({"calib_source": "exif"}, "no EXIF focal length"),
    ({"fx": ""}, "fx is required"),
    ({"calib_source": "hw2"}, "No saved HW2 calibration"),
])
def test_missing_inputs_show_a_clear_error(client, pair, over, message):
    r = client.post("/hw7/stereo/upload", data=form(pair, **over), content_type="multipart/form-data",
                    follow_redirects=True)
    assert message in r.data.decode()
