import io

import cv2
import numpy as np
import pytest

from Homework.hw2_calibration import routes as hw2
from Homework.hw_7 import routes as hw7

CAL = {"camera_matrix": [[3031.56, 0, 2084.16], [0, 3039.37, 1513.18], [0, 0, 1.0]],
       "dist_coeffs": [[0.05091, -0.09764, -0.00035, 0.00244, 0.02393]],
       "reprojection_error": 0.5, "image_size": [4032, 3024]}


def png(img):
    return io.BytesIO(cv2.imencode(".png", img)[1].tobytes())


def chessboard_views(cols=9, rows=6, square=60):
    """Perspective views of a board, so calibrateCamera has real input."""
    board = np.full(((rows + 3) * square, (cols + 3) * square), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                x, y = square * (c + 1), square * (r + 1)
                board[y:y + square, x:x + square] = 0
    h, w = board.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    views = []
    for dx, dy in ((70, 20), (-60, 40), (30, -70), (-40, -30), (90, 60), (-80, 10)):
        dst = src * 0.7 + [150, 120] + np.float32([[dx, dy], [0, 0], [-dy, dx], [0, -dx]])
        H = cv2.getPerspectiveTransform(src, np.float32(dst))
        views.append(cv2.warpPerspective(board, H, (1000, 800), borderValue=255))
    return views


@pytest.fixture
def client(tmp_path, monkeypatch):
    import main
    monkeypatch.setattr(hw2, "CALIB_FILE", str(tmp_path / "camera_calib.npz"))
    monkeypatch.setattr(hw2, "UPLOAD_CALIB_DIR", str(tmp_path))
    monkeypatch.setattr(hw2, "RESULTS_DIR", str(tmp_path))
    main.app.config["TESTING"] = True
    return main.app.test_client()


def save_calibration():
    np.savez(hw2.CALIB_FILE, camera_matrix=np.array(CAL["camera_matrix"]), dist_coeffs=np.array(CAL["dist_coeffs"]),
             reprojection_error=CAL["reprojection_error"], image_size=np.array(CAL["image_size"]))


def test_saved_intrinsics_stay_on_the_hw2_page(client):
    save_calibration()
    page = client.get("/hw2/calibrate").data.decode()
    assert "Saved camera intrinsics" in page
    for value in ("3031.56", "3039.37", "2084.16", "1513.18", "0.05091"):
        assert value in page


def test_hw7_pages_offer_and_show_the_saved_intrinsics(client):
    save_calibration()
    for url in ("/hw7/stereo", "/hw7/room"):
        page = client.get(url).data.decode()
        assert 'value="hw2"' in page and "fx 3031.56" in page


def test_without_a_calibration_pages_say_so(client):
    assert "Saved camera intrinsics" not in client.get("/hw2/calibrate").data.decode()
    for url in ("/hw7/stereo", "/hw7/room"):
        page = client.get(url).data.decode()
        assert 'value="hw2"' not in page and "No saved HW2 calibration yet" in page


def test_new_calibration_reaches_hw2_reload_and_hw7(client):
    views = chessboard_views()
    data = {"cols": "9", "rows": "6", "square": "25",
            "images": [(png(v), f"board{i}.png") for i, v in enumerate(views)]}
    r = client.post("/hw2/calibrate", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and b"Saved to static/calib/camera_calib.npz" in r.data

    fx = hw2.get_current_calibration()["camera_matrix"][0, 0]
    assert f"{fx:.2f}" in client.get("/hw2/calibrate").data.decode()
    assert f"fx {fx:.2f}" in client.get("/hw7/stereo").data.decode()


def test_hw7_processing_uses_the_saved_matrix(client):
    save_calibration()
    K, dist, source = hw7._intrinsics({"calib_source": "hw2"}, (4032, 3024))
    assert K[0, 0] == pytest.approx(3031.56)
    assert dist[0] == pytest.approx(0.05091)
    assert "HW2" in source
