import cv2
import numpy as np

from Homework.hw5_motion_sfm import bilinear


def test_bilinear_matches_cv2_remap():
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=(40, 50)).astype(np.float64)

    xs = rng.uniform(1, 48, size=200)
    ys = rng.uniform(1, 38, size=200)

    ours = bilinear.bilinear_sample(image, xs, ys)

    map_x = xs.reshape(-1, 1).astype(np.float32)
    map_y = ys.reshape(-1, 1).astype(np.float32)
    cv2_result = cv2.remap(image.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR)
    cv2_result = cv2_result.reshape(-1)

    assert np.allclose(ours, cv2_result, atol=1e-2)


def test_bilinear_exact_at_integer_coordinates():
    image = np.arange(30, dtype=np.float64).reshape(5, 6)
    xs = np.array([0, 3, 5])
    ys = np.array([0, 2, 4])
    result = bilinear.bilinear_sample(image, xs, ys)
    assert np.allclose(result, image[ys, xs])
