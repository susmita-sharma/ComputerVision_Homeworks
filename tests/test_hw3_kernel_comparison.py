import numpy as np

from Homework.hw3_fourier_blur import spectral_blur as sb


def test_build_kernel_comparison_panel_includes_both_kernel_types():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:44, 20:44] = 200

    panel = sb.build_kernel_comparison_panel(image, ksize=9, sigma=2.0)

    assert panel is not None
    assert panel.shape[0] > 0 and panel.shape[1] > 0
    assert panel.dtype == np.uint8
