# Bilinear interpolation, written from scratch (no cv2.remap here on
# purpose - this is the actual sub-pixel sampler the from-scratch
# Lucas-Kanade tracker in tracking.py leans on).
#
# The idea: a digital image only gives us intensity at integer (x, y).
# Optical flow displacements are almost never integers, so to ask "what is
# the intensity of frame 2 at (x + 3.7, y - 1.2)" we need to invent a value
# in between the four nearest pixels. Bilinear interpolation does that by
# treating intensity as varying linearly along x, then linearly along y.

import numpy as np


def bilinear_sample(image, xs, ys):
    """Sample `image` (H, W) float array at fractional coordinates xs, ys.

    xs, ys can be scalars or same-shaped arrays. Points that fall outside
    the valid interior are clamped to the border (matches cv2.remap's
    BORDER_REPLICATE) rather than raising, since the LK iterations can
    briefly step outside the image while converging.

    Returns an array the same shape as xs/ys.
    """
    image = np.asarray(image, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    H, W = image.shape

    xs = np.clip(xs, 0.0, W - 1 - 1e-6)
    ys = np.clip(ys, 0.0, H - 1 - 1e-6)

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = xs - x0  # fraction of the way from x0 to x1
    wy = ys - y0

    Ia = image[y0, x0]  # top-left
    Ib = image[y0, x1]  # top-right
    Ic = image[y1, x0]  # bottom-left
    Id = image[y1, x1]  # bottom-right

    top = Ia * (1 - wx) + Ib * wx
    bottom = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bottom * wy


def bilinear_sample_patch(image, cx, cy, half_size):
    """Sample a square patch of side (2*half_size + 1) centered at the
    possibly-fractional point (cx, cy). Used by the LK tracker to grab a
    window out of frame 2 at a warped, sub-pixel location.
    """
    offsets = np.arange(-half_size, half_size + 1)
    grid_x, grid_y = np.meshgrid(offsets, offsets)
    xs = cx + grid_x
    ys = cy + grid_y
    return bilinear_sample(image, xs, ys)


def _derivation_markdown():
    """Not called anywhere - the derivation lives here as the single
    source of truth and the hw5 tracking template quotes it verbatim, so
    the explanation and the implementation can't drift apart.
    """
    return r"""
    Bilinear interpolation derivation
    ----------------------------------
    Given a query point (x, y) with x0 = floor(x), x1 = x0 + 1,
    y0 = floor(y), y1 = y0 + 1, and known intensities at the four
    surrounding integer pixels I(x0,y0), I(x1,y0), I(x0,y1), I(x1,y1):

    1. Interpolate linearly along x at row y0:
         R0 = I(x0,y0) * (1 - wx) + I(x1,y0) * wx,   wx = x - x0
    2. Interpolate linearly along x at row y1:
         R1 = I(x0,y1) * (1 - wx) + I(x1,y1) * wx
    3. Interpolate linearly along y between those two results:
         I(x,y) ~= R0 * (1 - wy) + R1 * wy,           wy = y - y0

    Expanding gives the single bilinear form:
      I(x,y) ~= I(x0,y0)(1-wx)(1-wy) + I(x1,y0)wx(1-wy)
               + I(x0,y1)(1-wx)wy   + I(x1,y1)wx wy
    which is exactly what bilinear_sample computes.
    """
