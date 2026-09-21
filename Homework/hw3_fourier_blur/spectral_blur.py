# HW3 - blurring an image two different ways and checking they match.
#
# The whole point of this assignment is to prove (by actually running the
# numbers) that blurring in the spatial domain and blurring in the frequency
# domain are the same operation. So below there are two totally separate
# code paths that both compute a blur, and then we diff the outputs.
#
# path 1 - spatial: slide the kernel over the image (cv2.filter2D)
# path 2 - frequency: FFT the image, FFT the kernel, multiply, IFFT back
#
# If the convolution theorem is right, these should come out (almost)
# pixel-identical.

import numpy as np
import cv2


def build_kernel(kernel_type, ksize, sigma=None):
    """Make a normalized box or gaussian kernel (values sum to 1).

    Both kernels are symmetric, which is convenient: it means convolution
    and correlation are the same thing for them, so cv2.filter2D (which
    technically does correlation, not convolution) can be used directly
    without flipping the kernel first.
    """
    if ksize % 2 == 0:
        ksize += 1  # keep it centered on a pixel, odd sizes only

    if kernel_type == "box":
        kernel = np.ones((ksize, ksize), dtype=np.float64)
    elif kernel_type == "gaussian":
        s = sigma if sigma and sigma > 0 else ksize / 6.0
        ax = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * s * s))
    else:
        raise ValueError(f"unknown kernel_type: {kernel_type}")

    kernel /= kernel.sum()
    return kernel


def spatial_blur_channel(channel, kernel):
    """The "normal" way to blur - just convolve the kernel over the image.

    Using BORDER_CONSTANT (zero padding) here on purpose, so the edges match
    what the FFT path below does (FFT convolution is implicitly zero-padded).
    Otherwise the two methods would disagree right at the image borders for
    a reason that has nothing to do with the theorem.
    """
    return cv2.filter2D(channel.astype(np.float64), -1, kernel, borderType=cv2.BORDER_CONSTANT)


def frequency_blur_channel(channel, kernel):
    """Same blur, but done through the frequency domain.

    Trick here: if you just FFT the image and the kernel at the image's own
    size and multiply, you get circular convolution (wraps around at the
    edges), not the linear convolution filter2D gives you. To fix that, both
    get zero-padded up to (H + kh - 1, W + kw - 1) before transforming - that
    padded size is big enough that the wraparound never overlaps the real
    result. Afterwards we crop back down to the original H x W using the
    kernel's center offset.
    """
    H, W = channel.shape
    kh, kw = kernel.shape
    full_h, full_w = H + kh - 1, W + kw - 1

    image_spectrum = np.fft.fft2(channel.astype(np.float64), s=(full_h, full_w))
    kernel_spectrum = np.fft.fft2(kernel, s=(full_h, full_w))

    full_result = np.fft.ifft2(image_spectrum * kernel_spectrum).real

    pad_h, pad_w = kh // 2, kw // 2
    cropped = full_result[pad_h:pad_h + H, pad_w:pad_w + W]
    return cropped, image_spectrum, kernel_spectrum


def run_both_domains(image_bgr, kernel_type, ksize, sigma=None):
    """Runs both blur methods on every channel and packs up the results the
    web page needs: the two blurred images, how different they are, and the
    raw spectra for the spectrum-comparison figure.
    """
    kernel = build_kernel(kernel_type, ksize, sigma)

    if image_bgr.ndim == 2:
        channels = [image_bgr]
    else:
        channels = cv2.split(image_bgr)

    spatial_channels = []
    freq_channels = []
    last_image_spectrum = None
    last_kernel_spectrum = None

    for ch in channels:
        spatial = spatial_blur_channel(ch, kernel)
        freq, img_spec, ker_spec = frequency_blur_channel(ch, kernel)
        spatial_channels.append(spatial)
        freq_channels.append(freq)
        # only need one channel's spectrum for the picture, so just keep
        # overwriting and use whatever's left after the loop
        last_image_spectrum = img_spec
        last_kernel_spectrum = ker_spec

    def to_uint8(chans):
        clipped = [np.clip(c, 0, 255) for c in chans]
        if len(clipped) == 1:
            return clipped[0].astype(np.uint8)
        return cv2.merge([c.astype(np.uint8) for c in clipped])

    spatial_result = to_uint8(spatial_channels)
    freq_result = to_uint8(freq_channels)

    diff = np.abs(
        np.stack(spatial_channels, axis=-1) - np.stack(freq_channels, axis=-1)
        if len(spatial_channels) > 1
        else spatial_channels[0] - freq_channels[0]
    )
    metrics = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "mse": float(np.mean(diff ** 2)),
        "kernel_shape": kernel.shape,
    }

    return {
        "kernel": kernel,
        "spatial_result": spatial_result,
        "freq_result": freq_result,
        "metrics": metrics,
        "image_spectrum": last_image_spectrum,
        "kernel_spectrum": last_kernel_spectrum,
    }


def log_magnitude_spectrum(spectrum):
    """Turn a raw FFT array into something you can actually look at:
    log(1 + |F|), shifted so DC sits in the middle, scaled to 0-255.
    """
    mag = np.fft.fftshift(np.abs(spectrum))
    log_mag = np.log1p(mag)
    log_mag -= log_mag.min()
    if log_mag.max() > 0:
        log_mag = log_mag / log_mag.max() * 255.0
    return log_mag.astype(np.uint8)


def _as_bgr(img):
    """Make sure an image has 3 channels so it can sit next to color tiles
    in the same montage without cv2.hconcat complaining about shapes."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def build_comparison_panel(labeled_images, tile_height=260, pad=10, label_bar_h=30):
    """Stitch a list of (label, image) pairs into one labeled strip, e.g.

        [Original] [Spatial] [Frequency] [|diff| x40]

    so the whole comparison shows up as a single PNG instead of four
    separate <img> tags scattered around the page. Each tile gets resized
    to the same height, a small dark bar with its caption burned on top,
    and a thin gap between tiles so it's obvious where one image ends and
    the next starts.
    """
    tiles = []
    for label, img in labeled_images:
        img = _as_bgr(img)
        h, w = img.shape[:2]
        scale = tile_height / h
        resized = cv2.resize(img, (int(round(w * scale)), tile_height))

        bar = np.zeros((label_bar_h, resized.shape[1], 3), dtype=np.uint8)
        bar[:] = (35, 35, 35)
        cv2.putText(bar, label, (8, label_bar_h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        tile = np.vstack([bar, resized])
        tiles.append(tile)

    # equal-width gap between tiles so it reads as separate panels, not one blob
    gap = np.full((tiles[0].shape[0], pad, 3), 255, dtype=np.uint8)
    row = tiles[0]
    for t in tiles[1:]:
        row = np.hstack([row, gap, t])

    # a little white margin around the whole thing
    border = 12
    framed = cv2.copyMakeBorder(row, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return framed
