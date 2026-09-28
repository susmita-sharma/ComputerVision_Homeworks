# HW5 part 1 - dense optical flow over a real video clip.
#
# Pipeline: read frames -> convert to gray -> Farneback dense flow between
# every consecutive pair -> color-code each flow field (Middlebury-style
# HSV wheel: hue = direction, value = speed) -> write those as a video ->
# also roll up per-frame statistics (mean/max speed, moving-area fraction,
# dominant direction) so the "what can be inferred" write-up has actual
# numbers behind it instead of just a picture.

import os

import cv2
import numpy as np


def dense_flow(prev_gray, next_gray):
    """Farneback dense optical flow: one (dx, dy) vector per pixel."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def flow_to_color(flow):
    """Classic HSV flow visualization: angle -> hue, magnitude -> value.

    This is the standard way to turn a 2-channel vector field into
    something a human can read at a glance - color tells you *which way*
    things moved, brightness tells you *how fast*.
    """
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)          # hue: 0-180 for OpenCV's 0-360 angle
    hsv[..., 1] = 255                                   # full saturation
    mag_norm = np.clip(mag / (mag.max() + 1e-6) * 255, 0, 255)
    hsv[..., 2] = mag_norm.astype(np.uint8)             # value: normalized speed

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_stats(flow):
    """Numeric summary of one flow field - the "evidence" numbers."""
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    moving_mask = mag > 1.0  # pixels that moved more than ~1px between frames
    moving_frac = float(moving_mask.mean())

    if moving_mask.any():
        dominant_dir = float(np.median(ang[moving_mask]))
    else:
        dominant_dir = 0.0

    return {
        "mean_mag": float(mag.mean()),
        "max_mag": float(mag.max()),
        "moving_frac": moving_frac,
        "dominant_dir_deg": dominant_dir,
    }


def draw_direction_wheel(size=90):
    """Small legend image: color <-> direction, so the flow video's colors
    are actually decodable instead of just "pretty".
    """
    wheel = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size / 2, size / 2
    ys, xs = np.mgrid[0:size, 0:size]
    dx = (xs - cx)
    dy = (ys - cy)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (size / 2) * 255, 0, 255).astype(np.uint8)
    wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    wheel[mag > size / 2] = 0
    return wheel


def process_video(video_path, out_dir, stamp, max_seconds=30, max_dim=480, sample_stride=1):
    """Full HW5-part-1 pipeline for one uploaded video.

    - reads up to `max_seconds` of frames starting at t=0
    - downsizes so Farneback stays fast in a web request
    - computes flow for every consecutive pair, writes an MP4 of the
      color-coded flow field side-by-side with the original frame
    - returns file URLs (relative to /static) plus the stats time series
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open that video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if fps > 1 else 25.0
    max_frames = int(max_seconds * fps)

    frames = []
    count = 0
    while count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if count % sample_stride == 0:
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frames.append(frame)
        count += 1
    cap.release()

    if len(frames) < 2:
        raise ValueError("Need at least 2 readable frames - clip too short or unreadable.")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    h, w = frames[0].shape[:2]
    wheel = draw_direction_wheel(size=min(90, h // 3))
    wheel_h, wheel_w = wheel.shape[:2]

    out_path = os.path.join(out_dir, f"{stamp}_flow.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / sample_stride, (w * 2, h))

    series = []
    sample_frames = {}
    n_pairs = len(frames) - 1
    mid = n_pairs // 2

    for i in range(n_pairs):
        flow = dense_flow(grays[i], grays[i + 1])
        color = flow_to_color(flow)
        color[0:wheel_h, 0:wheel_w] = wheel  # burn the direction legend into the corner

        side_by_side = np.hstack([frames[i], color])
        writer.write(side_by_side)

        stats = flow_stats(flow)
        stats["t_sec"] = i / (fps / sample_stride)
        series.append(stats)

        if i in (0, mid, n_pairs - 1):
            sample_frames[i] = {
                "frame_bgr": frames[i].copy(),
                "flow_color": color.copy(),
                "stats": stats,
            }

    writer.release()

    return {
        "flow_video_path": out_path,
        "series": series,
        "sample_frames": sample_frames,
        "fps": fps,
        "n_frames_used": len(frames),
        "frame_size": (w, h),
    }
