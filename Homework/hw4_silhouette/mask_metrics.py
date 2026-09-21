# compares my OpenCV mask against a SAM2 mask I ran somewhere else (Colab -
# see the README for the snippet I used). SAM2 itself never runs in this app,
# I just upload the mask PNG it spits out and score it here.

import cv2
import numpy as np


def load_binary_mask(path, target_shape_hw):
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"could not read mask image: {path}")
    resized = cv2.resize(raw, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def compare_masks(mask_opencv, mask_sam2):
    a = mask_opencv > 0
    b = mask_sam2 > 0
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    area_a, area_b = int(a.sum()), int(b.sum())

    iou = intersection / union if union else 0.0
    dice = (2.0 * intersection) / (area_a + area_b) if (area_a + area_b) else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "area_opencv_px": area_a,
        "area_sam2_px": area_b,
        "intersection_px": intersection,
        "union_px": union,
    }


def agreement_overlay(mask_opencv, mask_sam2):
    """green = both agree it's foreground, red = only my mask thinks so,
    blue = only SAM2 thinks so."""
    h, w = mask_opencv.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    a = mask_opencv > 0
    b = mask_sam2 > 0
    vis[np.logical_and(a, b)] = (60, 200, 60)
    vis[np.logical_and(a, np.logical_not(b))] = (40, 40, 220)
    vis[np.logical_and(np.logical_not(a), b)] = (220, 140, 40)
    return vis
