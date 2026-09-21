# stuff shared by both the RGB and thermal pipelines - thresholding, cleanup,
# watershed, contour extraction. all plain OpenCV, no trained models or GMMs
# anywhere in here (the assignment specifically says no ML/DL so I kept it to
# thresholds, morphology and the classic watershed algorithm)

import cv2
import numpy as np


def border_touch_fraction(mask):
    """how much of the image border is foreground. Otsu doesn't know which
    side of the threshold is the person and which is the background, so this
    is the trick I use to guess: in most photos the background touches the
    frame edges way more than the subject does.
    """
    h, w = mask.shape
    border_pixels = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(np.mean(border_pixels > 0))


def choose_foreground_by_border(mask_a, mask_b):
    # whichever mask touches the border less is probably the person, not the wall behind them
    return mask_a if border_touch_fraction(mask_a) <= border_touch_fraction(mask_b) else mask_b


def largest_component_mask(binary_mask):
    """keep just the biggest blob, drop everything else. assumes the human
    is the largest connected foreground region, which held up fine in my
    test photos but obviously breaks if there's a second big object."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if n_labels <= 1:
        return binary_mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (np.uint8(labels == largest_label)) * 255


def fill_holes(mask):
    """flood fill in from the corner to mark the real background, then
    invert - anything left over that wasn't reached is an enclosed hole
    inside the person (like a gap between an arm and the torso)."""
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes_filled = cv2.bitwise_not(flood)
    return mask | holes_filled


def clean_mask(mask, close_ksize=9, open_ksize=5):
    # close first to seal small gaps, then open to knock out stray noise specks
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
    return fill_holes(opened)


def watershed_refine(gray, mask):
    """marker-based watershed to tighten the boundary up against the actual
    intensity edges instead of leaving it as a blocky threshold outline.
    this is the classic flooding algorithm, not anything learned.
    """
    if cv2.countNonZero(mask) == 0:
        return mask

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return mask
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 1:
        return mask
    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    refined = np.uint8(markers > 1) * 255
    return refined if cv2.countNonZero(refined) > 0 else mask


def extract_boundary(mask):
    """grab the biggest outer contour in the mask - that's our boundary."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    return largest, filled


def draw_boundary_overlay(image_bgr, contour, color=(0, 255, 60), thickness=3):
    vis = image_bgr.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, color, thickness)
    return vis


def run_classical_pipeline(gray, image_bgr, raw_mask):
    """the part that's identical for RGB and thermal - once you've got a
    rough binary mask with the polarity figured out, everything from here
    down is the same: biggest blob, clean it up, watershed, contour.
    """
    largest = largest_component_mask(raw_mask)
    cleaned = clean_mask(largest)
    refined = watershed_refine(gray, cleaned)
    refined = largest_component_mask(refined)
    contour, boundary_mask = extract_boundary(refined if cv2.countNonZero(refined) else cleaned)
    overlay = draw_boundary_overlay(image_bgr, contour)
    return {
        "otsu_mask": raw_mask,
        "cleaned_mask": cleaned,
        "refined_mask": refined,
        "boundary_mask": boundary_mask,
        "contour": contour,
        "overlay": overlay,
    }
