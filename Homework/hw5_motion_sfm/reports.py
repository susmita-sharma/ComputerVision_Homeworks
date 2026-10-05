# Human-readable write-ups saved next to each run's outputs (and shown on
# the pages): the tracking worked example and the full SfM math workout,
# with the real numbers from that run plugged in.

import numpy as np


def bmatrix(M, fmt="{:.4g}"):
    """numpy array -> LaTeX bmatrix string (for MathJax on the pages)."""
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))

    def cell(v):
        txt = fmt.format(v)
        return txt[1:] if txt.startswith("-") and not txt.strip("-0.") else txt  # no "-0"

    rows = [" & ".join(cell(v) for v in row) for row in M]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def _md_matrix(M, fmt="{:10.4f}"):
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    return "\n".join("    [" + " ".join(fmt.format(v) for v in row) + " ]" for row in M)


# ------------------------------------------------------------ tracking --

def tracking_report_md(label, rows, logs, window, bil, summary, wk=0):
    lines = [f"# Two-frame tracking validation - {label}", "",
             f"Window {window}x{window} px, pyramid levels {len(logs[0]) if logs else 0}.", "",
             "Columns: start (x0,y0) in frame t; from-scratch LK prediction; NCC template-match location "
             "(independent 'actual' position); OpenCV pyramidal LK; errors in px; RMS brightness residual "
             "of the window with zero motion vs. at the predicted location.", "",
             "| pt | x0, y0 | LK (ours) | NCC actual | OpenCV LK | err vs NCC | err vs OpenCV | NCC score | RMS no-motion | RMS tracked |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in enumerate(rows, 1):
        lines.append(f"| {k} | {r['x0']:.1f}, {r['y0']:.1f} | {r['x_pred']:.2f}, {r['y_pred']:.2f} | "
                     f"{r['x_match']:.2f}, {r['y_match']:.2f} | {r['x_actual']:.2f}, {r['y_actual']:.2f} | "
                     f"{r['err_vs_match_px']:.3f} | {r['agreement_err_px']:.3f} | {r['ncc']:.3f} | "
                     f"{r['rms_no_motion']:.2f} | {r['rms_tracked']:.2f} |")
    lines += ["", f"Mean |LK - NCC| = {summary['mean_err_match']:.3f} px, mean |LK - OpenCV| = "
              f"{summary['mean_err_cv']:.3f} px, mean RMS brightness residual {summary['mean_rms_before']:.2f} -> "
              f"{summary['mean_rms_after']:.2f} grey levels.", ""]

    if logs:
        r = rows[wk]
        lg0 = logs[wk][-1]
        lines += [f"## Worked example - point {wk + 1}", "",
                  f"Start (x0, y0) = ({r['x0']:.2f}, {r['y0']:.2f}).", "",
                  "Coarse-to-fine pyramid (displacement estimate d in that level's pixels):", "",
                  "| level | scale | initial guess | result d | iterations | converged |", "|---|---|---|---|---|---|"]
        for lg in logs[wk]:
            lines.append(f"| {lg['level']} | 1/{int(lg['scale'])} | ({lg['guess_in'][0]:.3f}, {lg['guess_in'][1]:.3f}) | "
                         f"({lg['d_out'][0]:.3f}, {lg['d_out'][1]:.3f}) | {lg['n_iters']} | {lg['converged']} |")
        if "Sxx" in lg0:
            lines += ["", "Full-resolution normal equations over the window "
                      f"({lg0['n_pixels']} pixels):", "",
                      f"    A^T A = [[{lg0['Sxx']:.1f}, {lg0['Sxy']:.1f}], [{lg0['Sxy']:.1f}, {lg0['Syy']:.1f}]]",
                      f"    eigenvalues: {lg0['eig_min']:.1f}, {lg0['eig_max']:.1f}  (both large -> corner-like, well conditioned)",
                      "", "Newton iterations ( [du dv]^T = (A^T A)^-1 * -[sum IxIt, sum IyIt]^T ):", "",
                      "| it | x | y | sum IxIt | sum IyIt | du | dv | RMS It |", "|---|---|---|---|---|---|---|---|"]
            for itr in lg0["iters"]:
                lines.append(f"| {itr['it']} | {itr['x']:.3f} | {itr['y']:.3f} | {itr['SxIt']:.1f} | {itr['SyIt']:.1f} | "
                             f"{itr['du']:+.4f} | {itr['dv']:+.4f} | {itr['rms_residual']:.2f} |")
    if bil:
        lines += ["", "## Worked example - bilinear interpolation", "",
                  f"Sample frame t+1 at the tracked location (x, y) = ({bil['x']:.4f}, {bil['y']:.4f}):", "",
                  f"    x0 = {bil['x0']}, y0 = {bil['y0']}, wx = {bil['wx']:.4f}, wy = {bil['wy']:.4f}",
                  f"    I(x0,y0) = {bil['Ia']:.0f}, I(x0+1,y0) = {bil['Ib']:.0f}, I(x0,y0+1) = {bil['Ic']:.0f}, I(x0+1,y0+1) = {bil['Id']:.0f}",
                  f"    R0 = (1-wx)*{bil['Ia']:.0f} + wx*{bil['Ib']:.0f} = {bil['R0']:.4f}",
                  f"    R1 = (1-wx)*{bil['Ic']:.0f} + wx*{bil['Id']:.0f} = {bil['R1']:.4f}",
                  f"    I(x,y) = (1-wy)*R0 + wy*R1 = {bil['value']:.4f}",
                  f"    cv2.remap (reference) = {bil['cv2_remap']:.4f};  nearest pixel would give {bil['nearest']:.0f}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- sfm --

def sfm_report_md(res, meta):
    r = res
    K = r["K"]
    lines = [
        "# Structure from motion - planar object from 4 viewpoints", "",
        "## Object and images", "",
        f"- Object: flat rectangle, measured {meta['width_mm']} mm x {meta['height_mm']} mm",
        f"- World frame: origin at the top-left corner, X along the top edge, Y down the side edge, Z = 0 on the object",
        "- Image sizes (as displayed, after the phone's EXIF rotation): "
        + ", ".join(f"view {i + 1} {w}x{h}" for i, (w, h) in enumerate(meta.get("image_sizes", [meta["image_size"]] * 4))),
        f"- Images: {', '.join(meta['image_names'])}",
    ]
    if meta.get("notes"):
        lines += [f"- Capture notes: {meta['notes']}"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = meta["image_size"]
    lines += ["", "## Camera intrinsics (shared by all 4 views)", "",
              f"Source: {meta['k_source']}", "", "    K =", _md_matrix(K, "{:10.2f}"), "",
              f"- fx = {fx:.2f} px, fy = {fy:.2f} px, principal point ({cx:.1f}, {cy:.1f})",
              f"- FOV across the long side = 2 atan(long / 2f) = {np.degrees(2 * np.arctan(max(w, h) / (2 * fx))):.1f} deg, "
              f"across the short side = {np.degrees(2 * np.arctan(min(w, h) / (2 * fy))):.1f} deg",
              f"- lens distortion: {meta['dist_text']}", ""]
    if len({tuple(np.round(p["K"], 3).ravel()) for p in r["poses"]}) > 1:
        lines += ["Per-view intrinsics (same focal length; axes/principal point follow each photo's orientation):", ""]
        for i, p in enumerate(r["poses"], 1):
            lines += [f"    K{i} =", _md_matrix(p["K"], "{:10.2f}"), ""]

    lines += ["## Camera positions (extrinsics)", "",
              "| cam | position C = -R^T t (mm) | distance to object centre (mm) | tilt from plane normal (deg) | Rodrigues r (deg) | reproj. RMS (px) |",
              "|---|---|---|---|---|---|"]
    for i, p in enumerate(r["poses"], 1):
        lines.append(f"| {i} | ({p['C'][0]:.1f}, {p['C'][1]:.1f}, {p['C'][2]:.1f}) | {p['distance']:.1f} | "
                     f"{p['tilt_deg']:.1f} | ({p['rvec_deg'][0]:.1f}, {p['rvec_deg'][1]:.1f}, {p['rvec_deg'][2]:.1f}) | "
                     f"{p['reproj_rms']:.2f} |")

    lines += ["", "## Step 1 - homography from the 4 corners (per view)", "",
              "World corners (X, Y) = (0,0), (W,0), (W,H), (0,H). Each correspondence gives two rows of A h = 0:",
              "", "    [-X -Y -1  0  0  0  xX  xY  x]", "    [ 0  0  0 -X -Y -1  yX  yY  y]", "",
              "h = right singular vector of A (8 x 9) with the smallest singular value; H normalised so H33 = 1.", ""]
    for i, (p, v) in enumerate(zip(r["poses"], meta["views"]), 1):
        hp = p["hpose"]
        lines += [f"### View {i}", "",
                  "Clicked corners (px): " + ", ".join(f"({x:.1f}, {y:.1f})" for x, y in v["corners"]), "",
                  "    A =", _md_matrix(p["A_h"], "{:11.1f}"), "", "    H =", _md_matrix(p["H"], "{:12.5f}"), "",
                  "Step 2 - pose from H:  B = K^-1 H = [b1 b2 b3],  lambda = 1/||b1||", "",
                  "    B =", _md_matrix(hp["B"], "{:12.6f}"), "",
                  f"    lambda = {hp['lambda']:.4f}",
                  f"    r1 = lambda b1 = {np.round(hp['R_raw'][:, 0], 4).tolist()}",
                  f"    r2 = lambda b2 = {np.round(hp['R_raw'][:, 1], 4).tolist()}",
                  f"    r3 = r1 x r2  = {np.round(hp['R_raw'][:, 2], 4).tolist()}",
                  f"    t  = lambda b3 = {np.round(hp['t'], 2).tolist()} mm", "",
                  "    R (orthonormalised via SVD) =", _md_matrix(hp["R"], "{:9.4f}"), "",
                  "Refined with solvePnP (iterative, minimises reprojection error):", "",
                  "    R =", _md_matrix(p["R"], "{:9.4f}"),
                  f"    t = {np.round(p['t'], 2).tolist()} mm",
                  f"    difference homography-pose vs PnP: rotation {p['R_diff_deg']:.3f} deg, translation {p['t_diff']:.2f} mm",
                  f"    camera centre C = -R^T t = {np.round(p['C'], 2).tolist()} mm", "",
                  "    P = K [R | t] =", _md_matrix(p["P"], "{:12.3f}"), ""]

    wt = r.get("worked_triangulation")
    if wt is not None:
        lines += ["## Step 3 - triangulating boundary point B1 (worked)", "",
                  "Observations: " + ", ".join(f"view {k + 1} ({x:.1f}, {y:.1f})" for k, (x, y) in enumerate(wt["obs"])), "",
                  "Each view adds rows  x P3 - P1  and  y P3 - P2  (Pk = k-th row of P):", "",
                  "    A =", _md_matrix(wt["A"], "{:12.3f}"), "",
                  f"    singular values: {np.round(wt['S'], 5).tolist()}",
                  f"    X_h (last right singular vector) = {np.round(wt['X_h'], 6).tolist()}",
                  f"    X = X_h[:3] / X_h[3] = {np.round(wt['X'], 2).tolist()} mm", ""]

    lines += ["## Reconstructed points (world mm)", "", "| point | X | Y | Z |", "|---|---|---|---|"]
    for j, X in enumerate(r["corner_3d"], 1):
        lines.append(f"| corner {j} | {X[0]:.2f} | {X[1]:.2f} | {X[2]:.3f} |")
    for j, X in enumerate(r["boundary_3d"], 1):
        lines.append(f"| B{j} | {X[0]:.2f} | {X[1]:.2f} | {X[2]:.3f} |")

    be = r["boundary_estimate"]
    e = r["edge_lengths"]
    lines += ["", "## Estimated boundary and validation", "",
              f"- Edge lengths from re-triangulated corners: top {e[0]:.1f}, right {e[1]:.1f}, bottom {e[2]:.1f}, "
              f"left {e[3]:.1f} mm (measured {meta['width_mm']} x {meta['height_mm']})",
              f"- Estimated outline (corners + boundary points ordered around their centroid): area {be['area']:.0f} mm^2 "
              f"(rectangle {meta['width_mm'] * meta['height_mm']:.0f} mm^2), perimeter {be['perimeter']:.1f} mm",
              f"- Corner recovery error (mm): {[round(x, 3) for x in r['corner_recovery_error']]}",
              f"- Planarity RMS (distance of all points to best-fit plane): {r['planarity_rms']:.3f} mm",
              f"- Reprojection RMS per view (px): {[round(p['reproj_rms'], 2) for p in r['poses']]}"]
    return "\n".join(lines) + "\n"
