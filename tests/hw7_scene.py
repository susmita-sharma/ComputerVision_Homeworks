"""Ray-cast renderer for HW7 tests: textured planes seen by a pinhole camera,
so every test knows the true size of what the pipeline has to measure."""

import cv2
import numpy as np


def texture(seed, size=1500):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    for cell, weight in ((4, 0.2), (16, 0.35), (64, 0.45)):
        n = size // cell + 2
        layer = cv2.resize(rng.uniform(0, 255, (n, n, 3)).astype(np.float32), (size, size),
                           interpolation=cv2.INTER_NEAREST)
        img += weight * layer
    return np.clip(img, 0, 255).astype(np.uint8)


class Plane:
    """Rectangle from origin along unit axes u, v, with mm extents (w, h)."""

    def __init__(self, origin, u, v, w, h, seed, mask=None):
        self.o = np.asarray(origin, float)
        self.u = np.asarray(u, float) / np.linalg.norm(u)
        self.v = np.asarray(v, float) / np.linalg.norm(v)
        self.n = np.cross(self.u, self.v)
        self.w, self.h = w, h
        self.tex = texture(seed)
        self.mask = mask

    def point(self, a, b):
        return self.o + a * self.u + b * self.v


def render(planes, K, R, C, size):
    """R is world->camera, C the camera centre in the world."""
    w, h = size
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = np.stack([us, vs, np.ones_like(us)], axis=-1) @ (np.linalg.inv(K).T @ R)
    zbuf = np.full((h, w), np.inf)
    out = np.full((h, w, 3), 40, np.uint8)
    for pl in planes:
        denom = rays @ pl.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.dot(pl.o - C, pl.n) / denom
        hit = np.where(np.isfinite(t) & (t > 0), t, np.inf)
        pts = C + np.nan_to_num(hit, posinf=0.0)[..., None] * rays
        a = (pts - pl.o) @ pl.u
        b = (pts - pl.o) @ pl.v
        vis = np.isfinite(hit) & (a >= 0) & (a <= pl.w) & (b >= 0) & (b <= pl.h) & (hit < zbuf)
        if pl.mask is not None:
            vis &= pl.mask(a, b)
        s = pl.tex.shape[0] - 1
        col = cv2.remap(pl.tex, (a / pl.w * s).astype(np.float32), (b / pl.h * s).astype(np.float32), cv2.INTER_LINEAR)
        out[vis] = col[vis]
        zbuf[vis] = hit[vis]
    return out


def project(K, R, C, X):
    h = K @ (R @ (np.asarray(X, float) - C))
    return h[:2] / h[2]


def yaw(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
