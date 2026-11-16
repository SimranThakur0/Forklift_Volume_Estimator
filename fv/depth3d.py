"""Robust use of a metric point map (e.g. MoGe-2) for the load faces.

Why: reading the depth AT a clicked corner is unreliable (corners sit on object
boundaries where depth models smear foreground/background). Instead:
  1. fit a PLANE to all points inside each face (hundreds of thousands of pixels vote),
  2. take each clicked pixel's viewing RAY (its direction is correct even if its depth is not),
  3. intersect the ray with the face plane -> clean 3D corner.
Scale-free bonus: depth_over_length = |end-face edge| / |main-face row| is independent
of the model's absolute scale, so it tests how many pallets deep the load is.
"""
import numpy as np
import cv2


def _poly_mask(shape, poly, erode_px):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    return m.astype(bool)


def fit_plane(P, iters=4):
    """Robust plane fit. P (N,3). Returns unit normal n, centroid c, inlier rms (same units as P)."""
    P = P[np.isfinite(P).all(1)]
    if len(P) < 50:
        raise ValueError("too few valid points inside the face")
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        c = P[keep].mean(0)
        n = np.linalg.svd(P[keep] - c, full_matrices=False)[2][2]
        r = np.abs((P - c) @ n)
        sigma = 1.4826 * np.median(r[keep]) + 1e-9
        keep = r < max(2.5 * sigma, 1e-4)
    c = P[keep].mean(0)
    n = np.linalg.svd(P[keep] - c, full_matrices=False)[2][2]
    rms = float(np.sqrt(np.mean(((P[keep] - c) @ n) ** 2)))
    return n, c, rms


def pixel_ray(pts, xy, win=3):
    """Unit viewing ray of a pixel, taken from the point map's own direction."""
    H, W = pts.shape[:2]
    x, y = int(round(xy[0])), int(round(xy[1]))
    for w in range(win, win + 25, 4):
        sl = pts[max(0, y - w):min(H, y + w + 1), max(0, x - w):min(W, x + w + 1)].reshape(-1, 3)
        sl = sl[np.isfinite(sl).all(1)]
        if len(sl):
            d = sl / np.linalg.norm(sl, axis=1, keepdims=True)
            r = np.median(d, axis=0)
            return r / np.linalg.norm(r)
    raise ValueError(f"no valid depth near pixel {xy}")


def ray_plane(ray, n, c):
    return ray * (n @ c) / (n @ ray)


def _quad_points(pts, quad, erode_px):
    m = _poly_mask(pts.shape, quad, erode_px)
    return pts[m]


def perp3(p, a, b):
    return np.linalg.norm(np.cross(p - a, b - a)) / np.linalg.norm(b - a)


def analyze(pts, M, T, S=None, side_on="left", erode_px=8):
    """pts (H,W,3) metric point map in the image the pixel coordinates refer to.
    M (n+1,2), T (2,2), S (2,2) or None -- already in pts pixel coordinates.
    Returns dict (metres): row_len, load_len, height, seg_lens, plane_rms_main,
    and for 2-face frames: end_len, depth_over_length, plane_rms_end."""
    M, T = np.asarray(M, float), np.asarray(T, float)
    quad_main = [M[0], M[-1], T[1], T[0]]
    n1, c1, rms1 = fit_plane(_quad_points(pts, quad_main, erode_px))
    X = [ray_plane(pixel_ray(pts, p), n1, c1) for p in M]
    XT = [ray_plane(pixel_ray(pts, p), n1, c1) for p in T]
    out = {
        "row_len": float(np.linalg.norm(X[-1] - X[0])),
        "seg_lens": [float(np.linalg.norm(X[i + 1] - X[i])) for i in range(len(X) - 1)],
        "load_len": float(np.linalg.norm(XT[1] - XT[0])),
        "height": float(0.5 * (perp3(XT[0], X[0], X[-1]) + perp3(XT[1], X[0], X[-1]))),
        "plane_rms_main": rms1,
    }
    if S is not None:
        S = np.asarray(S, float)
        if side_on == "left":
            sh_b, sh_t = M[0], T[0]
            quad_end = [S[0], sh_b, sh_t, S[1]]
            shared3 = X[0]
        else:
            sh_b, sh_t = M[-1], T[1]
            quad_end = [sh_b, S[0], S[1], sh_t]
            shared3 = X[-1]
        n2, c2, rms2 = fit_plane(_quad_points(pts, quad_end, max(2, erode_px // 3)))
        Sb = ray_plane(pixel_ray(pts, S[0]), n2, c2)
        out["end_len"] = float(np.linalg.norm(Sb - shared3))
        out["depth_over_length"] = out["end_len"] / out["row_len"]
        out["plane_rms_end"] = rms2
        out["normal_angle_deg"] = float(np.degrees(np.arccos(min(1.0, abs(n1 @ n2)))))
    return out
