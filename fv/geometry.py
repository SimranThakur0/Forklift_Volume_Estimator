"""Core measurement maths. Vectorised over a leading batch dim so Monte Carlo reuses it.

Annotation per frame (pixels):
  M : (n+1,2)  bottom edge of the MAIN face, left->right, at the ends and at every
               joint between pallets (n = pallets_along). Segment i is ONE pallet.
  T : (2,2)    top-left and top-right corner of the load on the main face
  S : (2,2) or None   END face: [outer-bottom, outer-top]. The shared corner is
               M[0]/T[0] (side_on=left) or M[-1]/T[1] (side_on=right).

Model (weak perspective, camera roughly at load height, vertical edges vertical):
  main face real length La = n * Lp ; end face real depth Lb = deep * Ls
  TWO faces:  a/La = k cos(yaw),  b/Lb = k sin(yaw)
              -> k = sqrt((a/La)^2 + (b/Lb)^2),  yaw = atan2(b/Lb, a/La)
  ONE face:   k = a / (La cos(yaw)), yaw must be assumed (sampled in Monte Carlo)
  Height uses the LOCAL scale at each end (first / last pallet segment), which
  absorbs part of the perspective change across the face.
  Length = La * top_px / a_px (ratio trick: yaw cancels).
  Depth  = Lb * side_top_px / side_bottom_px (two faces) or Lb (one face).
"""
import numpy as np


def _n(v):
    return np.linalg.norm(v, axis=-1)


def perp_dist(p, a, b):
    ab, ap = b - a, p - a
    return np.abs(ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0]) / _n(ab)


def estimate(M, T, S, deep, Lp, Ls, side_on="left", theta=None, fill=1.0,
             f_main=1.0, f_side=1.0):
    M, T = np.asarray(M, float), np.asarray(T, float)
    n = M.shape[-2] - 1
    m0, mn = M[..., 0, :], M[..., -1, :]
    a_px = _n(mn - m0)
    La, Lb = n * Lp * f_main, deep * Ls * f_side

    if S is not None:
        S = np.asarray(S, float)
        sh_b = m0 if side_on == "left" else mn
        sh_t = T[..., 0, :] if side_on == "left" else T[..., 1, :]
        b_px, bt_px = _n(S[..., 0, :] - sh_b), _n(S[..., 1, :] - sh_t)
        k = np.sqrt((a_px / La) ** 2 + (b_px / Lb) ** 2)
        theta = np.arctan2(b_px / Lb, a_px / La)
        D = Lb * bt_px / b_px
    else:
        theta = np.asarray(theta, float) * np.ones_like(a_px)
        k = a_px / (La * np.cos(theta))
        D = Lb * np.ones_like(a_px)

    seg0, segn = _n(M[..., 1, :] - M[..., 0, :]), _n(M[..., -1, :] - M[..., -2, :])
    ct = np.cos(theta)
    k_l, k_r = seg0 / (Lp * f_main * ct), segn / (Lp * f_main * ct)
    h_l, h_r = perp_dist(T[..., 0, :], m0, mn), perp_dist(T[..., 1, :], m0, mn)
    H = 0.5 * (h_l / k_l + h_r / k_r)
    L = La * _n(T[..., 1, :] - T[..., 0, :]) / a_px
    V = L * D * H * fill / 1e9
    return {"L_mm": L, "D_mm": D, "H_mm": H, "V_m3": V, "yaw_deg": np.degrees(theta),
            "persp_ratio": seg0 / segn, "mm_per_px": 1.0 / k}


def crate_estimate(n_across, n_layers, cfg, deep):
    """Independent estimate from counting crates on the main face."""
    if not n_across or not n_layers:
        return None
    from .common import pallet_sides
    Lp, Ls = pallet_sides(cfg)
    along = cfg["pallets_along"]
    L = n_across * cfg["crate"]["along_face_mm"]
    H = n_layers * cfg["crate"]["height_mm"]
    D = deep * Ls
    return {"L_mm": L, "H_mm": H, "V_m3": L * D * H / 1e9,
            "span_vs_pallets": L / (along * Lp), "crates_per_pallet": n_across / along}


def frame_arrays(a):
    M = np.array(a["M"], float)
    T = np.array(a["T"], float)
    S = np.array(a["S"], float) if a.get("S") else None
    return M, T, S


def point_estimate(a, cfg, deep=None):
    from .common import pallet_sides
    Lp, Ls = pallet_sides(cfg)
    deep = deep or cfg["pallets_deep"]
    M, T, S = frame_arrays(a)
    theta = None if S is not None else 0.0   # one face: nominal side-on (see MC for spread)
    r = estimate(M, T, S, deep, Lp, Ls, a.get("side_on", cfg["side_on"]), theta)
    return {k: float(v) for k, v in r.items()}, deep


def monte_carlo(a, cfg, N, rng, deep=None):
    from .common import pallet_sides
    mc = cfg["mc"]
    Lp, Ls = pallet_sides(cfg)
    deep = deep or cfg["pallets_deep"]
    M, T, S = frame_arrays(a)
    s = mc["click_sigma_px"]
    Ms = M + rng.normal(0, s, (N,) + M.shape)
    Ts = T + rng.normal(0, s, (N,) + T.shape)
    Ss = None if S is None else S + rng.normal(0, s, (N,) + S.shape)
    theta = np.deg2rad(np.minimum(np.abs(rng.normal(0, mc["yaw_deg_sigma"], N)), mc["yaw_deg_max"]))
    fill = rng.uniform(mc["fill_low"], mc["fill_high"], N)
    fm, fs = 1 + rng.normal(0, mc["dim_tol"], N), 1 + rng.normal(0, mc["dim_tol"], N)
    r = estimate(Ms, Ts, Ss, deep, Lp, Ls, a.get("side_on", cfg["side_on"]),
                 theta, fill, fm, fs)
    V = r["V_m3"]
    return V[np.isfinite(V)]


def passes_qc(m, cfg):
    """m = point_estimate() dict. Returns (ok, reason)."""
    q = cfg.get("qc", {})
    why = []
    if m["yaw_deg"] > q.get("max_yaw_deg", 40):
        why.append(f"yaw {m['yaw_deg']:.0f}")
    pr = max(m["persp_ratio"], 1 / m["persp_ratio"])
    if pr > q.get("max_persp", 1.3):
        why.append(f"persp {pr:.2f}")
    return (not why), ", ".join(why)
