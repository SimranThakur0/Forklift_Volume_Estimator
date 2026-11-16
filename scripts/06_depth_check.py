"""STEP 6 (optional) - learned metric geometry cross-check with MoGe-2 (CPU is OK for a few frames).

    pip install torch torchvision git+https://github.com/microsoft/MoGe.git
    python scripts/06_depth_check.py [--resolution_level 9]

v2: corners are NOT read from the depth map directly (unreliable on object boundaries).
We fit a plane to each load face, intersect your clicked pixel rays with it (fv/depth3d.py).
The old 'naive' corner-lift numbers are kept in the CSV only for comparison.

Columns (per frame):
  pallet_ratio  model's length of your pallet row / nominal (n x pallet side). 1.0 = model's absolute scale trusted
  seg_ratio     first pallet / last pallet length from the model (should be ~1.0 for equal pallets)
  V_norm        volume with the model's SHAPE but the pallet row as the ruler (depth = nominal pallets_deep)
  depth/length  2-face frames only: end-face edge / main row, a SCALE-FREE test of pallets_deep.
                3 pallets deep -> about (3*Ls)/(n*Lp) = 1.00 ; 2 deep -> 0.67   (noisy: the end face is thin)
  implied_deep  the same ratio converted to a pallet count
  plane_rms_m   how flat the model thinks the face is (small = good)
"""
import _path  # noqa
import argparse, csv
import cv2, numpy as np
from fv.common import load_config, load_annotations, find_image, pallet_sides, ROOT
from fv.depth3d import analyze, perp3


def lift(points, xy, win=2):
    H, W = points.shape[:2]
    x, y = int(round(xy[0])), int(round(xy[1]))
    sl = points[max(0, y - win):min(H, y + win + 1), max(0, x - win):min(W, x + win + 1)].reshape(-1, 3)
    return np.median(sl[np.isfinite(sl).all(1)], axis=0)


def process_frame(pts, f, cfg, img_w, img_h):
    """pts: (H',W',3) metric point map (any resolution). Returns dict of results."""
    Lp, Ls = pallet_sides(cfg)
    sx, sy = pts.shape[1] / img_w, pts.shape[0] / img_h
    sc = lambda arr: np.array(arr, float) * [sx, sy]
    M, T = sc(f["M"]), sc(f["T"])
    S = sc(f["S"]) if f.get("S") else None
    n = len(M) - 1
    La = n * Lp / 1000.0
    Dn = cfg["pallets_deep"] * Ls / 1000.0

    r = analyze(pts, M, T, S, f.get("side_on", cfg["side_on"]))
    s = La / r["row_len"]
    res = {
        "pallet_ratio": r["row_len"] / La,
        "seg_ratio": r["seg_lens"][0] / r["seg_lens"][-1],
        "load_len_m": r["load_len"] * s, "height_m": r["height"] * s,
        "V_raw": r["load_len"] * Dn * r["height"],
        "V_norm": (r["load_len"] * s) * Dn * (r["height"] * s),
        "plane_rms_m": r["plane_rms_main"],
    }
    if S is not None:
        res["depth_over_length"] = r["depth_over_length"]
        res["implied_deep"] = r["depth_over_length"] * n * Lp / Ls
        res["V_norm_measdepth"] = (r["load_len"] * s) * (r["end_len"] * s) * (r["height"] * s)
    # old naive corner lift, for comparison only
    try:
        X = [lift(pts, p) for p in M]
        res["pallet_ratio_naive"] = float(np.linalg.norm(X[-1] - X[0]) / La)
    except Exception:
        res["pallet_ratio_naive"] = float("nan")
    return res


def main():
    import torch
    from moge.model.v2 import MoGeModel
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution_level", type=int, default=9, help="0-9, lower = faster on CPU")
    a = ap.parse_args()
    ann = load_annotations(cfg, selected=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()

    rows, keys = [], ["pallet_ratio", "seg_ratio", "V_raw", "V_norm", "depth_over_length",
                      "implied_deep", "V_norm_measdepth", "plane_rms_m", "pallet_ratio_naive"]
    for name, f in ann.items():
        img = cv2.cvtColor(cv2.imread(str(find_image(cfg, name))), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        t = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(t, use_fp16=(dev.type == "cuda"), resolution_level=a.resolution_level)
        pts = out["points"].cpu().numpy()
        try:
            r = process_frame(pts, f, cfg, w, h)
        except ValueError as e:
            print(name, "FAILED:", e); continue
        rows.append([name] + [round(float(r[k]), 3) if k in r else "" for k in keys])
        print(f"\n{name}")
        for k in keys:
            if k in r:
                print(f"   {k:20s} {float(r[k]):8.3f}")
    p = ROOT / cfg["results_dir"] / "depth_check.csv"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["frame"] + keys)
        w_.writerows(rows)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
