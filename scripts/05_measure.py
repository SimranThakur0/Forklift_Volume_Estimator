"""STEP 5 - per-frame measurement (pallet ruler, two-face solve, crate cross-check).

    python scripts/05_measure.py [--deep 3]

Columns: faces (1/2), yaw (deg; 2-face = solved, 1-face = assumed 0),
persp (first pallet px / last pallet px; 1.0 = no perspective, >1 = left end nearer),
L/D/H in mm, V_pix, V_crate, crates per pallet.


"""
import _path  # noqa
import argparse, csv
import cv2, numpy as np
from fv.common import load_config, load_annotations, find_image, ROOT
from fv.geometry import point_estimate, crate_estimate, passes_qc


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", type=int, default=None, help="override pallets_deep")
    ap.add_argument("--selected", action="store_true", help="only QC-selected frames")
    a = ap.parse_args()
    ann = load_annotations(cfg, selected=a.selected)
    out = ROOT / cfg["results_dir"]
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'frame':26s} f  yaw  persp  {'L':>5s} {'D':>5s} {'H':>5s}  V_pix V_crate cr/pal  QC")
    for name, f in ann.items():
        m, deep = point_estimate(f, cfg, a.deep)
        c = crate_estimate(f.get("n_across"), f.get("n_layers"), cfg, deep)
        faces = 2 if f.get("S") else 1
        vc = c["V_m3"] if c else float("nan")
        cpp = c["crates_per_pallet"] if c else float("nan")
        ok, why = passes_qc(m, cfg)
        rows.append([name, faces, m["yaw_deg"], m["persp_ratio"], m["L_mm"], m["D_mm"],
                     m["H_mm"], m["V_m3"], vc, cpp, "ok" if ok else "DROP: " + why])
        print(f"{name[:26]:26s} {faces}  {m['yaw_deg']:3.0f}  {m['persp_ratio']:5.2f}  "
              f"{m['L_mm']:5.0f} {m['D_mm']:5.0f} {m['H_mm']:5.0f}  {m['V_m3']:5.2f} {vc:7.2f} {cpp:6.2f}  {'ok' if ok else 'DROP: ' + why}")

        p = find_image(cfg, name)
        img = cv2.imread(str(p)) if p else None
        if img is not None:
            M = np.array(f["M"]).astype(int); T = np.array(f["T"]).astype(int)
            for i in range(len(M) - 1):
                cv2.line(img, tuple(M[i]), tuple(M[i + 1]), (0, 255, 0), 3)
            for q in M:
                cv2.circle(img, tuple(q), 6, (0, 255, 0), -1)
            cv2.line(img, tuple(T[0]), tuple(T[1]), (0, 140, 255), 3)
            cv2.line(img, tuple(M[0]), tuple(T[0]), (255, 255, 255), 2)
            cv2.line(img, tuple(M[-1]), tuple(T[1]), (255, 255, 255), 2)
            if f.get("S"):
                S = np.array(f["S"]).astype(int)
                sb = M[0] if f.get("side_on", cfg["side_on"]) == "left" else M[-1]
                st = T[0] if f.get("side_on", cfg["side_on"]) == "left" else T[1]
                cv2.line(img, tuple(sb), tuple(S[0]), (255, 255, 0), 3)
                cv2.line(img, tuple(st), tuple(S[1]), (255, 255, 0), 3)
                cv2.line(img, tuple(S[0]), tuple(S[1]), (255, 255, 255), 2)
            txt = f"{faces}-face  yaw={m['yaw_deg']:.0f}  L={m['L_mm']:.0f} D={m['D_mm']:.0f} H={m['H_mm']:.0f} mm  V={m['V_m3']:.1f} m3"
            cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.imwrite(str(out / "overlays" / (name.rsplit(".", 1)[0] + ".jpg")), img)

    with open(out / "point_estimates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "faces", "yaw_deg", "persp_ratio", "L_mm", "D_mm", "H_mm",
                    "V_pixel_m3", "V_crate_m3", "crates_per_pallet", "qc"])
        w.writerows(rows)
    V = np.array([r[7] for r in rows])
    good = np.array([r[7] for r in rows if r[10] == "ok"])
    print(f"\nall frames : median {np.median(V):.2f} m3 (range {V.min():.2f}-{V.max():.2f}, n={len(V)})")
    if len(good):
        print(f"QC-passing : median {np.median(good):.2f} m3 (range {good.min():.2f}-{good.max():.2f}, n={len(good)})  <-- use this")
    else:
        print("No frame passes QC. Re-take frames with a smaller turn angle or check your joint clicks.")
    print(f"pallets: {cfg['pallets_along']} along x {a.deep or cfg['pallets_deep']} deep")


if __name__ == "__main__":
    main()
