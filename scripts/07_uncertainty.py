"""STEP 7 - Monte Carlo uncertainty = the number and the 'how sure' for your pitch.

    python scripts/07_uncertainty.py                 # uses pallets_deep from config
    python scripts/07_uncertainty.py --deep 2 3      # compare scenarios side by side

Perturbs click positions, yaw (1-face frames only; 2-face frames solve it), pallet-size
tolerance and fill factor, spreading samples evenly over your frames.
A discrete unknown (how many pallets deep) is NOT averaged away: it is reported as
separate scenarios. State which one you believe and why.

"""
import _path  # noqa
import argparse, json
import numpy as np
from fv.common import load_config, load_annotations, ROOT
from fv.geometry import monte_carlo, point_estimate, passes_qc


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", type=int, nargs="+", default=[cfg["pallets_deep"]])
    ap.add_argument("--no_qc", action="store_true", help="keep frames that fail the quality gate")
    a = ap.parse_args()
    ann = load_annotations(cfg, selected=True)
    if not a.no_qc:
        kept = {}
        for n, f in ann.items():
            ok, why = passes_qc(point_estimate(f, cfg)[0], cfg)
            print(("keep " if ok else "DROP ") + n + ("" if ok else f"  ({why})"))
            if ok:
                kept[n] = f
        if not kept:
            raise SystemExit("No frame passes QC (see config.yaml qc:). Fix clicks / frames or use --no_qc.")
        ann = kept
    N = cfg["mc"]["n_samples"]
    per = max(1, N // len(ann))
    out = ROOT / cfg["results_dir"]
    out.mkdir(exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.5, 3.4))
    summary = {}
    for deep in a.deep:
        rng = np.random.default_rng(cfg["mc"]["seed"])
        V = np.concatenate([monte_carlo(f, cfg, per, rng, deep) for f in ann.values()])
        lo, med, hi = np.percentile(V, [5, 50, 95])
        summary[f"deep_{deep}"] = {"median_m3": round(float(med), 2), "p5_m3": round(float(lo), 2),
                                   "p95_m3": round(float(hi), 2), "frames": len(ann),
                                   "pallets_along": cfg["pallets_along"], "pallets_deep": deep}
        print(f"pallets deep = {deep}:  V = {med:.2f} m3  (90% interval {lo:.2f} - {hi:.2f})  [{len(ann)} frames]")
        plt.hist(V, bins=60, alpha=0.6, label=f"{deep} deep: {med:.1f} [{lo:.1f}-{hi:.1f}]")
    plt.xlabel("load volume (m3)"); plt.ylabel("samples"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "volume_hist.png", dpi=150)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
