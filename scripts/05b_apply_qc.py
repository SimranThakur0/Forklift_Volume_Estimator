"""STEP 5b - apply the quality gate to your annotated frames and MOVE the good ones to data/selected.

    python scripts/05b_apply_qc.py                  # move passing frames
    python scripts/05b_apply_qc.py --copy           # copy instead of move
    python scripts/05b_apply_qc.py --max_crate_diff 0.25   # also drop frames where pixel and crate volume
                                                           # disagree by > 25% (needs crate counts)

What it does
  1. reads data/annotations.json (ALL your clicks - never modified)
  2. computes yaw / perspective for each frame and applies config.yaml qc: thresholds
  3. moves/copies passing images from data/frames into data/selected/
  4. writes data/annotations_selected.json (passing frames only). Steps 04, 06, 07 use it automatically;
     05 uses it with --selected
  5. writes results/qc_report.csv with the reason for every dropped frame
Failing frames stay in data/frames; re-annotate them with  03_annotate.py --redo  if you suspect a bad click.
"""
import _path  # noqa
import argparse, csv, json, shutil
from fv.common import load_config, load_annotations, find_image, ROOT
from fv.geometry import point_estimate, crate_estimate, passes_qc


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--max_crate_diff", type=float, default=None)
    a = ap.parse_args()

    ann = load_annotations(cfg)
    sel_dir = ROOT / cfg["selected_dir"]
    sel_dir.mkdir(parents=True, exist_ok=True)
    keep, rows = {}, []
    for name, f in ann.items():
        m, deep = point_estimate(f, cfg)
        ok, why = passes_qc(m, cfg)
        if ok and a.max_crate_diff is not None:
            c = crate_estimate(f.get("n_across"), f.get("n_layers"), cfg, deep)
            if c:
                d = abs(m["V_m3"] - c["V_m3"]) / c["V_m3"]
                if d > a.max_crate_diff:
                    ok, why = False, f"pixel vs crate volume differ {d:.0%}"
        rows.append([name, round(m["yaw_deg"], 1), round(m["persp_ratio"], 2),
                     round(m["V_m3"], 2), "keep" if ok else "drop", why])
        print(f"{'KEEP' if ok else 'drop'}  {name:34s} yaw={m['yaw_deg']:3.0f} persp={m['persp_ratio']:.2f} "
              f"V={m['V_m3']:5.2f} {('(' + why + ')') if why else ''}")
        if not ok:
            continue
        keep[name] = f
        src = find_image(cfg, name)
        dst = sel_dir / name
        if src is None:
            print("   (image not found, annotation kept anyway)")
        elif src.resolve() != dst.resolve():
            (shutil.copy2 if a.copy else shutil.move)(str(src), str(dst))

    with open(ROOT / cfg["annotations_selected"], "w") as fh:
        json.dump(keep, fh, indent=2)
    out = ROOT / cfg["results_dir"]
    out.mkdir(exist_ok=True)
    with open(out / "qc_report.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "yaw_deg", "persp_ratio", "V_pixel_m3", "decision", "reason"])
        w.writerows(rows)
    print(f"\n{len(keep)}/{len(ann)} frames kept -> {sel_dir} and {cfg['annotations_selected']}")
    if len(keep) < 3:
        print("Fewer than 3 usable frames: annotate more frames with a smaller turn angle.")


if __name__ == "__main__":
    main()
