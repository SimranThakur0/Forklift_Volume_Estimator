"""STEP 4 (optional, GPU preferred) - compare your clicks with a SAM 2 mask of the load.

    pip install torch torchvision git+https://github.com/facebookresearch/sam2.git
    python scripts/04_sam_check.py

Box prompt = bounding box of your clicks on the main face. Reports how far the mask's
top edge / left / right extent are from your clicked T-left / T-right. > ~10 px means a
click is off, or film / glare confuses the edge. Overlays go to results/sam/.
Untested here (needs model download). If SAM misbehaves, that is a failure case for the pitch.
"""
import _path  # noqa
import csv
import cv2, numpy as np
from fv.common import load_config, load_annotations, find_image, ROOT


def main():
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg = load_config()
    ann = load_annotations(cfg, selected=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pred = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=dev)
    out = ROOT / cfg["results_dir"] / "sam"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, f in ann.items():
        img = cv2.imread(str(find_image(cfg, name)))
        M, T = np.array(f["M"]), np.array(f["T"])
        box = np.array([M[:, 0].min() - 10, T[:, 1].min() - 10, M[:, 0].max() + 10, M[:, 1].max()], float)
        pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        masks, scores, _ = pred.predict(box=box, multimask_output=False)
        m = masks[0].astype(bool)
        ys, xs = np.where(m)
        rows.append([name, round(float(scores[0]), 3), round(ys.min() - T[:, 1].mean(), 1),
                     round(xs.min() - T[0, 0], 1), round(xs.max() - T[1, 0], 1)])
        ov = img.copy()
        ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.imwrite(str(out / (name.rsplit(".", 1)[0] + ".jpg")), ov)
    p = ROOT / cfg["results_dir"] / "sam_check.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "sam_score", "top_edge_dy_px", "left_dx_px", "right_dx_px"])
        w.writerows(rows)
    print(p.read_text())


if __name__ == "__main__":
    main()
