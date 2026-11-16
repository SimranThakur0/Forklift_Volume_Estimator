"""STEP 3 - click the points (this replaces 'guessing' with measured pixels).

    python scripts/03_annotate.py [--redo]
    python scripts/03_annotate.py --src data/frames --t0 90 --t1 150 --min_sharp 120 --step 4
      (annotate straight from the extracted frames; thin them out first with the filters)

Click order per frame (n = pallets_along in config.yaml):
  1..n+1  bottom edge of the MAIN (big) face, LEFT to RIGHT:
          left end, every joint between neighbouring pallets, right end
          (frame 6: 3 clicks = 2 pallets). Click at the top of the pallet deck.
  next 2  top corners of the load above the left end and the right end
  then    press  2  to also click the END (narrow) face: outer-bottom, outer-top
          (the end furthest from the main face), or press Enter for ONE face only.
Keys: u=undo  Enter=accept  s=skip frame  q=save & quit  2=add end face
Terminal then asks: crates along main face, crate layers, which side the end face is on.

"""
import _path  # noqa
import argparse
import cv2
from fv.common import (load_config, load_annotations, save_annotations, list_images,
                       filter_frames, ROOT)


def ask_int(prompt):
    s = input(prompt).strip()
    return int(s) if s.isdigit() else None


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--src", default=None, help="folder to annotate (default: selected_dir); use data/frames")
    ap.add_argument("--t0", type=float, default=None, help="only frames at/after this second")
    ap.add_argument("--t1", type=float, default=None, help="only frames at/before this second")
    ap.add_argument("--min_sharp", type=int, default=None, help="skip blurry frames (score in file name)")
    ap.add_argument("--step", type=int, default=1, help="take every Nth remaining frame")
    a = ap.parse_args()
    try:
        ann = load_annotations(cfg)
    except SystemExit:
        ann = {}
    n = int(cfg["pallets_along"])
    base = [f"M{i}" for i in range(n + 1)] + ["T-left", "T-right"]
    cols_base = [(0, 255, 0)] * (n + 1) + [(0, 140, 255)] * 2

    src = ROOT / (a.src or cfg["selected_dir"])
    todo = filter_frames(list_images(src), a.t0, a.t1, a.min_sharp, a.step)
    print(f"{len(todo)} frames to look at in {src} "
          f"({sum(p.name in ann for p in todo)} already annotated)")
    for path in todo:
        if path.name in ann and not a.redo:
            continue
        img = cv2.imread(str(path))
        W = img.shape[1]
        scale = min(1.0, 1400 / W)
        labels, cols, pts = list(base), list(cols_base), []

        def redraw():
            v = cv2.resize(img, None, fx=scale, fy=scale)
            P = [(int(x * scale), int(y * scale)) for x, y in pts]
            for i, p in enumerate(P):
                cv2.circle(v, p, 5, cols[i], -1)
                cv2.putText(v, labels[i], (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cols[i], 2)
            for i in range(min(len(P), n + 1) - 1):
                cv2.line(v, P[i], P[i + 1], (0, 255, 0), 1)
            if len(P) >= n + 3:
                cv2.line(v, P[n + 1], P[n + 2], (0, 140, 255), 1)
            if len(P) == len(labels):
                if len(labels) == n + 3:
                    msg = "Enter=accept (one face)  2=add end face  u=undo  s=skip"
                else:
                    msg = "Enter=accept  u=undo  s=skip"
            else:
                msg = f"{path.name}: click {labels[len(pts)]}"
            cv2.rectangle(v, (0, 0), (v.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(v, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("annotate", v)

        def on_mouse(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(labels):
                pts.append((x / scale, y / scale))
                redraw()

        cv2.namedWindow("annotate")
        cv2.setMouseCallback("annotate", on_mouse)
        redraw()
        action = None
        while action is None:
            k = cv2.waitKey(50) & 0xFF
            if k == ord("u") and pts:
                pts.pop(); redraw()
            elif k == ord("s"):
                action = "skip"
            elif k == ord("q"):
                action = "quit"
            elif k == ord("2") and len(pts) == len(labels) == n + 3:
                labels += ["S-out-bottom", "S-out-top"]
                cols += [(255, 255, 0)] * 2
                redraw()
            elif k in (13, 10) and len(pts) == len(labels):
                action = "ok"
        if action == "ok":
            two = len(labels) == n + 5
            side_on = None
            if two:
                s = input(f"[{path.name}] end face on the (l)eft or (r)ight of the main face? ").strip().lower()
                side_on = "right" if s.startswith("r") else "left"
            n_across = ask_int(f"[{path.name}] crates along the main face (Enter=skip): ")
            n_layers = ask_int(f"[{path.name}] crate layers (Enter=skip): ")
            ann[path.name] = {"M": pts[:n + 1], "T": pts[n + 1:n + 3],
                              "S": pts[n + 3:n + 5] if two else None,
                              "side_on": side_on or cfg["side_on"],
                              "n_across": n_across, "n_layers": n_layers}
            save_annotations(cfg, ann)
            print("saved", path.name)
        elif action == "quit":
            break
    cv2.destroyAllWindows()
    save_annotations(cfg, ann)
    print(f"{len(ann)} annotated frames -> {cfg['annotations']}")


if __name__ == "__main__":
    main()
