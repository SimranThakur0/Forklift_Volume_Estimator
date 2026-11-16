"""STEP 2 - use YOUR OWN screenshots (skip step 1 entirely if you already have them).

    python scripts/02_use_my_frames.py --src "C:/Users/me/Pictures/forklift"

Copies every png/jpg from --src into data/selected/ and prints their sizes.
"""
import _path  # noqa
import argparse, shutil
import cv2
from fv.common import load_config, list_images, ROOT


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    dst = ROOT / cfg["selected_dir"]
    dst.mkdir(parents=True, exist_ok=True)
    imgs = list_images(a.src)
    if not imgs:
        raise SystemExit(f"No images found in {a.src}")
    for p in imgs:
        shutil.copy(p, dst / p.name.replace(" ", "_"))
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        warn = "  <-- low resolution, clicks will be less precise" if w < 900 else ""
        print(f"{p.name}: {w}x{h}{warn}")
    print(f"copied {len(imgs)} images to {dst}")


if __name__ == "__main__":
    main()
