"""Shared helpers: config, annotations, paths."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_config(path="config.yaml"):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def list_images(folder):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMG_EXT)


def load_annotations(cfg, selected=False):
    """selected=True -> use annotations_selected.json (QC-passing frames) if it exists."""
    p = ROOT / cfg["annotations"]
    if selected:
        ps = ROOT / cfg.get("annotations_selected", "data/annotations_selected.json")
        if ps.exists():
            p = ps
    if not p.exists():
        raise SystemExit(f"No annotations at {p}. Run scripts/03_annotate.py first.")
    with open(p) as f:
        return json.load(f)


def save_annotations(cfg, ann):
    p = ROOT / cfg["annotations"]
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(ann, f, indent=2)


def pallet_sides(cfg):
    """(Lp, Ls): pallet side running ALONG the main face, and ALONG the end face (mm)."""
    p = cfg["pallet"]
    if cfg["visible_side"] == "long":
        return float(p["long_mm"]), float(p["short_mm"])
    return float(p["short_mm"]), float(p["long_mm"])


def find_image(cfg, name):
    for d in (cfg["selected_dir"], cfg["frames_dir"]):
        p = ROOT / d / name
        if p.exists():
            return p
    return None


_FRAME_RE = None


def filter_frames(imgs, t0=None, t1=None, min_sharp=None, step=1):
    """Thin out step-1 frames (names like f000474_t0018.96_s80.jpg) before annotating.
    Files that do not follow that pattern (your own screenshots) are never filtered out."""
    import re
    rx = re.compile(r"_t([\d.]+)_s(\d+)")
    out = []
    for p in imgs:
        m = rx.search(p.name)
        if m:
            t, sh = float(m.group(1)), int(m.group(2))
            if (t0 is not None and t < t0) or (t1 is not None and t > t1):
                continue
            if min_sharp is not None and sh < min_sharp:
                continue
        out.append(p)
    return out[::max(1, int(step))]
