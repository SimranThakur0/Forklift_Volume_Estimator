"""STEP 1 - download the video, detect shot cuts, extract frames with sharpness score.

    python scripts/01_download_extract.py                 # defaults from config.yaml
    python scripts/01_download_extract.py --every 3 --start 20 --end 90

The video is an edited promo with many shots. shots.csv tells you where each
shot starts so you can find the 'forklift drives past with a load' shots fast.
"""
import _path  # noqa
import argparse, csv, subprocess, sys
import cv2
from fv.common import load_config, ROOT


def download(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("video already there:", out)
        return
    cmd = [sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080]/b",
           "--merge-output-format", "mp4", "-o", str(out), url]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (960, int(g.shape[0] * 960 / g.shape[1])))
    return int(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3, help="keep every Nth frame")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--cut_thresh", type=float, default=0.5,
                    help="histogram correlation below this = new shot (raise for more cuts)")
    a = ap.parse_args()

    video = ROOT / cfg["video_path"]
    if not a.skip_download:
        download(cfg["video_url"], video)
    if not video.exists():
        raise SystemExit(f"{video} missing. Download manually and put it there.")

    fdir = ROOT / cfg["frames_dir"]
    fdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{fps:.2f} fps, {total} frames, {total / fps:.1f} s")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))

    cuts, prev_hist, shot_start = [], None, a.start
    i = int(a.start * fps)
    saved = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        if a.end is not None and t > a.end:
            break
        hist = cv2.calcHist([cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)],
                            [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < a.cut_thresh:
            cuts.append((round(shot_start, 2), round(t, 2)))
            shot_start = t
        prev_hist = hist
        if i % a.every == 0:
            cv2.imwrite(str(fdir / f"f{i:06d}_t{t:07.2f}_s{sharpness(f)}.jpg"), f)
            saved += 1
        i += 1
    cuts.append((round(shot_start, 2), round(i / fps, 2)))

    with open(ROOT / cfg["results_dir"] / "shots.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_s", "end_s"])
        w.writerows(cuts)
    print(f"saved {saved} frames to {fdir}; {len(cuts)} shots -> results/shots.csv")


if __name__ == "__main__":
    main()
