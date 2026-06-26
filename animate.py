"""
Animate a Humans-in-Kitchens scene: render the kitchen + everyone's moving
skeletons over a time window, with a fading floor-trail per person so you can
see who walks from where to where.

Usage:
  source .venv/bin/activate
  python animate.py [kitchen] [start_frame] [n_frames] [step]

Defaults: kitchen A, a ~1 min (1500-frame) auto-picked busy window, every 5th
frame. Output: anim_<kitchen>_<start>.mp4

Env: HIK_DATA overrides the dataset path.
"""
import os
import sys
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from hik.data.kitchen import Kitchen
from hik.data import PersonSequences
from hik.vis import plot_pose, get_lims

DATA = os.environ.get("HIK_DATA", "/home/dre/Downloads/hik_dataset/data")
POSES = os.path.join(DATA, "poses")
SCENES = os.path.join(DATA, "scenes")

TRAIL_LEN = 250  # frames of history shown in the trail (~10s @25Hz)


def pick_busy_window(ps, kitchen, last, n_frames):
    """Scan coarsely and return the start frame of the busiest n_frames window."""
    step = 250
    samples = [(f, len(ps.get_frame(kitchen, f)))
               for f in range(0, max(last - n_frames, 1), step)]
    best_start, best_score = 0, -1
    for start, _ in samples:
        score = sum(c for f, c in samples if start <= f < start + n_frames)
        if score > best_score:
            best_score, best_start = score, start
    return best_start


def person_color(pid):
    return cm.tab20(pid % 20)


def main():
    kitchen = sys.argv[1] if len(sys.argv) > 1 else "A"
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    step = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    print(f"[*] Loading kitchen {kitchen} + poses ...")
    k = Kitchen.load_for_dataset(dataset=kitchen, data_location=SCENES)
    ps = PersonSequences(person_path=POSES)
    last = k.last_frame

    if len(sys.argv) > 2:
        start = int(sys.argv[2])
    else:
        print("[*] Auto-picking a busy window ...")
        start = pick_busy_window(ps, kitchen, last, n_frames)
    end = min(start + n_frames, last)
    frames = list(range(start, end, step))
    print(f"[i] kitchen {kitchen}, frames {start}..{end} step {step} "
          f"-> {len(frames)} rendered frames")

    # Stable camera: global limits from every pose across the window.
    all_pts = []
    for f in frames:
        for p in ps.get_frame(kitchen, f):
            all_pts.append(p["pose3d"])
    if not all_pts:
        print("[!] No people in this window. Try another start frame.")
        sys.exit(1)
    xlim, ylim, _ = get_lims(np.concatenate(all_pts, axis=0)[None])
    zlim = [0.0, 2.0]  # real human height range, so skeletons aren't crushed
    dx, dy, dz = xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]

    # Per-person rolling trail of root (x, y) positions.
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN // step))

    tmp = tempfile.mkdtemp(prefix="hik_anim_")
    fig = plt.figure(figsize=(10, 10))
    try:
        for i, f in enumerate(frames):
            ax = fig.add_subplot(111, projection="3d")
            k.plot(ax, f)
            people = ps.get_frame(kitchen, f)
            for p in people:
                col = person_color(int(p["pid"]))
                plot_pose(ax, p["pose3d"], lcolor=col, rcolor=col, mcolor=col,
                          linewidth=1.5, alpha=0.95)
                root = p["transforms"][:3]
                trails[int(p["pid"])].append((root[0], root[1]))
            # draw trails on the floor
            for pid, pts in trails.items():
                if len(pts) < 2:
                    continue
                arr = np.array(pts)
                ax.plot(arr[:, 0], arr[:, 1], np.zeros(len(arr)) + 0.02,
                        color=person_color(pid), linewidth=2.5, alpha=0.7)
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
            ax.set_box_aspect((dx, dy, dz))
            ax.set_title(f"Kitchen {kitchen}  frame {f}  ({len(people)} people)")
            ax.axis("off")
            ax.view_init(elev=32, azim=-60)
            fig.savefig(os.path.join(tmp, f"{i:05d}.png"), dpi=90)
            fig.clf()
            if i % 20 == 0:
                print(f"    {i+1}/{len(frames)} frames rendered")

        out = f"anim_{kitchen}_{start}.mp4"
        fps = 15
        print(f"[*] Encoding {out} @ {fps}fps ...")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps), "-i",
            os.path.join(tmp, "%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out,
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[OK] wrote {out}  ({len(frames)} frames, "
              f"{len(frames)/fps:.1f}s video)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
