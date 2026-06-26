"""
Runnable examples for the Humans-in-Kitchens (hik) dataset API.

Usage:
  source .venv/bin/activate
  python examples.py <command> [args]

Commands:
  stats                       per-kitchen sequence/frame counts
  activities                  list all 82 activity labels
  find  <kitchen> <activity>  frames where someone does an activity (e.g. find A drink)
  scene <kitchen> [frame]     objects present in the scene at a frame
  pose  <kitchen> [frame]     pose/SMPL detail for people at a frame
  render <kitchen> [frame]    render scene+people to render_<kitchen>_<frame>.png

If [frame] is omitted or out of range, a valid busy frame is auto-picked.
Valid scene frame ranges: A 0-129722, B 0-177638, C 0-175556, D 0-177636.

Default data dir: /home/dre/Downloads/hik_dataset/data  (override with $HIK_DATA)
"""
import os
import sys
import numpy as np

DATA = os.environ.get("HIK_DATA", "/home/dre/Downloads/hik_dataset/data")
POSES = os.path.join(DATA, "poses")
SCENES = os.path.join(DATA, "scenes")

LABELS = {1: "WHITEBOARD", 2: "MICROWAVE", 3: "KETTLE", 4: "COFFEE_MACHINE",
          5: "TABLE", 6: "SITTABLE", 7: "CUPBOARD", 8: "OCCLUDER",
          9: "DISHWASHER", 10: "DRAWER", 11: "SINK", 12: "TRASH", 13: "OUT_OF_BOUND"}


def cmd_stats():
    from hik.data import PersonSequences
    ps = PersonSequences(person_path=POSES)
    for d in ["A", "B", "C", "D"]:
        seqs = ps.get_sequences(d)
        frames = sum(len(s) for s in seqs)
        print(f"Kitchen {d}: {len(seqs):3d} sequences, {frames:>9,} frames "
              f"({frames/25/60:6.1f} min @25Hz)")


def cmd_activities():
    from hik.data.constants import activity2index
    names = sorted(activity2index)
    print(f"{len(names)} activities:")
    for i, n in enumerate(names):
        print(f"  {n}", end="\n" if i % 3 == 2 else "")
    print()


def cmd_find(kitchen, activity):
    from hik.data import PersonSequences
    ps = PersonSequences(person_path=POSES)
    total = 0
    for s in ps.get_sequences(kitchen):
        fr = s.get_frames_where_action([activity])
        if len(fr):
            total += len(fr)
            print(f"  person {s.pid:2d}: {len(fr):5d} frames, first @ {int(fr[0])}")
    print(f"Kitchen {kitchen}: '{activity}' happens in {total} person-frames total")


def _valid_frame(kitchen, frame=None):
    """Pick a busy, in-range frame if none/out-of-range given."""
    from hik.data.kitchen import Kitchen
    from hik.data import PersonSequences
    k = Kitchen.load_for_dataset(dataset=kitchen, data_location=SCENES)
    last = k.last_frame
    if frame is not None and 0 <= int(frame) <= last:
        return int(frame)
    ps = PersonSequences(person_path=POSES)
    best = (0, last // 2)
    for f in range(0, last, 1000):
        c = len(ps.get_frame(kitchen, f))
        if c > best[0]:
            best = (c, f)
    print(f"[i] auto-picked frame {best[1]} (busiest, {best[0]} people; "
          f"valid range 0..{last})")
    return best[1]


def cmd_scene(kitchen, frame=None):
    from hik.data.kitchen import Kitchen
    frame = _valid_frame(kitchen, frame)
    k = Kitchen.load_for_dataset(dataset=kitchen, data_location=SCENES)
    objs = k.get_environment(frame=int(frame))
    print(f"Kitchen {kitchen}, frame {frame}: {len(objs)} objects")
    counts = {}
    for o in objs:
        lid = int(np.argmax(o.label)) + 1
        counts[LABELS.get(lid, lid)] = counts.get(LABELS.get(lid, lid), 0) + 1
    for name, c in sorted(counts.items()):
        print(f"  {c:2d}x {name}")


def cmd_pose(kitchen, frame=None):
    from hik.data import PersonSequences
    from hik.data.constants import activity2index
    frame = _valid_frame(kitchen, frame)
    idx2act = {v: k for k, v in activity2index.items()}
    ps = PersonSequences(person_path=POSES)
    people = ps.get_frame(kitchen, int(frame))
    print(f"Kitchen {kitchen}, frame {frame}: {len(people)} people")
    for p in people:
        acts = [idx2act[i] for i in np.nonzero(p["act"] > 0.5)[0]]
        print(f"  person {p['pid']:2d}: pose3d {p['pose3d'].shape}, "
              f"smpl {p['smpl'].shape}, doing={acts or ['(none)']}")


def cmd_render(kitchen, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pylab as plt
    from hik.data.kitchen import Kitchen
    from hik.data import PersonSequences
    from hik.vis import plot_pose
    frame = _valid_frame(kitchen, frame)
    ps = PersonSequences(person_path=POSES)
    k = Kitchen.load_for_dataset(dataset=kitchen, data_location=SCENES)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    k.plot(ax, frame)
    people = ps.get_frame(kitchen, frame)
    for p in people:
        plot_pose(ax, p["pose3d"], linewidth=1)
    ax.axis("off")
    out = f"render_{kitchen}_{frame}.png"
    fig.savefig(out, dpi=100)
    print(f"Rendered {len(people)} people -> {out}")


if __name__ == "__main__":
    cmds = {"stats": cmd_stats, "activities": cmd_activities, "find": cmd_find,
            "scene": cmd_scene, "pose": cmd_pose, "render": cmd_render}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(0)
    cmds[sys.argv[1]](*sys.argv[2:])
