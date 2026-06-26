"""
Smoke test for the hik setup.

Usage:
  # 1) Import-only check (no dataset needed):
  python smoke_test.py

  # 2) Full check once the dataset is downloaded/extracted:
  python smoke_test.py /path/to/dataset/data
"""
import sys


def check_imports():
    print("[*] Checking imports...")
    import numpy  # noqa
    import torch  # noqa
    import smplx  # noqa
    import matplotlib  # noqa
    from hik.data import PersonSequences  # noqa
    from hik.data.kitchen import Kitchen  # noqa
    from hik.data.constants import activity2index  # noqa
    from hik.vis import plot_pose  # noqa
    print("    OK - all core modules import cleanly")
    print(f"    torch {torch.__version__}, numpy {numpy.__version__}")


def check_dataset(data_dir):
    import os
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pylab as plt
    from hik.data.kitchen import Kitchen
    from hik.data import PersonSequences
    from hik.vis import plot_pose

    scenes = os.path.join(data_dir, "scenes")
    poses = os.path.join(data_dir, "poses")
    print(f"[*] Loading poses from {poses} ...")
    person_seqs = PersonSequences(person_path=poses)

    dataset = "A"
    print(f"[*] Loading kitchen geometry for dataset {dataset} ...")
    kitchen = Kitchen.load_for_dataset(dataset=dataset, data_location=scenes)

    seqs = person_seqs.get_sequences(dataset)
    print(f"    dataset {dataset}: {len(seqs)} person-sequences")
    seq = seqs[0]
    frame = int(seq.frames[len(seq) // 2])
    print(f"    rendering frame {frame} ...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    kitchen.plot(ax, frame)
    people = person_seqs.get_frame(dataset, frame)
    for person in people:
        plot_pose(ax, person["pose3d"], linewidth=1)
    ax.axis("off")
    out = "smoke_render.png"
    fig.savefig(out, dpi=100)
    print(f"    OK - rendered {len(people)} people -> {out}")


if __name__ == "__main__":
    check_imports()
    if len(sys.argv) > 1:
        check_dataset(sys.argv[1])
    else:
        print("\n[i] No data dir given. Import check passed.")
        print("    Run with the data path once downloaded:")
        print("    python smoke_test.py /path/to/dataset/data")
