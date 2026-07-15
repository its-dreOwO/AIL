"""Generate HUMOF's hik_preprocessed/H25F50/hik_{A..D}/tus.pkl directly from raw HIK.

Why this exists: HUMOF's own datasets/dataset_preprocess/hik/preprocess.py is
broken — it windows the recording and then asserts len(tus)==1, which can never
hold for a windowed sequence. It also depends on the SAST repo. But DatasetHik
only ever reads tus.pkl, and that file is just the raw dense arrays rearranged:

    tus.pkl = (subseq, kid, present)
        subseq  : (P, n_frames, 29, 3) float32   <- Scene.poses3d transposed
        kid     : 'A'|'B'|'C'|'D'
        present : (P, n_frames) bool             <- Scene.masks transposed

so we can bypass SAST entirely and build it from hik.data.scene.Scene.

(This script previously existed only on the GCP VM and was lost when the VM was
deleted. It lives in the repo now.)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def build(dataset_root: str, out_dir: str, kids: str = "ABCD"):
    from hik.data import PersonSequences
    from hik.data.kitchen import Kitchen
    from hik.data.scene import Scene

    dataset_root = Path(dataset_root)
    person_path = str(dataset_root / "poses")
    scene_path = str(dataset_root / "scenes")
    smplx_path = str(dataset_root / "body_models")

    # Load once and reuse across recordings — PersonSequences reads every .npz
    # for all four datasets, so rebuilding it per kid would quadruple the work.
    print(f"loading PersonSequences from {person_path} ...", flush=True)
    person_seqs = PersonSequences(person_path=person_path)
    print("PersonSequences loaded", flush=True)

    out_dir = Path(out_dir)
    for kid in kids:
        print(f"\n=== {kid} ===", flush=True)
        kitchen = Kitchen.load_for_dataset(dataset=kid, data_location=scene_path)
        scene = Scene(
            dataset=kid,
            person_seqs=person_seqs,
            kitchen=kitchen,
            smplx_path=smplx_path,
        )

        # (n_frames, P, 29, 3) -> (P, n_frames, 29, 3)
        subseq = np.ascontiguousarray(
            scene.poses3d.transpose(1, 0, 2, 3).astype(np.float32)
        )
        # (n_frames, P) -> (P, n_frames); masks are 0.0/1.0 floats
        present = np.ascontiguousarray(scene.masks.transpose(1, 0)) > 0.5

        assert subseq.shape[:2] == present.shape[:2], (subseq.shape, present.shape)
        assert subseq.shape[2:] == (29, 3), subseq.shape
        print(f"{subseq.shape=} {present.shape=} present_frac={present.mean():.4f}",
              flush=True)

        d = out_dir / f"hik_{kid}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "tus.pkl", "wb") as f:
            pickle.dump((subseq, kid, present), f, protocol=4)
        print(f"wrote {d / 'tus.pkl'}", flush=True)

        del scene, subseq, present, kitchen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                    help="dir containing poses/ scenes/ body_models/")
    ap.add_argument("--out-dir", required=True,
                    help="hik_preprocessed/H25F50")
    ap.add_argument("--kids", default="ABCD")
    a = ap.parse_args()
    build(a.dataset_root, a.out_dir, a.kids)
