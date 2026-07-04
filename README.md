# AIL — Multi-person 3D human motion forecasting on *Humans in Kitchens*

This repository develops **forecasting models** for the *Humans in Kitchens* (HIK)
benchmark — predicting future 3D human motion (multi-person, with scene context)
from past motion. It **vendors** the official HIK data API + evaluation harness and
adds a `forecasting/` package with our own models, training, and evaluation.

> **HIK itself ships no model** — it is the dataset loader, visualizer, and
> benchmark. This repo is where the models live.

## Credits — original *Humans in Kitchens* work

The vendored `hik/` package, `testdata/`, `documentation/`, and `notebooks/` are the
**official HIK API by Julian Tanke, Oh-Hun Kwon, Felix Mueller, Andreas Doering, and
Juergen Gall** (NeurIPS 2023 Datasets & Benchmarks Track). Original repository:
<https://github.com/jutanke/hik>. Used and redistributed here under its MIT license
(see [`LICENSE`](LICENSE)). All credit for the dataset and harness is theirs.

```bibtex
@article{tanke2023humanskitch,
  title={Humans in Kitchens: A Dataset for Multi-Person Human Motion Forecasting with Scene Context},
  author={Tanke, Julian and Kwon, Oh-Hun and Mueller, Felix and Doering, Andreas and Gall, Juergen},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Repository layout

```
hik/                 # vendored official HIK harness (data, eval, transforms, vis)
testdata/test.json   # official test split (vendored)
documentation/       # original HIK docs (vendored)
notebooks/           # original HIK notebooks (vendored)
forecasting/         # OUR models, training, evaluation  (WIP)
docs/superpowers/    # design specs / implementation plans for our work
setup.py             # installs the hik package (pip install -e .)
```

## Our work

The current roadmap (see [`docs/superpowers/specs/`](docs/superpowers/specs/)):

1. **Single-person siMLPe baseline** — implemented under
   [`forecasting/`](forecasting/) with DCT-MLP forecasting, canonical
   normalization, `Evaluator.execute3d` integration, and MPJPE scoring against a
   zero-velocity reference. See [`forecasting/README.md`](forecasting/README.md)
   for the runbook, baseline numbers, and GPU VM training steps.
2. Multi-person social attention.
3. Scene-aware conditioning (kitchen-object context).
4. Generative (diffusion / CVAE) head for the long horizon.

## Setup

```bash
pip install -e .          # installs the vendored hik package
pip install torch         # required by the SMPL-X body model and our models

# point at the dataset (path has spaces -> use the env var, don't hardcode)
export HIK_DATA="/mnt/elements/dataset AIL/Humans_in_Kitchen/Humans_in_Kitchen"

python smoke_test.py "$HIK_DATA"   # full check: loads data, renders one frame
```

See [`CLAUDE.md`](CLAUDE.md) for a detailed map of the harness internals.
