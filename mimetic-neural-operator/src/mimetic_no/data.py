"""Stage 4 — trajectory generation, OOD split, dataset save/load.

`make_trajectories` and `make_nl` are verbatim from the notebook. The save/load
helpers are new plumbing only: the notebook kept everything in memory between
cells, whereas separate stage scripts need the data on disk. `save_dataset_npz`
writes exactly the `dataset.npz` the notebook wrote.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import config
from .dynamics import simulate_nl, simulate_true
from .graph import build_graph, edge_index, incidence_matrix


def make_trajectories(n_traj, n_nodes=15, edge_prob=0.25, T=120, dt=0.05,
                      seed_base=0, k_low=0.5, k_high=1.5):
    trajs = []
    for t in range(n_traj):
        s = seed_base + t
        n, e, k = build_graph(n_nodes, edge_prob, seed=s, k_low=k_low, k_high=k_high)
        B = incidence_matrix(n, e)
        i_idx, j_idx = edge_index(e)
        traj = simulate_true(n, e, B, k, T=T, dt=dt, E0=torch.rand(n), seed=s)
        trajs.append(dict(i_idx=i_idx, j_idx=j_idx, B=B, k=k, traj=traj,
                          seed=s, dt=dt, n_nodes=n, T=T, edges=e))
    return trajs


def make_nl(n_traj, seed_base):
    out = []
    for t in range(n_traj):
        s = seed_base + t
        n, e, k = build_graph(15, 0.25, seed=s)
        B = incidence_matrix(n, e)
        i_idx, j_idx = edge_index(e)
        tr = simulate_nl(n, e, B, k, 110, 0.05, torch.rand(n), s)
        out.append(dict(i_idx=i_idx, j_idx=j_idx, B=B, k=k, traj=tr, seed=s,
                        dt=0.05, n_nodes=n, T=110, edges=e))
    return out


# --------------------------------------------------------------------------
# Persistence (plumbing only — no change to the data itself)
# --------------------------------------------------------------------------
def save_dataset_npz(train_trajs, id_test, ood_test, path: Path | str = None):
    """Write the compact dataset artifact exactly as the notebook did."""
    path = Path(path) if path is not None else config.DATASET_NPZ
    # Save a compact dataset artifact (example trajectory + all test initial conditions/graphs)
    np.savez(str(path),
             example_traj=train_trajs[0]["traj"].numpy(),
             example_k=train_trajs[0]["k"].numpy(),
             id_test_E0=np.stack([t["traj"][0].numpy() for t in id_test]),
             ood_test_E0=np.stack([t["traj"][0].numpy() for t in ood_test], dtype=object),
             allow_pickle=True)
    print(f"saved {path}")
    return path


def save_trajectories(train_trajs, id_test, ood_test, path: Path | str = None):
    """Full tensors, so later stages can reload the identical splits."""
    path = Path(path) if path is not None else config.TRAJ_PT
    torch.save({"train": train_trajs, "id_test": id_test, "ood_test": ood_test}, str(path))
    print(f"saved {path}")
    return path


def load_trajectories(path: Path | str = None):
    path = Path(path) if path is not None else config.TRAJ_PT
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/stage4_generate_data.py` first."
        )
    blob = torch.load(str(path), weights_only=False)
    return blob["train"], blob["id_test"], blob["ood_test"]
