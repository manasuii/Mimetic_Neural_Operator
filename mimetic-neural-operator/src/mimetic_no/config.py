"""Paths, seeds and dataset specs.

Nothing here changes the science — it only centralises the constants that were
hard-coded inline in the notebook so every stage script reads the same values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# Paths (repo-root relative; all generated dirs are git-ignored)
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_NPZ = DATA_DIR / "dataset.npz"
TRAJ_PT = DATA_DIR / "trajectories.pt"


# --------------------------------------------------------------------------
# Seeding — notebook header: torch.manual_seed(0); np.random.seed(0)
# --------------------------------------------------------------------------
GLOBAL_SEED = 0


def set_seeds(seed: int = GLOBAL_SEED) -> None:
    """Reproduce the notebook's global seeding."""
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Stage 1 demo graph
# --------------------------------------------------------------------------
DEMO_GRAPH_SEED = 1

# --------------------------------------------------------------------------
# Stage 2 unit test
# --------------------------------------------------------------------------
UNIT_TEST = dict(seed=3, T=400, dt=0.05)
UNIT_TEST_TOL = 1e-9

# --------------------------------------------------------------------------
# Stage 4 dataset specs
# --------------------------------------------------------------------------
TRAIN_SPEC = dict(n_traj=12, n_nodes=15, T=110, seed_base=0)
ID_TEST_SPEC = dict(n_traj=6, n_nodes=15, T=120, seed_base=500)
OOD_TEST_SPEC = dict(n_traj=6, n_nodes=22, T=120, seed_base=900, k_low=1.5, k_high=2.5)
SWEEP_SPEC = dict(n_traj=10, n_nodes=15, T=100, seed_base=0)

# --------------------------------------------------------------------------
# Stage 3 training
# --------------------------------------------------------------------------
EPOCHS = 8
K = 4
LAM_SOFT = 1.0

# --------------------------------------------------------------------------
# Stage 5 evaluation
# --------------------------------------------------------------------------
EVAL_GRAPH = dict(n_nodes=15, edge_prob=0.25, seed=1234)
EVAL_T = 500
PLAUSIBILITY_T = 900
SPLIT_T = 400

# --------------------------------------------------------------------------
# Stage 5+ robustness
# --------------------------------------------------------------------------
LAM_GRID = [0.0, 0.1, 1.0, 10.0, 100.0]
K_GRID = [1, 2, 4, 8]
SCALE_N = [30, 60, 100]

# --------------------------------------------------------------------------
# Stage 6 topologies
# --------------------------------------------------------------------------
TOPO_T = 450
N_INST = 5

MODEL_NAMES = ["Mimetic (hard)", "Soft (ablation)", "Black-box GNN"]

CKPT_FILES = {
    "Mimetic (hard)": CKPT_DIR / "mimetic.pt",
    "Soft (ablation)": CKPT_DIR / "soft.pt",
    "Black-box GNN": CKPT_DIR / "blackbox.pt",
}


def figure(name: str) -> str:
    """Absolute path for a figure file, as a string for plt.savefig."""
    return str(FIGURES_DIR / name)
