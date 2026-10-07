"""Stage 3 — K-step rollout trainer (exposure-bias fix; `lam>0` = soft conservation
penalty), plus checkpoint helpers.

`train_model` is verbatim from the notebook. The save/load functions are new
plumbing so training and evaluation can live in separate scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config
from .dynamics import source_term
from .models import MODEL_CLASSES


def train_model(model, trajs, epochs=10, K=4, lr=1e-3, lam=0.0, stride=3, log_every=3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    for ep in range(epochs):
        np.random.shuffle(trajs)
        run = 0.0
        cnt = 0
        for tr in trajs:
            traj = tr["traj"]
            i_idx, j_idx, B, k = tr["i_idx"], tr["j_idx"], tr["B"], tr["k"]
            seed, dt, n, T = tr["seed"], tr["dt"], tr["n_nodes"], tr["T"]
            for t0 in range(0, T - K, stride):
                opt.zero_grad()
                E = traj[t0].clone()
                loss = 0.0
                pen = 0.0
                for step in range(K):
                    S = source_term((t0 + step) * dt, n, seed=seed)
                    dEdt, imb = model(E, i_idx, j_idx, B, S, k)
                    E = E + dt * dEdt
                    loss = loss + mse(E, traj[t0 + step + 1])
                    pen = pen + imb ** 2
                (loss / K + lam * pen / K).backward()
                opt.step()
                run += (loss / K).item()
                cnt += 1
        if ep % log_every == 0:
            print(f"  epoch {ep:2d}: rollout-MSE = {run/cnt:.3e}")
    return model


# --------------------------------------------------------------------------
# Checkpoints (plumbing only)
# --------------------------------------------------------------------------
def save_checkpoint(model, name: str, path: Path | str = None):
    path = Path(path) if path is not None else config.CKPT_FILES[name]
    torch.save(model.state_dict(), str(path))
    print(f"saved {path}")
    return path


def load_checkpoint(name: str, path: Path | str = None):
    """Rebuild the model class for `name` and load its trained weights."""
    path = Path(path) if path is not None else config.CKPT_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/stage3_train_models.py` first."
        )
    model = MODEL_CLASSES[name]()
    model.load_state_dict(torch.load(str(path), weights_only=True))
    model.eval()
    return model


def load_all_models():
    """Return the `models` dict the notebook used in Stages 5-6."""
    return {name: load_checkpoint(name) for name in config.MODEL_NAMES}
