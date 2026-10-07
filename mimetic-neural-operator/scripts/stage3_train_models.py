"""Stage 3 — train Mimetic (hard), Soft (ablation) and Black-box GNN.

Requires stage 4 to have been run (it needs data/trajectories.pt).

    python scripts/stage3_train_models.py [--epochs 8] [--K 4]

Writes checkpoints/{mimetic,soft,blackbox}.pt
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from mimetic_no import config
from mimetic_no.data import load_trajectories
from mimetic_no.models import BlackBoxGNN, MimeticOperator, SoftMimeticOperator
from mimetic_no.training import save_checkpoint, train_model


def main(epochs: int = config.EPOCHS, K: int = config.K) -> None:
    config.set_seeds()
    train_trajs, _, _ = load_trajectories()

    mimetic = MimeticOperator()
    soft = SoftMimeticOperator()
    blackbox = BlackBoxGNN()

    print("Train MIMETIC (hard)...")
    train_model(mimetic, train_trajs, epochs=epochs, K=K, lam=0.0)
    print("Train SOFT ablation (penalised)...")
    train_model(soft, train_trajs, epochs=epochs, K=K, lam=config.LAM_SOFT)
    print("Train BLACK-BOX GNN...")
    train_model(blackbox, train_trajs, epochs=epochs, K=K, lam=0.0)

    save_checkpoint(mimetic, "Mimetic (hard)")
    save_checkpoint(soft, "Soft (ablation)")
    save_checkpoint(blackbox, "Black-box GNN")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 3: train the three operators")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--K", type=int, default=config.K, help="rollout-training horizon")
    args = ap.parse_args()
    main(epochs=args.epochs, K=args.K)
