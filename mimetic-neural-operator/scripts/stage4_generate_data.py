"""Stage 4 — data generation (varied ICs / conductances / sources) + OOD split + saved dataset.

Run BEFORE stage 3 training: training consumes the trajectories this writes.

    python scripts/stage4_generate_data.py

Writes data/trajectories.pt (full tensors for later stages) and data/dataset.npz
(the compact artifact the notebook saved).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from mimetic_no import config
from mimetic_no.data import make_trajectories, save_dataset_npz, save_trajectories


def main() -> None:
    config.set_seeds()

    # Train / in-distribution test: 15-node graphs, k in [0.5,1.5]
    train_trajs = make_trajectories(**config.TRAIN_SPEC)
    id_test = make_trajectories(**config.ID_TEST_SPEC)
    # OOD: unseen LARGER topology (22 nodes) AND shifted conductance range [1.5,2.5]
    ood_test = make_trajectories(**config.OOD_TEST_SPEC)
    print(f"train={len(train_trajs)}  id_test={len(id_test)}  ood_test={len(ood_test)}")

    save_dataset_npz(train_trajs, id_test, ood_test)
    save_trajectories(train_trajs, id_test, ood_test)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Stage 4: generate trajectories + OOD split").parse_args()
    main()
