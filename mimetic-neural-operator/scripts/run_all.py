"""Run the whole pipeline end to end, in dependency order.

    python scripts/run_all.py
    python scripts/run_all.py --fast          # fewer epochs, smaller sweeps
    python scripts/run_all.py --skip 5plus 6

Order: stage1 -> stage2 -> stage4 (data) -> stage3 (train) -> stage5 -> stage5+ -> stage6.
Note stage 4 runs before stage 3: training needs the generated trajectories.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time

import stage1_graph_and_dynamics
import stage2_conservation_test
import stage3_train_models
import stage4_generate_data
import stage5_evaluate
import stage5plus_robustness
import stage6_topologies

from mimetic_no import config
from mimetic_no.plotting import use_headless_backend

ORDER = ["1", "2", "4", "3", "5", "5plus", "6"]


def _banner(label: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {label}")
    print("=" * 78)


def main(skip=None, fast: bool = False, show: bool = False) -> None:
    if not show:
        use_headless_backend()
    skip = set(skip or [])
    t0 = time.time()
    epochs = 2 if fast else config.EPOCHS

    if "1" not in skip:
        _banner("STAGE 1 — graph, incidence operator, true dynamics")
        stage1_graph_and_dynamics.main()

    if "2" not in skip:
        _banner("STAGE 2 — closed-system conservation unit test")
        stage2_conservation_test.main(show=show)

    if "4" not in skip:
        _banner("STAGE 4 — data generation + OOD split")
        stage4_generate_data.main()

    if "3" not in skip:
        _banner("STAGE 3 — train mimetic / soft / black-box")
        stage3_train_models.main(epochs=epochs)

    if "5" not in skip:
        _banner("STAGE 5 — evaluation (drift, RMSE, plausibility, OOD)")
        stage5_evaluate.main(show=show)

    if "5plus" not in skip:
        _banner("STAGE 5+ — robustness experiments A-E")
        stage5plus_robustness.main(only=["C", "E"] if fast else None, show=show)

    if "6" not in skip:
        _banner("STAGE 6 — structured DH topologies")
        stage6_topologies.main(show=show, skip_pandapipes=fast)

    print(f"\nAll done in {time.time() - t0:.1f}s. Figures: {config.FIGURES_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run every stage in order")
    ap.add_argument("--skip", nargs="+", choices=ORDER, help="stages to skip")
    ap.add_argument("--fast", action="store_true",
                    help="smoke-test mode: fewer epochs, subset of experiments")
    ap.add_argument("--show", action="store_true", help="display figures interactively")
    args = ap.parse_args()
    main(skip=args.skip, fast=args.fast, show=args.show)
