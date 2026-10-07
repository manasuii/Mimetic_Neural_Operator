"""Single entry point: `python -m mimetic_no.cli <stage>`.

Thin dispatcher over the scripts/ files, so the package is usable after
`pip install -e .` without touching the scripts directory:

    python -m mimetic_no.cli stage1
    python -m mimetic_no.cli stage5plus --only C E
    python -m mimetic_no.cli all --fast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

STAGES = {
    "stage1": "stage1_graph_and_dynamics",
    "stage2": "stage2_conservation_test",
    "stage4": "stage4_generate_data",
    "stage3": "stage3_train_models",
    "stage5": "stage5_evaluate",
    "stage5plus": "stage5plus_robustness",
    "stage6": "stage6_topologies",
    "all": "run_all",
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mimetic-no",
                                 description="Structure-preserving neural operator pipeline")
    ap.add_argument("stage", choices=sorted(STAGES), help="which stage to run")
    ap.add_argument("--only", nargs="+", help="stage5plus: experiments to run (A-E)")
    ap.add_argument("--show", action="store_true", help="display figures interactively")
    ap.add_argument("--fast", action="store_true", help="all: smoke-test mode")
    args = ap.parse_args(argv)

    import importlib
    mod = importlib.import_module(STAGES[args.stage])

    if args.stage == "stage5plus":
        mod.main(only=args.only, show=args.show)
    elif args.stage == "all":
        mod.main(fast=args.fast, show=args.show)
    elif args.stage in ("stage2", "stage5", "stage6"):
        mod.main(show=args.show)
    else:
        mod.main()


if __name__ == "__main__":
    main()
