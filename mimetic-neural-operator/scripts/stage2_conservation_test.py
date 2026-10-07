"""Stage 2 unit test — closed system (no sources) conserves energy to machine precision.

    python scripts/stage2_conservation_test.py [--show]

Writes results/figures/stage2_unit_test.png and asserts the residual floor.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from mimetic_no import config
from mimetic_no.dynamics import conservation_unit_test
from mimetic_no.plotting import plot_unit_test, use_headless_backend


def main(show: bool = False) -> None:
    if not show:
        use_headless_backend()
    config.set_seeds()

    res64 = conservation_unit_test(**config.UNIT_TEST)
    print("closed-system conservation residual (float64):  max = %.3e" % res64.max())
    assert res64.max() < config.UNIT_TEST_TOL, "discretization is NOT mimetic!"
    print("PASS: discretization conserves energy to machine precision.")

    plot_unit_test(res64, show=show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 2: mimetic conservation unit test")
    ap.add_argument("--show", action="store_true", help="display figures interactively")
    args = ap.parse_args()
    main(show=args.show)
