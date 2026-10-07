"""mimetic_no — Structure-Preserving Neural Operator for district-heating networks.

Library code extracted verbatim from the original Colab notebook
`Mimetic_corrected_with_topology (1).ipynb`. No logic has been changed:
every function/class body is the notebook's, only relocated into modules.

Module map
----------
config      paths, seeds, dataset specs
graph       Stage 1: random graphs, signed incidence matrix, edge index
dynamics    Stage 1-2: true flux, source term, simulators, conservation unit test
data        Stage 4: trajectory generation, OOD split, dataset save/load
models      Stage 3: EdgeFluxNet, Mimetic / Soft / Black-box operators
training    Stage 3: K-step rollout trainer + checkpoint helpers
evaluation  Stage 5: rollouts, conservation drift, RMSE, integrator sweep
topologies  Stage 6: tree / Delaunay / grid / geometric / pandapipes generators
plotting    all figures from the notebook
cli         `python -m mimetic_no.cli <stage>` entry point
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    config,
    data,
    dynamics,
    evaluation,
    graph,
    models,
    plotting,
    topologies,
    training,
)

__all__ = [
    "config",
    "graph",
    "dynamics",
    "data",
    "models",
    "training",
    "evaluation",
    "topologies",
    "plotting",
]
