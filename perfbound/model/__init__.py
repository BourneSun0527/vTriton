# M4 — Two Analytical Models (pure functions, no I/O, no compilation)
#
# Grid model (Tier 1):
#   T_grid_floor = T_total_work / (n_cores · occupancy · load_balance · I_binding)
#
# Component model (Tier 2):
#   I_c per component via weighted-harmonic mean (Eq. 4),
#   T_core_floor = max_c(O_c / I_c)
#
# Serialization split:
#   Classify each handoff as mandatory vs avoidable.
#   Mandatory → T_serial_irreducible.  Errs toward "avoidable"
#   to preserve bound soundness.

from .bounds import compute_bounds, BoundPieces
from .grid_model import compute_grid_floor, GridBound
from .component_model import (
    compute_component_floor, compute_component_floor_from_db,
    ComponentBound, ComponentRate,
)
from .serialization import classify_handoffs, SerializationSplit

__all__ = [
    "compute_bounds", "BoundPieces",
    "compute_grid_floor", "GridBound",
    "compute_component_floor", "compute_component_floor_from_db",
    "ComponentBound", "ComponentRate",
    "classify_handoffs", "SerializationSplit",
]
