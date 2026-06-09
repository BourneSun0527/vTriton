# M4 — Single entry point wiring both analytical models.
#
# compute_bounds picks i_binding and total_work consistently
# (memory-bound: BW_gm_ub + Σ MTE bytes; compute-bound: Cube FLOP/us + Σ flops)
# and calls compute_grid_floor, compute_component_floor, classify_handoffs.
#
# This replaces ad-hoc grid construction scattered in tests and bound_from_extract.
# The choice mirrors the component model's binding component: whichever component
# has the highest O_c/I_c determines the grid's i_binding unit.
#
# Source: .omc/plans/a4_two_analytical_models.md Change #3

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..extract.hivm_extractor import HIVMExtract
from ..extract.dsl_extractor import GridInfo
from ..calibration.constants import CoreConfig
from .grid_model import GridBound, compute_grid_floor
from .component_model import ComponentBound, compute_component_floor
from .serialization import SerializationSplit, classify_handoffs

if TYPE_CHECKING:
    from ..calibration.constants import CalibrationDB


@dataclass
class BoundPieces:
    """The three pieces of an analytical bound, pre-combination.

    A.5's combine() consumes this to produce T_bound with attribution.
    """
    grid: GridBound
    component: ComponentBound
    serial: SerializationSplit


def compute_bounds(
    grid_info: GridInfo,
    extract: HIVMExtract,
    calib_db: "CalibrationDB",
    is_cube_kernel: bool = True,
) -> BoundPieces:
    """Compute the three bound pieces from grid + extract + calibration.

    Picks i_binding and total_work consistently:
      - memory-bound (MTE binds at component level): i_binding = BW_gm_ub,
        total_work = Σ MTE bytes * loop_multiplier
      - compute-bound (Cube binds): i_binding = Cube FLOP/us,
        total_work = Σ flops * loop_multiplier

    The grid floor's unit matches i_binding by construction. The component
    floor's binding component determines which path executes; if the component
    model says MTE binds, the grid should too (same bottleneck).

    Args:
        grid_info: M2-extracted grid quantities.
        extract: M3 HIVM extraction result.
        calib_db: Calibration database with sustained rates.
        is_cube_kernel: True for Cube-bearing kernels (20 AIC).

    Returns:
        BoundPieces with grid, component, and serial pieces.
    """
    core = calib_db.core
    cube = calib_db.cube
    vector = calib_db.vector
    memory = calib_db.memory

    # Compute component floor first to discover which component binds
    comp = compute_component_floor(extract, cube, vector, memory, core)

    # Pick i_binding and total_work to match the binding component's unit
    binding = comp.binding_component
    from ..extract.op_classifier import Component

    if binding in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
        # Memory-bound: i_binding = BW in B/us, total_work = bytes
        try:
            i_binding, _ = memory.lookup_bw("gm", "ub")
        except KeyError:
            i_binding = 1.0
        total_work = sum(
            float(op.bytes_transferred) * float(op.loop_multiplier)
            for op in extract.operations
        )
    else:
        # Compute-bound (Cube, Vector): i_binding = throughput in FLOP/us
        if binding == Component.CUBE:
            from .component_model import _get_cube_throughput_ops_per_us, _prec_to_dtype
            from ..extract.op_classifier import Precision
            i_binding = _get_cube_throughput_ops_per_us(
                _prec_to_dtype(Precision.FP16), cube
            )
        else:
            # Vector: use aggregate TFLOPS
            i_binding = vector.throughput_fp16_tflops * 1e6  # FLOP/us
        total_work = sum(
            float(op.flops if op.flops > 0 else op.elements)
            * float(op.loop_multiplier)
                for op in extract.operations
                if op.component not in (
                    Component.MTE_GM, Component.MTE_L1, Component.MTE_UB,
                )
            )

    total_work = max(total_work, 1.0)  # avoid division by zero

    grid = compute_grid_floor(grid_info, core, i_binding, total_work,
                              is_cube_kernel=is_cube_kernel)

    serial = classify_handoffs(
        extract.handoffs,
        mandatory_handoff_cycles=calib_db.mandatory_handoff_cycles,
        clock_ghz=core.clock_freq_ghz,
    )

    return BoundPieces(grid=grid, component=comp, serial=serial)
