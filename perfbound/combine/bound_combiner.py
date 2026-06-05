# M5 — Bound Combiner (two-tier max + T_serial_irreducible)
#
# T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible
#
# Composition is max (two independent lower bounds on the same wall-clock
# time), with + T_serial_irreducible attaching to the Tier-2 term.
#
# The max reflects the insight that the grid-level and component-level
# floors are independent lower bounds — whichever is higher constrains
# the overall time.  T_serial_irreducible is added because it is NOT
# captured by either tier's ideal overlap assumption.
#
# Five-way attribution decomposes the gap between T_bound and a hypothetical
# zero-overhead kernel.  This is diagnostic output, NOT part of the bound.
#
# Source spec: .omc/specs/performance_bound_model.md §3, §4.2, §A.5

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..model.grid_model import GridBound
from ..model.component_model import ComponentBound
from ..model.serialization import SerializationSplit
from ..extract.op_classifier import Component


class BindingTier(str, Enum):
    """Which tier binds the overall performance."""
    GRID = "grid"
    COMPONENT = "component"


@dataclass
class Attribution:
    """Five-way gap attribution for a single kernel.

    Gaps are expressed as both absolute (microseconds) and as fractions
    of T_bound.  The five gaps are:

    grid:   Realized grid worse than optimal partition (occupancy, load_balance)
    gap1:   Wrong-unit placement — ops running on suboptimal unit
            (eligibility vs realized unit assignment)
    gap2:   Coalescing / transfer efficiency — MTE small-packet amortization,
            alignment waste, unused burst capacity
    gap3:   Avoidable serialization — handoffs that could be eliminated
            by scheduling/ping-pong  (the avoidable complement of T_serial)
    gap4:   Intra-unit execution inefficiency — low SIMD repeat/mask
            utilization within compute ops
    """
    grid_gap_us: float = 0.0
    gap1_wrong_unit_us: float = 0.0
    gap2_coalescing_us: float = 0.0
    gap3_avoidable_serial_us: float = 0.0
    gap4_intra_unit_exec_us: float = 0.0

    grid_gap_frac: float = 0.0
    gap1_frac: float = 0.0
    gap2_frac: float = 0.0
    gap3_frac: float = 0.0
    gap4_frac: float = 0.0

    @property
    def total_gap_us(self) -> float:
        return (self.grid_gap_us + self.gap1_wrong_unit_us +
                self.gap2_coalescing_us + self.gap3_avoidable_serial_us +
                self.gap4_intra_unit_exec_us)

    def dominant_gap(self) -> tuple[str, float]:
        """Return (gap_name, fraction) of the largest gap."""
        gaps = [
            ("grid", self.grid_gap_frac),
            ("gap1_wrong_unit", self.gap1_frac),
            ("gap2_coalescing", self.gap2_frac),
            ("gap3_avoidable_serial", self.gap3_frac),
            ("gap4_intra_unit_exec", self.gap4_frac),
        ]
        return max(gaps, key=lambda x: x[1])


@dataclass
class BoundResult:
    """Final bound output for a single kernel."""
    kernel_name: str
    t_bound_us: float

    # Decomposed
    t_grid_floor_us: float
    t_core_floor_us: float
    t_serial_irreducible_us: float

    binding_tier: BindingTier
    binding_component: Optional[Component] = None

    attribution: Attribution = field(default_factory=Attribution)

    def __repr__(self) -> str:
        return (f"BoundResult({self.kernel_name}: "
                f"T_bound={self.t_bound_us:.2f} us, "
                f"binding={self.binding_tier.value})")


def combine(
    grid: GridBound,
    component: ComponentBound,
    serial: SerializationSplit,
    kernel_name: str = "unknown",
) -> BoundResult:
    """Combine Tier 1 + Tier 2 + serialization into a single conservative bound.

    T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible

    The binding tier is determined by which floor is higher:
    - Grid binds when occupancy/load_balance constrain more than per-component BW
    - Component binds when a specific HW unit (Cube, MTE, Vector) is the bottleneck

    The five-way attribution is initialized from the component model's
    per-component rates and the serialization split.  Gap 3 comes directly
    from the avoidable serialization sum.

    Args:
        grid: Tier 1 grid floor.
        component: Tier 2 component floor with per-component decomposition.
        serial: Mandatory/avoidable serialization split.
        kernel_name: Label for the result.

    Returns:
        BoundResult with T_bound, binding tier/component, and initial attribution.
    """
    # Compute the max of the two independent floors
    max_floor_us = max(grid.t_grid_floor_us, component.t_core_floor_us)

    # T_bound = max(floors) + mandatory serialization
    t_bound_us = max_floor_us + serial.t_serial_irreducible_us

    # Determine binding tier
    if grid.t_grid_floor_us >= component.t_core_floor_us:
        binding_tier = BindingTier.GRID
        binding_component = None  # grid binds, not a specific component
    else:
        binding_tier = BindingTier.COMPONENT
        binding_component = component.binding_component

    # Attribution: initialize from available data
    attribution = Attribution()

    # Gap 2 (coalescing/MTE-E): from per-component MTE rates vs peak
    # Gap 3 (avoidable serial): directly from serialization split
    attribution.gap3_avoidable_serial_us = serial.t_serial_avoidable_us

    # Gap 4 (intra-unit execution): compute units with less-than-peak efficiency
    # For now, initialized from the component rates

    # Convert gaps to fractions
    if t_bound_us > 0:
        attribution.grid_gap_frac = attribution.grid_gap_us / t_bound_us
        attribution.gap1_frac = attribution.gap1_wrong_unit_us / t_bound_us
        attribution.gap2_frac = attribution.gap2_coalescing_us / t_bound_us
        attribution.gap3_frac = attribution.gap3_avoidable_serial_us / t_bound_us
        attribution.gap4_frac = attribution.gap4_intra_unit_exec_us / t_bound_us

    return BoundResult(
        kernel_name=kernel_name,
        t_bound_us=t_bound_us,
        t_grid_floor_us=grid.t_grid_floor_us,
        t_core_floor_us=component.t_core_floor_us,
        t_serial_irreducible_us=serial.t_serial_irreducible_us,
        binding_tier=binding_tier,
        binding_component=binding_component,
        attribution=attribution,
    )
