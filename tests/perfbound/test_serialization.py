# A.4 — Serialization dedup tests.
#
# Verify that T_serial_irreducible sums over DISTINCT mandatory edges,
# deduplicating same-edge repeats across loop iterations.
#
# Source: .omc/plans/a4_two_analytical_models.md Change #5

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import HandoffRecord
from perfbound.extract.op_classifier import Component
from perfbound.model.serialization import classify_handoffs


class TestDedupSameEdge:
    """Same (producer, consumer) edge repeated across loop iterations.

    Steady-state pipelining means the same edge costs once, not per iteration.
    """

    def test_same_edge_repeated_counted_once(self):
        """Two Cube->Vector handoffs (same edge) count as one."""
        handoffs = [
            HandoffRecord(1, 3, Component.CUBE, Component.VECTOR, 1024,
                          is_mandatory=None),
            # Same edge, different op ids (different loop iteration)
            HandoffRecord(5, 7, Component.CUBE, Component.VECTOR, 1024,
                          is_mandatory=None),
        ]
        serial = classify_handoffs(handoffs, mandatory_handoff_cycles=2000,
                                   clock_ghz=1.85)
        # Both classified as mandatory
        assert len(serial.mandatory_handoffs) == 2
        # But deduped to 1 distinct edge
        expected = 1 * (2000.0 / 1850.0)  # 1.081 us (NOT 2.162)
        assert serial.t_serial_irreducible_us == pytest.approx(expected, rel=1e-3)

    def test_three_same_edges_still_one(self):
        """Three Cube->Vector handoffs (same edge) count as one."""
        handoffs = [
            HandoffRecord(1, 2, Component.CUBE, Component.VECTOR, 512,
                          is_mandatory=None),
            HandoffRecord(3, 4, Component.CUBE, Component.VECTOR, 512,
                          is_mandatory=None),
            HandoffRecord(5, 6, Component.CUBE, Component.VECTOR, 512,
                          is_mandatory=None),
        ]
        serial = classify_handoffs(handoffs, mandatory_handoff_cycles=3000,
                                   clock_ghz=1.85)
        expected = 1 * (3000.0 / 1850.0)
        assert serial.t_serial_irreducible_us == pytest.approx(expected, rel=1e-3)


class TestDistinctEdges:
    """Distinct mandatory edges sum."""

    def test_two_distinct_edges_sum(self):
        """Cube->Vector and Vector->Cube are distinct and sum."""
        handoffs = [
            HandoffRecord(1, 2, Component.CUBE, Component.VECTOR, 1024,
                          is_mandatory=None),
            HandoffRecord(2, 3, Component.VECTOR, Component.CUBE, 1024,
                          is_mandatory=None),
        ]
        serial = classify_handoffs(handoffs, mandatory_handoff_cycles=2000,
                                   clock_ghz=1.85)
        expected = 2 * (2000.0 / 1850.0)  # 2.162 us
        assert serial.t_serial_irreducible_us == pytest.approx(expected, rel=1e-3)

    def test_mixed_distinct_and_repeated(self):
        """Cube->Vector (2x) and Vector->Cube (1x): 2 distinct edges sum."""
        handoffs = [
            HandoffRecord(1, 2, Component.CUBE, Component.VECTOR, 1024,
                          is_mandatory=None),
            HandoffRecord(3, 4, Component.CUBE, Component.VECTOR, 1024,
                          is_mandatory=None),  # repeat of same edge
            HandoffRecord(4, 5, Component.VECTOR, Component.CUBE, 1024,
                          is_mandatory=None),
        ]
        serial = classify_handoffs(handoffs, mandatory_handoff_cycles=2000,
                                   clock_ghz=1.85)
        expected = 2 * (2000.0 / 1850.0)  # 2 distinct edges
        assert serial.t_serial_irreducible_us == pytest.approx(expected, rel=1e-3)


class TestNoCalibration:
    """Conservative behavior when calibration is missing."""

    def test_zero_cycles_gives_zero_tserial(self):
        """mandatory_handoff_cycles=0 -> T_serial=0 (conservative)."""
        handoffs = [
            HandoffRecord(1, 2, Component.CUBE, Component.VECTOR, 1024,
                          is_mandatory=None),
        ]
        serial = classify_handoffs(handoffs, mandatory_handoff_cycles=0,
                                   clock_ghz=1.85)
        assert serial.t_serial_irreducible_us == 0.0
        assert len(serial.mandatory_handoffs) == 1  # still classified correctly

    def test_no_handoffs(self):
        """Empty handoff list -> T_serial=0."""
        serial = classify_handoffs([], mandatory_handoff_cycles=2000,
                                   clock_ghz=1.85)
        assert serial.t_serial_irreducible_us == 0.0
        assert len(serial.mandatory_handoffs) == 0
