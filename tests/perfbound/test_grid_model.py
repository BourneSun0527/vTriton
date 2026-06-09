# A.4 — Grid model tests.
#
# Verify compute_grid_floor with explicit total_work parameter
# (not derived from GridInfo.work).
#
# Source: .omc/plans/a4_two_analytical_models.md Change #2

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.calibration.constants import CoreConfig, MemBandwidth
from perfbound.extract.dsl_extractor import GridInfo
from perfbound.model.grid_model import compute_grid_floor


class TestGridFloorBasic:
    """Grid floor with hand-computed values."""

    def test_perfect_occupancy_balance(self):
        """20 cores, occupancy=1.0, balance=1.0: T = total_work / (20 * i_binding).

        total_work = 1,000,000 bytes, i_binding = 180,000 B/us
        T = 1,000,000 / (20 * 180,000) = 0.2778 us
        """
        grid = GridInfo(
            grid_dims=(20,), total_programs=20,
            tile_assignment={i: (i,) for i in range(20)},
            work={i: 50000.0 for i in range(20)},
            occupancy=1.0, load_balance=1.0, redundancy=1.0,
            busiest_core_id=0,
        )
        core = CoreConfig()
        result = compute_grid_floor(grid, core, i_binding=180000.0,
                                    total_work=1_000_000.0)
        assert result.t_grid_floor_us == pytest.approx(1_000_000 / (20 * 180000),
                                                        rel=1e-3)
        assert result.n_cores == 20
        assert result.occupancy == 1.0
        assert result.load_balance == 1.0

    def test_partial_occupancy(self):
        """16/20 cores: occupancy=0.8.

        total_work = 800,000, i_binding = 180,000 B/us
        T = 800,000 / (20 * 0.8 * 1.0 * 180,000) = 0.2778 us
        """
        grid = GridInfo(
            grid_dims=(16,), total_programs=16,
            tile_assignment={i: (i,) for i in range(16)},
            work={i: 50000.0 for i in range(16)},
            occupancy=0.8, load_balance=1.0, redundancy=1.0,
            busiest_core_id=0,
        )
        core = CoreConfig()
        result = compute_grid_floor(grid, core, i_binding=180000.0,
                                    total_work=800_000.0)
        expected = 800_000.0 / (20 * 0.8 * 1.0 * 180000.0)
        assert result.t_grid_floor_us == pytest.approx(expected, rel=1e-3)

    def test_imbalanced_load(self):
        """Load balance < 1.0: slowest core has more work.

        total_work = 1,000,000, i_binding = 100,000
        occupancy = 1.0, load_balance = 0.8
        T = 1,000,000 / (20 * 1.0 * 0.8 * 100,000) = 0.625 us
        """
        grid = GridInfo(
            grid_dims=(20,), total_programs=20,
            tile_assignment={i: (i,) for i in range(20)},
            work={0: 100000.0, **{i: 47368.0 for i in range(1, 20)}},
            occupancy=1.0, load_balance=0.8, redundancy=1.0,
            busiest_core_id=0,
        )
        core = CoreConfig()
        result = compute_grid_floor(grid, core, i_binding=100000.0,
                                    total_work=1_000_000.0)
        expected = 1_000_000.0 / (20 * 1.0 * 0.8 * 100000.0)
        assert result.t_grid_floor_us == pytest.approx(expected, rel=1e-3)


class TestGridFloorUnits:
    """Verify total_work unit matches i_binding unit."""

    def test_compute_bound_flops(self):
        """total_work in FLOPs, i_binding in FLOP/us."""
        grid = GridInfo(
            grid_dims=(20,), total_programs=20,
            tile_assignment={i: (i,) for i in range(20)},
            work={i: 1.0 for i in range(20)},
            occupancy=1.0, load_balance=1.0, redundancy=1.0,
            busiest_core_id=0,
        )
        core = CoreConfig()
        total_flops = 280_000_000.0  # 280 MFLOPs
        i_binding = 280_000_000.0    # 280 MFLOP/us (280 TFLOPS)
        result = compute_grid_floor(grid, core, i_binding=i_binding,
                                    total_work=total_flops)
        # T = 280e6 / (20 * 1.0 * 1.0 * 280e6) = 1/20 = 0.05 us
        assert result.t_grid_floor_us == pytest.approx(0.05, rel=1e-3)

    def test_redundancy_scales_work(self):
        """Redundancy > 1 amplifies total_work in the formula."""
        grid = GridInfo(
            grid_dims=(20,), total_programs=20,
            tile_assignment={i: (i,) for i in range(20)},
            work={i: 1.0 for i in range(20)},
            occupancy=1.0, load_balance=1.0, redundancy=1.5,
            busiest_core_id=0,
        )
        core = CoreConfig()
        result = compute_grid_floor(grid, core, i_binding=180000.0,
                                    total_work=100_000.0)
        # T = (100,000 * 1.5) / (20 * 1.0 * 1.0 * 180,000)
        expected = (100_000.0 * 1.5) / (20 * 180_000.0)
        assert result.t_grid_floor_us == pytest.approx(expected, rel=1e-3)


class TestGridFloorEdgeCases:
    """Edge cases for grid floor computation."""

    def test_vector_only_kernel(self):
        """Vector-only kernel uses 40 AIV cores."""
        grid = GridInfo(
            grid_dims=(40,), total_programs=40,
            tile_assignment={i: (i,) for i in range(40)},
            work={i: 25000.0 for i in range(40)},
            occupancy=1.0, load_balance=1.0, redundancy=1.0,
            busiest_core_id=0,
        )
        core = CoreConfig()
        result = compute_grid_floor(grid, core, i_binding=180000.0,
                                    total_work=1_000_000.0,
                                    is_cube_kernel=False)
        assert result.n_cores == 40
        expected = 1_000_000.0 / (40 * 180_000.0)
        assert result.t_grid_floor_us == pytest.approx(expected, rel=1e-3)

    def test_single_core(self):
        """1 program on 20 cores: occupancy=0.05."""
        grid = GridInfo(
            grid_dims=(1,), total_programs=1,
            tile_assignment={0: (0,)},
            work={0: 100_000.0},
            occupancy=0.05, load_balance=1.0, redundancy=1.0,
            busiest_core_id=0,
        )
        core = CoreConfig()
        result = compute_grid_floor(grid, core, i_binding=180000.0,
                                    total_work=100_000.0)
        expected = 100_000.0 / (20 * 0.05 * 1.0 * 180_000.0)
        assert result.t_grid_floor_us == pytest.approx(expected, rel=1e-3)
