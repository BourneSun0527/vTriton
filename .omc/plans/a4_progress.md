# A.4 Progress — Module 4: The Two Analytical Models

## Status: COMPLETE

## Changes Made

### Change #1: Component model consumes flops (US-A4-001)
**File:** `perfbound/model/component_model.py` (line 196-202)
- Cube/Vector/Scalar branch now uses `work_raw = op.flops if op.flops > 0 else op.elements`
- Work-unit invariant pinned in comment: Cube rate is FLOP/us, Vector fallback is FLOP/us
- MTE unaffected (bytes are correct)
- Spine check: T_cube = 3.252 us on real 910B3 calib (flops consumed, not elements)

### Change #2: Grid model explicit total_work parameter (US-A4-002)
**File:** `perfbound/model/grid_model.py` (line 47-112)
- `compute_grid_floor` signature: `(grid, core, i_binding, total_work, is_cube_kernel=True)`
- `total_work` is caller-supplied (bytes or FLOPs) matching `i_binding` units
- `GridInfo.work` used only for occupancy/load_balance ratios (units cancel)
- Redundancy applied internally: `scaled_work = total_work * redundancy`

### Change #3: Delete bypass in bound_from_extract (US-A4-003)
**File:** `perfbound/combine/bound_combiner.py` (line 210-262)
- `bound_from_extract` constructs minimal `GridInfo` from caller's occupancy/load_balance
- Calls `compute_grid_floor(grid_info, core, i_binding, total_work)` — single path
- Hand-rolled GridBound construction deleted

### Change #5: Serialization Σ over distinct mandatory edges (US-A4-004)
**File:** `perfbound/model/serialization.py` (line 175-191)
- Dedup mandatory handoffs by `(producer_component, consumer_component)` edge
- `t_serial_irreducible_us = len(distinct_edges) * (mandatory_handoff_cycles / cycles_per_us)`
- Same edge repeated across loop iterations → counted once (pipeline steady-state)
- Distinct edges in dependency chain → sequential and sum
- `mandatory_handoff_cycles == 0` → 0.0 conservative path preserved

### Change #4: Acceptance golden test (US-A4-005, US-A4-006)
**File:** `tests/perfbound/test_component_model.py`
- Fixtures relabeled: `flops=2*M*N*K`, `elements=M*N` (2K× ratio, regression guard asserts `flops != elements`)
- `TestAttentionGolden`: FlashAttention-shaped kernel (Cube→Vector→Cube)
  - 7 operations, 6 handoffs (2 mandatory, 4 avoidable)
  - All intermediates verified to 3 sig figs via `pytest.approx(rel=1e-3)`
  - Spreadsheet in docstring: T_cube=0.120us, T_vector=0.00182us, T_mte_gm=2.913us, T_serial=2.162us

### Change #3: M4 driver — compute_bounds (post-review fix)
**File (new):** `perfbound/model/bounds.py`
- `compute_bounds(grid_info, extract, calib_db, is_cube_kernel)` — single entry point
- Picks `i_binding` and `total_work` consistently:
  - Memory-bound (MTE binds): `i_binding = BW_gm_ub`, `total_work = Σ MTE bytes`
  - Compute-bound (Cube/Vector binds): `i_binding = Cube/Vector throughput`, `total_work = Σ flops`
- Returns `BoundPieces(grid, component, serial)` for A.5's `combine` to consume
- Exported from `perfbound/model/__init__.py`

### Post-review fixes
- **serialization.py**: Added soundness comment noting dependency-chain assumption for edge summing
- **bound_combiner.py**: GridInfo synthesis uses `work={}` (empty dict) instead of stuffing bytes into work field
- **model/__init__.py**: Now exports all public API symbols

## Files Changed
| File | Change |
|------|--------|
| `perfbound/model/component_model.py` | flops consumption (3 lines) |
| `perfbound/model/grid_model.py` | total_work parameter, formula rewrite |
| `perfbound/model/serialization.py` | Σ over distinct edges + soundness comment |
| `perfbound/model/bounds.py` | NEW: compute_bounds driver (memory-bound / compute-bound grid) |
| `perfbound/model/__init__.py` | Public API exports |
| `perfbound/combine/bound_combiner.py` | bypass deletion, compute_grid_floor call, work={} fix |
| `tests/perfbound/test_component_model.py` | fixture relabel + attention golden |
| `tests/perfbound/test_serialization.py` | NEW: dedup tests |
| `tests/perfbound/test_grid_model.py` | NEW: grid floor tests |

## Verification
- 141 passed, 3 xfailed in full `tests/perfbound/` suite
- Spine check: T_cube = 3.252 us (flops path, real calib)
- Attention golden: all intermediates match spreadsheet to 3 sig figs

## Open Items Closed
| Item | Source | Closed by |
|------|--------|-----------|
| flops-consumption loop | a3_progress "A.4 Handoff" | Change #1 |
| Grid floor never wired end-to-end | A.4 analysis | Changes #2, #3 |
| compute_grid_floor dead/bypassed | A.4 analysis | Changes #2, #3 |
| T_serial_irreducible flat-cost vs Σ | spec §2.2/§4.0 | Change #5 |

## Known Divergences (conservative-safe)
- **Scalar t_c = 0**: no P_scalar calibration constant exists yet. Zeroing a component can only lower the floor (conservative). The acceptance golden kernel has no Scalar work. Gap-1 / Group-V attribution (A.5/B.4) is where this matters.
- **avoidable-serial stub**: `serialization.py:198` `t_serial_avoidable_us += 0.0` feeds Gap 3 (A.5 attribution). Not blocking A.4.

## NOT in A.4 (A.5+ scope)
- Avoidable serialization cost computation (Gap 3)
- bound_combiner gap wiring (_compute_gap1/2/4, _wire_gaps)
- L2-cache hit-rate BW model
- Wire `bound_from_extract` to call `compute_bounds` instead of inline construction (deferred: current path works, just not via the driver)
