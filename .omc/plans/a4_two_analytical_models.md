# A.4 Plan — Module 4: The Two Analytical Models (units discipline + acceptance)

> On approval, save the canonical copy to `.omc/plans/a4_two_analytical_models.md`
> (consistent with `a0_`–`a3_` naming) and create `.omc/plans/a4_progress.md` on completion.

---

## Context

A.4 (Module 4) is the spec's **two pure analytical models** (`.omc/specs/implementation_and_paper_plan.md` §A.4, lines 135–149):

- **Grid model (Tier 1):** M2 `GridInfo` + calibration → `T_grid_floor`, `busiest_core_id`.
- **Component model (Tier 2):** M3 `HIVMExtract` + calibration → `I_c` per component (Eq. 4 harmonic mean), `T_core_floor = max_c(O_c/I_c)`, and the **serialization split** (mandatory vs avoidable; mandatory min costs sum into `T_serial_irreducible`).

**Acceptance (the only bar, lines 148–149):** on a hand-computed kernel, every intermediate — `I_c`, `T_core_floor`, `T_serial_irreducible` — matches a spreadsheet to **3 significant figures**.

The three model files already exist as a scaffold (commit `52d9839`): `perfbound/model/{grid_model,component_model,serialization}.py`. They are **not spec-compliant** because of one root defect that appears in both floor functions:

> **Units defect (the spine of this plan):** the work *quantity* and its *rate* are in different units. The component model aggregates Cube/Vector work from `op.elements` (`component_model.py:198`) while the rate `I_c` is FLOP/us. The grid model sums `GridInfo.work[p]` — tile/row/element counts (`dsl_extractor.py:223`, `grid_idioms.py:137,191`) — while `i_binding` is B/us or FLOP/us. The existing tests pass only because the fixtures **stuff the FLOP count into `elements`**, and the golden/`bound_from_extract` paths **bypass `compute_grid_floor` entirely** and recompute the grid floor from `total_bytes` by hand. So both pure functions are wrong on the path that matters, and the bypasses hide it.

A.3 already did the upstream half: `OpRecord` carries `flops` and `elements` as **separate** fields, `extract_hivm` aggregates with `op.flops if op.flops > 0 else op.elements` (`hivm_extractor.py:407`), and `_infer_flops_from_loads` infers `2·M·N·K` for Cube matmul. That inferred work currently never reaches `T_core_floor`. A.4 closes this loop.

**Intended outcome:** the two floor functions consume the correct work field in the correct unit; the hand-rolled bypasses are deleted in favor of the real functions; and a single coherent golden test verifies all three intermediates to 3 sig figs with realistic field semantics (`flops ≠ elements`, bytes for MTE).

---

## Scope boundary (hold the A.4/A.5 line)

**In A.4:** the two pure floor functions, the mandatory/avoidable *classification* (already correct), `T_serial_irreducible` (**needs the Σ fix — Change #5**), a single M4 driver, and the acceptance golden test.

**NOT in A.4 (A.5 attribution / later):**
- `serialization.py:198` `t_serial_avoidable_us += 0.0` stub — feeds **Gap 3**, an A.5 attribution input. It does **not** block A.4 acceptance (which covers only `T_serial_irreducible`). Leave it; annotate as A.5-facing.
- `bound_combiner.py` gap wiring (`_compute_gap1/2/4`, `_wire_gaps`) — A.5. Do not rewrite.
- L2-cache hit-rate BW model — explicitly deferred to A.5/A.7 in `open-questions.md`.

**Stated divergence from `performance_bound_model.md` (sound, not aligned):**
- Spec §2.1 (lines 137, 154) lists **Scalar** as a first-class component with `O_c/I_c` in `T_core_floor = max over components c`. We keep `scalar t_c = 0` because no `P_scalar` calibration constant exists yet. This is *conservative-safe* for a lower bound (zeroing a component can only lower the floor, never break `T_bound ≤ T_measured`) but literally diverges: a Scalar-bound kernel gets `T_core_floor = 0`. Label it a **known divergence** in `a4_progress.md`, not alignment; the Gap-1 / Group-V attribution path (A.5/B.4) is where it matters. The acceptance golden kernel has no Scalar work, so this does not affect A.4 acceptance.

---

## Changes

### 1. Component model — consume `flops` for compute work  *(surgical; golden numbers unchanged)*
**File:** `perfbound/model/component_model.py`

- Line ~197–200, the Cube/Vector/Scalar branch: change
  `work = float(op.elements) * float(op.loop_multiplier)`
  → `work_raw = op.flops if op.flops > 0 else op.elements` then `work = float(work_raw) * float(op.loop_multiplier)`.
  This matches the aggregation contract already used in `hivm_extractor.py:407` and `bound_combiner._compute_gap1`.
- **Vector caveat — do not blanket-swap.** The Vector rate has two paths (`_get_vector_throughput_ops_per_us`: per-op `elements/us` vs fallback FLOP/us). Today the fallback (FLOP/us) runs because `op_name` is never passed (`:259`). Pin this explicitly: Vector work-unit must match whichever rate path executes. Keep the fallback FLOP/us path → `flops` is consistent. Add a one-line comment stating the invariant; do not silently enable the dead per-op path in A.4.
- Cube is unaffected dimensionally (rate already FLOP/us); MTE unaffected (bytes are correct).

### 2. Grid model — separate dimensionless efficiency from absolute work
**File:** `perfbound/model/grid_model.py`

The grid floor's numerator must be in the **same unit as `i_binding`** (bytes for memory-bound, FLOPs for compute-bound). `GridInfo.work[p]` counts are correct for the **ratios** `occupancy`/`load_balance` (units cancel) but **not** for the absolute numerator.

- Add an explicit `total_work` parameter (bytes or FLOPs, caller-supplied to match `i_binding`) to `compute_grid_floor`, instead of deriving it from `sum(grid.work.values())`. Keep `occupancy`, `load_balance`, `redundancy`, `busiest_core_id` sourced from `GridInfo`.
- Signature becomes `compute_grid_floor(grid, core, i_binding, total_work, is_cube_kernel=True)`. Formula unchanged: `total_work·redundancy / (n_cores·occupancy·load_balance·i_binding)`.
- This is the canonical path the bypasses were emulating. After this, **delete the hand-rolled grid floor** in `bound_combiner.bound_from_extract` (`:240–254`) and have it call `compute_grid_floor` with `total_work = Σ bytes·loop_multiplier` and `i_binding = BW_gm_ub`. (`bound_from_extract` is A.5 surface, but the bypass is the symptom of the A.4 defect; fixing it here proves the function works end-to-end.)

### 3. M4 driver — one entry point wiring both models
**File (new):** `perfbound/model/bounds.py`

A thin pure function `compute_bounds(grid_info, extract, calib_db, is_cube_kernel) -> (GridBound, ComponentBound, SerializationSplit)` that:
- picks `i_binding` and `total_work` consistently (memory-bound: `i_binding = BW_gm_ub`, `total_work = Σ MTE bytes`; compute-bound: Cube FLOP/us and `Σ flops`) — the choice mirrors the component model's binding component,
- calls `compute_grid_floor`, `compute_component_floor_from_db`, `classify_handoffs`,
- returns the three pieces for A.5's `combine` to consume.
Export from `perfbound/model/__init__.py`. This replaces ad-hoc grid construction scattered in tests and `bound_from_extract`.

### 5. Serialization — `T_serial_irreducible` must be a Σ over *distinct* mandatory edges
**File:** `perfbound/model/serialization.py`

Spec §2.2 (lines 170-178) and §4.0 (line 274): `T_serial_irreducible = Σ over mandatory handoffs h of min_cost(h)`. The current `:182-189` charges **one flat cost** (`mandatory_handoff_cycles / cycles_per_us`) regardless of how many distinct mandatory handoffs exist — neither a sum nor a correct pipelined-max. A FlashAttention-style kernel has **two distinct, sequential** mandatory handoffs (Cube→Vector for QKᵀ→softmax, then Vector→Cube for softmax→×V) that do **not** overlap each other; the current code counts one → under-counts serialization.

- Replace the flat assignment with: **dedup mandatory handoffs by their `(producer_component, consumer_component)` edge** (loop-iteration repeats of the *same* edge pipeline → counted once), then `t_serial_irreducible_us = Σ over distinct edges of (mandatory_handoff_cycles / cycles_per_us)`.
- Rationale for dedup-not-raw-sum: the same edge fired every loop iteration steady-states to one handoff latency; distinct edges in a dependency chain are sequential and sum. This is the faithful reading of §2.2's Σ.
- Keep the `mandatory_handoff_cycles == 0` → `0.0` conservative path and its no-fail behavior.
- Per-edge `min_cost` stays the single calibrated `mandatory_handoff_cycles` for now (one measured L0C→GM+GM→UB chain); a per-edge byte-dependent cost is a later refinement, not A.4.

### 4. Acceptance golden test — the centerpiece
**File:** `tests/perfbound/test_component_model.py` (extend) + the fixtures

- **Relabel fixtures so `flops ≠ elements`.** Today `matmul`/`matmul_vector` fixtures set `elements = 2·M·N·K`. Change to `flops = 2·M·N·K`, `elements = M·N` (true output count). After change #1 the golden `T_cube` stays **3.252 us** (real 910B3 constants) / **0.060 us** (synthetic) — verify both. The test must assert from the flops path with `flops` plainly distinct from `elements` (by 2K×), so a regression to `elements` is caught.
- **One coherent cross-path acceptance kernel with TWO distinct mandatory edges.** Extend the `matmul_vector_extract` fixture into an attention-shaped kernel (Cube→Vector→Cube: QKᵀ on Cube, softmax on Vector, ×V on Cube) so there are **two distinct mandatory handoffs** (Cube→Vector and Vector→Cube). This makes the test distinguish the Σ fix (Change #5) from the old flat cost — with two distinct edges, `T_serial_irreducible = 2 × (cycles/clock)`, not `1 ×`. Add a spreadsheet block (in the docstring) hand-deriving every `I_c`, each `O_c/I_c`, `T_core_floor`, and `T_serial_irreducible`, and assert each to **3 sig figs** (`pytest.approx(rel=1e-3)`).
- **Keep an existing single-handoff serialization test** (the current `test_serial_irreducible_from_calibration`) so the `1 edge → 1×` case is still pinned, and add a `same edge repeated across loop iterations → counted once` test (dedup behavior).
- **Grid-floor test through the real function.** Add a test that runs M2 `GridInfo` (e.g. from a K-case in the A.2 suite) → `compute_grid_floor(..., total_work=bytes)` and asserts `T_grid_floor` against a hand value — exercising the function the bypasses replaced.
- Add minimal `test_serialization.py` / `test_grid_model.py` if cleaner than overloading `test_component_model.py` (grid + serialization currently have no dedicated unit file).

---

## Open items this stage closes

| Item | Source | Closed by |
|------|--------|-----------|
| flops-consumption loop (A.3 emits/infers flops → A.4 must consume) | a3_progress "A.4 Handoff" | Change #1 |
| `repeat`/`mask` conservative no-gap default formalized as A.4 behavior | a3_progress "Known Limitations" #1 | Document: gap4 already treats `repeat=1`/`mask=0` as 100% util — confirm + note in a4_progress |
| Grid floor never wired to M2 `GridInfo` end-to-end | this analysis | Changes #2, #3, #4 (grid-floor test) |
| `compute_grid_floor` dead/bypassed | this analysis | Change #2 (+ delete bypass in `bound_from_extract`) |
| `T_serial_irreducible` flat-cost vs spec §2.2 Σ over mandatory edges | `performance_bound_model.md` §2.2/§4.0 | Change #5 (Σ over distinct edges; dedup loop repeats) |

**Do NOT claim closed:** A.1 non-blocking items (`DEFAULT_SOC_VERSION`, dead `has_vector_consumer`), L2-cache model (deferred to A.5/A.7 per `open-questions.md`), the avoidable-serial Gap-3 stub (A.5).

---

## Verification

```bash
# Python-only; no rebuild needed (A.4 is pure functions)
cd /mnt/d/work/git/vTriton
python3 -m pytest tests/perfbound/test_component_model.py -q          # golden 3-sig-fig intermediates
python3 -m pytest tests/perfbound/ -q                                  # full perfbound suite (no regressions)

# Spine check: component floor consumes flops, not elements
python3 -c "
from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord
from perfbound.extract.op_classifier import Component, Precision
from perfbound.calibration.calib_loader import load_default_calib_db
from perfbound.model.component_model import compute_component_floor_from_db
# flops != elements: matmul with flops=2MNK, elements=MN
op = OpRecord(op_id=1, op_name='matmul', component=Component.CUBE,
              precision=Precision.FP16, pipe='Cube', bytes_transferred=0,
              elements=128*64, flops=2*128*64*32, loop_multiplier=32, depends_on=[])
ex = HIVMExtract(operations=[op], handoffs=[], unit_assignment={1:'cube'})
r = compute_component_floor_from_db(ex, load_default_calib_db())
# T_cube must reflect FLOPs (2*128*64*32*32), not elements (128*64*32)
print('T_cube us =', r.per_component_us['cube'])
assert r.per_component_us['cube'] > 3.0, 'regressed to elements!'
print('OK: consumes flops')
"
```

**Acceptance gate:** `test_component_model.py` golden intermediates (`I_c`, `T_core_floor`, `T_serial_irreducible`) match the in-test spreadsheet to 3 sig figs, with `flops ≠ elements` in the fixtures; full `tests/perfbound/` green.

## Risks

| Risk | Mitigation |
|------|------------|
| Relabel keeps `flops == elements` numerically → test can't catch regression | Acceptance: assert `op.flops != op.elements` in the fixture and assert `T_cube` from the flops magnitude (2K× larger) |
| Vector blanket-swap to flops breaks the per-op path | Keep fallback FLOP/us path; pin work-unit invariant in a comment; don't enable the dead per-op path |
| Grid `total_work` unit chosen inconsistently with `i_binding` | Driver (#3) selects the pair together; never derive numerator from `GridInfo.work` |
| Scope creep into A.5 (avoidable serial, gaps) | Explicit scope boundary; A.4 = floors + classification + `T_serial_irreducible` only |
| Σ fix over-counts loop-iteration repeats of one edge → inflates `T_serial`, risks `T_bound > T_measured` (unsound) | Dedup by `(producer,consumer)` edge before summing; add the "same edge repeated → counted once" test |
| Σ fix mis-keys distinct edges as same (under-counts) | Key dedup on the component pair, not op-ids; two-mandatory-edge golden kernel asserts `2×` |
