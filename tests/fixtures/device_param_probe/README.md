# TASK-330.8 AC#1 verification fixtures

Hand-written decks (SPICE structure derived from `tests/fixtures/op_point/`'s
decks) and their captured ngspice/Xyce output, backing TASK-330.8's spike:
**"one probe spelling for a device parameter, used the same way in every
analysis"** — specifically, whether the parameters a *probe* would request
are readable **per point**, in **op, DC sweep, transient and AC**, when the
`save`/`.print` line is issued **before** the analysis runs (the shape a
probe needs, as opposed to `save_device_params`'s post-run `op1` capture —
see TASK-330.3/330.4). Nothing here is consumed by production code; the
design proposal (AC#2) and the build come in a later spawn.

Environment: `PDK_ROOT=$PDK_ROOT` (sky130A). ngspice 42 = system
`/usr/bin/ngspice`; ngspice 47 = `$SILK_NGSPICE_BIN`
(`$SILK_NGSPICE_BIN`). Xyce = `$SILK_XYCE_BIN`
(7.10.0-opensource).

## Device hierarchy

Same as `tests/fixtures/op_point/`: a `dut` subckt instanced as `xdut`,
wrapping `sky130_fd_pr__nfet_01v8`/`__pfet_01v8` as `XMmmn`/`XMmmp` — silk's
own `add_module` shape. ngspice `'@'` path:
`m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[<param>]` (three `m`s). The 17-param
list is `DEVICE_OP_PARAMS` (`silk/simulation/backend.py`): `id vgs vds vbs
vth vdsat gm gds gmbs cgg cgs cgd cgso cgdo cgbo capbd capbs`. NFET bias
Vg=1.0V Vd=1.0V Vs=Vb=0V; PFET bias (absolute) Vs=Vb=1.8V Vg=0.3V Vd=0V —
identical to `op_probe.sp`, so values cross-check against TASK-330.1's
hand-read numbers.

## Files

- `dc_sweep_nfet.sp` / `dc_sweep_pfet.sp` — `.dc` sweep of the gate
  (NFET: Vgs 0V→1.8V step 0.05V at Vds=1.0V fixed; PFET: Vg 1.8V→0V at the
  same absolute biasing as `op_probe.sp`, so Vsg goes 0V→1.8V), all 17
  params `save`d **before** `.dc` runs, then `print`ed. 37 points each.
- `op_all17_probe.sp` — op-point probe of the **full** 17-param list for
  both polarities (330.1's `op_probe.sp` only covered 12-14; this adds
  `cgbo`/`capbd`/`capbs` and completes the PFET side), `save`d before `op`.
- `ac_all17_probe.sp` — `.ac dec 10 1 1e9`, full 17-param list, both
  polarities, `save`d before the bare `.ac` (no preceding explicit `op`).
- `tran_10k_baseline.sp` / `tran_10k_with_params.sp` — 20 flat NMOS
  instances (same topology as `op_point/tran_size_{baseline,with_params}.sp`,
  scaled from `.tran 10n 10u` to **`.tran 10n 100u`** = 10,001 points), to
  measure per-probe capture cost on a *long* transient as AC#1 asks (330.1's
  own cost check used ~1000 points).
- `flat_hierarchy_check.sp` / `flat_tran_check.sp` — isolation decks (flat
  top-level `XM1`, no `dut` wrapper, matching
  `tests/test_op_point_params.py::_single_nmos_tb`'s exact geometry/bias)
  used to pin down the ngspice-42-vs-47 `cgso`/`cgdo`/`cgbo` divergence
  below to *save ordering*, not hierarchy depth.
- `xyce_dc_sweep.cir` — Xyce, `.dc vvg 0 1.8 0.05` (37 points), prints
  `N(...:GM/GDS/VTH/VDSAT)` + `ID(...)` per point. Answers the task's
  explicitly-flagged "per-point N() in .dc ... on Xyce 7.10 is unverified".
- `xyce_tran.cir` — Xyce, `.tran 1n 5n`, prints the same N()/ID() columns
  per timestep. Answers "... and .tran ... is unverified".
- `xyce_tran_10k_baseline.cir` / `xyce_tran_10k_with_params.cir` — Xyce
  20-device analogue of the ngspice tran-cost decks, `XYCE_OP_CAPABILITY`
  columns only (`id vgs vds vbs vth vdsat gm gds` — Xyce doesn't expose
  `gmbs`/capacitances, TASK-330.1/330.5). `.tran 10n 100u 0 10n` — the
  4th arg (step ceiling) was required to force ~10k actual output rows;
  without it Xyce's adaptive stepper takes a handful of large steps on
  this dynamics-free (all-DC-bias) circuit and only ~20 rows come out.
- `compare_vectors.py` — loads two rawfiles via silk's own
  `read_rawfile` (read-only; `PYTHONPATH=<worktree>` needed) and diffs
  every vector name and value the two share, to check the "no perturbation
  of the main analysis's own vectors" claim (AC#1) at full 10k-point scale.

## Committed rawfiles

Every `.raw` above except the four `tran_10k_*`/`xyce_tran_10k_*` cost
decks' outputs (8.2MB/35.5MB ngspice, 0.24MB/13MB Xyce — regenerate with
the commands below) is force-added (`git add -f`) past the repo-wide
`*.raw` gitignore rule, following `tests/fixtures/op_point/`'s precedent.

## Regenerating

```
PDK_ROOT=$PDK_ROOT /usr/bin/ngspice -b dc_sweep_nfet.sp
PDK_ROOT=$PDK_ROOT $SILK_NGSPICE_BIN -b dc_sweep_nfet.sp
PDK_ROOT=$PDK_ROOT $SILK_XYCE_BIN xyce_dc_sweep.cir
PYTHONPATH=<worktree> <worktree>/.venv/bin/python compare_vectors.py \
    ngspice42_tran_10k_baseline.raw ngspice42_tran_10k_with_params.raw
```

## AC#1 findings

### (a) Readable per point, by analysis and backend

**ngspice (both 42 and 47), pre-run `save`:**

| param(s)                                   | op | dc sweep | ac | tran |
|---------------------------------------------|:--:|:--:|:--:|:--:|
| id vgs vds vbs vth vdsat gm gds gmbs cgg cgs cgd capbd capbs (14) | ✅ | ✅ varies with sweep | ✅ constant | ✅ |
| cgso cgdo cgbo (overlap caps)                | ⚠️ version-dependent — see (c) | ⚠️ version-dependent | ⚠️ version-dependent | ⚠️ version-dependent |

DC-sweep values vary correctly with the swept bias and land exactly on
TASK-330.1's hand-read op-point numbers at the matching bias: NFET
`id`@Vgs=1.0V = `6.942150e-05` (330.1: `6.94215e-05`); NFET `gm`@Vgs=1.0V =
`3.825087e-04` (330.1: exact digit match); PFET `id`@Vg=0.3V =
`1.293831e-04` (330.1: exact digit match). AC values are constant across
all 91 frequency points at the bias-point value (id = `6.942150e-05` with
`0` imaginary part throughout `ac_all17_probe.sp`) — generalises 330.1
check (5)'s gm/vth-only finding to the full 17-param list.

**Xyce 7.10, `N()`/`ID()` in `.print`:** `id vgs vds vbs vth vdsat gm gds`
(the `XYCE_OP_CAPABILITY` 8) are readable **per point in both `.dc` sweep
and `.tran`** — previously unverified per the task description. DC-sweep
values match ngspice's DC-sweep values to the printed precision at every
checked bias point (`gm`@Vgs=1.0V = `3.825087e-04` on both backends,
`id`@Vgs=1.0V = `6.942151e-05` Xyce vs `6.942150e-05` ngspice — 6 sig
figs). `.tran` per-timestep printing of `N()`/`ID()` works with **no
refusal and no special deck shape** — Xyce just evaluates the same
solution-vector query at every accepted timestep. `gmbs` and every
capacitance remain unavailable on Xyce (confirmed again, matches
330.1/330.5) — a bad `.print` symbol still aborts the whole run before any
data is written (330.1's finding, re-confirmed unchanged).

### (b) Cost on a 10k-point transient

Real numbers, this spike's decks (not 330.1's ~1000-point ones):

| backend | baseline | +params | Δbytes | Δ ratio | Δwall time |
|---|---:|---:|---:|---:|---:|
| ngspice 42, 20 dev × 17 params, 10,001 pts | 8,250,822 B / 7.03 s | 35,492,192 B / 7.35 s | +27,241,370 B | 4.30× | +0.32 s (~5%) |
| Xyce, 20 dev × 8 params, 10,001 pts | 240,292 B / 0.65 s | 13,049,670 B / 0.71 s | +12,809,378 B | 54.3× (see note) | +0.06 s |

Xyce's ratio looks alarming but is a baseline-composition artefact, not a
backend difference: its baseline `.print` line names only 2 node voltages
(3 columns total) vs. ngspice's `save all` baseline (103 columns,
subcircuit-internal nodes included). Normalising both to **bytes per
(extra-column × point)** gives the same answer on both backends:
ngspice 27,241,370 / (340×10,001) ≈ **8.01 B/value**; Xyce 12,809,378 /
(160×10,001) ≈ **8.005 B/value** — the per-value ASCII-rawfile cost is
backend-independent; what dominates total cost is simply "how many extra
columns × how many points", exactly as 330.1's smaller-scale measurement
implied. **Wall-clock cost from the extra saves themselves is negligible
on both backends** (≤5% here) — the size cost, not the run-time cost, is
what a per-probe design needs to budget for on long transients. Confirms
330.1's check (8) scales linearly, no super-linear surprise at 10k points.

### (c) SURPRISE — cgso/cgdo/cgbo, ngspice 42 vs. 47, under pre-run `save`

**This is a genuine correction to TASK-330.1's own finding**, not a new
gap: 330.1's notes claim ngspice 42/47 output is "BYTE-IDENTICAL... except
the timestamp comment" for every check, including check (1)'s param
availability — but that check never exercised `cgbo`, and its `cgso`/
`cgdo` "0 long" reading was only spot-verified as identical across
binaries at the level of "same log text", which happened to be true for
the params it *did* test.

Direct re-test (`op_all17_probe.sp`, `dc_sweep_{n,p}fet.sp`,
`ac_all17_probe.sp`, and the isolation decks `flat_hierarchy_check.sp`/
`flat_tran_check.sp`) shows a **real, reproducible divergence** when the
`cgso`/`cgdo`/`cgbo` `save` lines are issued **before** the analysis runs
(the probe style this spike is testing — *not* `save_device_params`'s
post-run-then-`op` order, which both binaries already agree gives `nan`
for these three per `tests/test_op_point_params.py::test_op_only` and
`test_tran_ergonomics_and_op1_content`, still passing on both binaries):

- **ngspice 42**: `cgso`/`cgdo`/`cgbo` are zero-length ("not available or
  has zero length") in **every** analysis tested — op, dc sweep, ac, and
  even a genuine per-timestep read straight from `tran1` itself (not a
  subsequent `op1`) — `flat_tran_check.sp`. So on 42, a pre-run probe for
  these three gets nothing, in *any* analysis, contradicting the task
  description's "read 0 outside .tran" premise (on 42, under this save
  order, they're absent even *inside* .tran, not just outside it).
- **ngspice 47**: the *same* deck, same bias, same pre-run `save` order,
  gives real non-zero values in **every** analysis tested — op
  (`cgso=cgdo=2.463135e-16`, `cgbo=1.302418e-20`, NFET), dc sweep (same
  values, flat across the whole Vgs sweep — plausible, since sky130's
  overlap-cap model terms are only weakly bias-dependent at this
  geometry), ac (same values, constant across frequency), and tran (same
  values, constant across all 59 timesteps, since bias is DC-constant
  here). PFET values differ but are internally consistent
  (`cgbo=0.000000e+00` exactly for the PFET specifically — a real zero,
  not zero-length: no `checkvalid` warning was emitted for it).
- **Isolation result**: `flat_hierarchy_check.sp` (no `dut` wrapper, exact
  geometry/bias of the *passing* `test_op_point_params.py` benches) shows
  the identical 42-absent/47-present split — **this is not a hierarchy-
  depth effect**, it is specifically about *when* the `save @...[cgso]`
  line is issued relative to `run`. Post-run save (production's own
  order) → `nan` on both binaries (existing passing tests). Pre-run save
  (what a probe naturally needs, so ngspice populates it per-point) →
  `nan` on 42, **real values on 47**, on every analysis type.

**Implication for AC#2 (not resolved here, spike only):** the "a probe
may name any parameter the backend supports, validated before the run,
refused by name+what's-available if not" design (task Implementation
Notes) needs the "what's available" answer to be **per ngspice binary**,
not just per (backend, param, analysis) — at least for `cgso`/`cgdo`/
`cgbo` on sky130. Whether 47's non-zero values are a real physical
overlap-cap computation (plausible — the magnitudes are sane for a W=1µm
device and self-consistent across analyses) or a stale/uninitialized read
that happens to look plausible was **not** determined here (would need a
sensitivity/geometry sweep to confirm physical scaling, out of scope for
a spike); flagging as a fact-with-evidence for AC#2/the reviewer to rule
on, per this task's own "if this probe surprises, flag it" convention.

### (d) Byte-identity of the main analysis's own vectors

Confirmed at full 10k-point scale on **both** backends via
`compare_vectors.py`:

- ngspice: 103 shared vectors between `tran_10k_baseline.sp` and
  `tran_10k_with_params.sp`, **0 mismatches**, 340 extra `@` vectors added,
  0 missing → `RESULT: BYTE-IDENTICAL`.
- Xyce: 3 shared vectors (`TIME`, `V(D0)`, `V(G)`) between
  `xyce_tran_10k_baseline.cir`/`_with_params.cir`, **0 mismatches**, 160
  extra `N()`/`ID()` vectors added, 0 missing → `RESULT: BYTE-IDENTICAL`.

Requesting extra per-device saves does not perturb either simulator's
Newton solve or its existing output vectors, on real data, at the scale
the task asked for.
