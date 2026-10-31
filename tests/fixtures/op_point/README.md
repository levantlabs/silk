# TASK-330.1 verification fixtures

Hand-written decks (not silk-generated, except where noted) and their captured
ngspice/Xyce output, backing the op-table verification spike (see TASK-330.1's
notes for the full check-by-check write-up). Nothing here is consumed by
production code; later TASK-330.x subtasks may read these as reference data.

Environment: `PDK_ROOT=$PDK_ROOT` (sky130A). ngspice 42 = system
`/usr/bin/ngspice`; ngspice 47 = `$SILK_NGSPICE_BIN`
(`$SILK_NGSPICE_BIN`). Xyce = `$SILK_XYCE_BIN`
(7.10.0-opensource).

## Device hierarchy used throughout

A `dut` subckt (ports `d g s b`, or `dn gn sn bn` / `dp gp sp bp` for the
two-device deck) instantiated as `xdut` at top level, wrapping one
`sky130_fd_pr__nfet_01v8`/`__pfet_01v8` placed as `XMmmn`/`XMmmp` — the
same `<DUT-instance>.<X-element>` shape silk's own `Module.add_module`
emits (confirmed against a real generated deck before hand-writing these).
ngspice's internal hierarchical name for the '@' vectors is
`m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[<param>]` — note **three** `m`s
(`x` + `M` + `mmn` all lowercased), not two; the PDK's own `.pm3.spice`
model card names the raw M-element inside the device subckt
`msky130_fd_pr__nfet_01v8`/`__pfet_01v8` (grep
`libs.ref/sky130_fd_pr/spice/sky130_fd_pr__nfet_01v8__tt.pm3.spice`).

## Files

- `op_probe.sp` / `ngspice{42,47}_op_probe.{log,raw}` — checks (1)-(4), (6)
  partially: one NFET (Vgs=1.0V Vds=1.0V) and one PFET (Vg=0.3V Vd=0V,
  Vs=Vb=1.8V, i.e. the absolute biasing the task specifies) in one deck,
  `save`s the 14 BSIM4 op-vars per device individually, runs `.op`
  (interactive `op`+`run`), then both `show ... : <params>` (device-model
  query path) and `print @m...[param]` (vector-read path) for every value.
- `op_probe_ac.sp` / `ngspice{42,47}_op_probe_ac.{log,raw}` — check (5):
  NFET only. Runs a bare `.ac dec 10 1 1e9` first (no preceding `op`),
  prints the `@` vectors from that sweep, then an explicit `op`, then a
  second `.ac` — `setplot` at the end lists every retained plot.
- `op_probe_all_form.sp` / `ngspice{42,47}_op_probe_all_form.log` — check
  (6): tries `save @m....[all]` as a wildcard short form. No `.raw` — see
  the note below.
- `op_probe_ammeter.sp` / `ngspice{42,47}_op_probe_ammeter.{log,raw}` —
  check (7): **silk-generated** (`Module(..., save_device_currents=True,
  terminal_current_method='ammeter_wrapper')`, via `add_module` into a
  `dut` subckt, one `Transient` analysis so `generate_netlist()` doesn't
  refuse), with a hand-appended `.control` block probing the extra
  `.xmdev` hierarchy segment the wrapper subckt inserts.
- `tran_size_baseline.sp` / `tran_size_with_params.sp` +
  `ngspice{42,47}_tran_size_{baseline,with_params}.{log,raw}` — check (8):
  20 flat NMOS instances (**silk-generated**, `Module.generate_netlist()`
  with no DUT wrapper), one `.tran 10n 10u` (~1000 points). The
  `_with_params` deck adds one `save @m.xmmm<i>.msky130_fd_pr__nfet_01v8[<p>]`
  line per (device, param) pair for the same 17-param list TASK-330's
  design note gives (`id vgs vds vbs vth vdsat gm gds gmbs cgg cgs cgd
  cgso cgdo cgbo capbd capbs`) x 20 devices = 340 lines; the baseline has
  none. Same bias, same `.tran`, only the `save` lines differ.
- `xyce_probe.cir` — **silk-generated** (`Module(simulator='xyce')`, one
  NFET DUT, `analysis.DC` single-point sweep), used as the base deck for
  every Xyce check below (only the `.print` line changes between copies).
- `xyce_names.txt` / `xyce_namesfile.log` — `Xyce -namesfile` output.
- `xyce_probe_available.cir/.log/.raw` — every `N()`/`ID()` variable that
  worked, in one `.print dc` line.
- `xyce_probe_unavailable.cir/.log` — `N(...:CGG)` and `N(...:GMBS)` in one
  `.print dc` line, to capture the exact "Function or variable" refusal
  text. No `.raw`: Xyce aborts before simulating when a `.print` symbol is
  undefined.

## Known quirk: `op_probe_all_form.sp` writes no rawfile

With the invalid `save @m...[all]` line present, ngspice runs `op`/`run`/
`display` fine (no error, only the "not available or has zero length"
warning) but a subsequent `write <file>.raw` silently produces no file and
no error message either — confirmed in isolation: the exact same
`op`/`run`/`write` sequence *without* the invalid `save` line writes a
rawfile normally. Not investigated further (out of scope for this spike);
the check (6) evidence is the warning text in the log, not a rawfile.

## Regenerating

Each `.sp`/`.cir` file is self-contained (absolute `PDK_ROOT`-relative
`.lib`/`.include` paths baked in by the generator at write time — these
fixtures are not portable to a different `PDK_ROOT`). Re-run with e.g.:

```
PDK_ROOT=$PDK_ROOT /usr/bin/ngspice -b op_probe.sp
PDK_ROOT=$PDK_ROOT $SILK_NGSPICE_BIN -b op_probe.sp
PDK_ROOT=$PDK_ROOT $SILK_XYCE_BIN xyce_probe_available.cir
```

## Committed rawfiles

`*.raw` is gitignored repo-wide, so the small probe rawfiles above were
force-added (`git add -f`) at the TASK-330.1 gate. The two
`tran_size_*` rawfiles (0.8 MB and 3.6 MB) are NOT committed: their logs
record the sizes, and regenerating them takes one ngspice run of the
committed deck.

## TASK-330.3 probe: the post-run save + control-mode 'op' placement

Answers reviewer stop-point ruling 1's five questions (task notes), on
real ngspice 42 AND 47 (every `ngspice42_*`/`ngspice47_*` pair below is
byte-structurally identical across both versions — same plot names,
point counts, vector counts, '@'-vector counts; no warnings on any
successful run). Decks (`*.sp`) are hand-written, not silk-generated,
same `dut`/`xdut` device hierarchy as the TASK-330.1 fixtures above.

**The uniform `.control` sequence that works, on every analysis kind**
(`tran_op_after17.sp`, `ac_op_after.sp`, `dc_op_after.sp`,
`op_only_via_sequence.sp`):

```
save all
run
save @m.<path>.<leaf>[<param1>]
save @m.<path>.<leaf>[<param2>]
...
op
set wr_singlescale
set wr_vecnames
option numdgt=3
write <rawfile> <main-plot>.all op1.all
```

i.e. the main declared analysis runs FIRST, with no `'@'` saves active —
only AFTER that `run` do the device-param `'@'` saves get added, followed
by one interactive `op` command, then `write` names both the main
analysis's plot and `op1`. Q1-3 answers, all confirmed on both ngspice
versions:

1. **Vector counts per plot** (`tran_op_after17.raw`, all 17 fixed
   params saved): `tran1` has 8 vectors, **0** `'@'` vectors, 59 points.
   `op1` has 24 vectors, **all 17** `'@'` vectors, 1 point. Same shape
   for `ac_op_after.raw` (`ac1`: 8 vectors/0 `'@'`, 91 points; `op1`: 10
   vectors/3 `'@'`, 1 point — only `id`/`gm`/`vth` saved in that deck to
   keep it short) and `dc_op_after.raw` (`dc1`: 8/0, 10 points; `op1`:
   10/3, 1 point).
2. **`tran1`/`ac1`/`dc1` vs switch-off baseline**: identical variable
   SET and identical data (`np.array_equal`/`np.allclose`, exact match,
   every vector) against `tran_baseline_off.raw`/`ac_baseline_off.raw`/
   `dc_baseline_off.raw` (same bench, no `'@'` saves at all). No dropped
   defaults, no contamination — confirmed on both ngspice versions.
3. **Plot name and order**: the post-run interactive `op` produces a
   plot named `Operating Point` (`op1` in ngspice's own listing), reading
   as `'op'` via the ALREADY-MERGED `select_plot`/`plot_silk_analysis`
   (TASK-330.2) with no changes needed. Plot order in the rawfile follows
   the ORDER NAMED IN THE `write` COMMAND, not analysis/solve order —
   verified by reversing the argument order (`write ... op1.all tran1.all`
   vs `... tran1.all op1.all`) and observing `raw.plots[0]` change
   accordingly. **`select_plot(raw, None)` (the "test declares one
   analysis, no `analysis=` needed" ergonomic path) now RAISES
   `SimulationError`** once `save_device_params` adds the second `op1`
   plot to an otherwise single-analysis test's rawfile — confirmed
   directly against `tran_op_after17.raw` in a real Python session. This
   is not a bug (multi-plot files already require naming an analysis,
   TASK-212) but IS a real ergonomics change: turning the switch on
   requires every existing `probe()`/`load_signal()` call in that test to
   start passing `analysis=<word>` explicitly, even in a test that only
   ever declared one analysis.

   **Resolved under reviewer stop-point ruling 2, condition 2**:
   `Module.load_signal` now defaults an omitted `analysis` to the test's
   one declared NON-op analysis whenever `save_device_params` is on —
   never to `'op'` itself, which stays reachable only by naming it. See
   `silk/circuit/modules_results.py`'s `load_signal` and
   `tests/test_op_point_params.py`.

**Q4 (AC bench and DC() sweep)**: see (1)/(2) above — `ac_op_after.raw`/
`dc_op_after.raw` and their baselines prove the identical result for
both analysis kinds, not just `.tran`.

**Q5 (Op() declared alongside another analysis)** — this is where the
naive extension of the above sequence breaks, on both ngspice versions:

- `tran_plus_op_naive.sp`: netlist declares BOTH `.tran` and `.op` cards
  (what TASK-330.2's existing `Op()` emission already does for a
  `Transient()+Op()` bench), `'@'` saves added BEFORE `run` (the
  REJECTED placement). Result: `tran1` gets **3** `'@'` vectors at every
  one of its 59 points — the exact contamination the ruling forbids,
  reproduced even though `Op()` was explicitly declared (not
  auto-inserted).
- `tran_plus_op_fixed.sp` **(revised under reviewer stop-point ruling 2,
  condition 1 — its FIRST version, kept only in git history, still had
  the netlist-level `.op` card and demonstrated the bug above, not a
  fix, despite the filename)**: `Op()` declared alongside `Transient()`,
  but with NO netlist-level `.op` card at all — only `.tran`. `'@'`
  saves are added AFTER `run`, then the interactive `op`. Result on both
  ngspice 42 and 47: exactly two plots, `tran1` (59 points, **0** `'@'`
  vectors — byte-identical in shape to the switch-off baseline) and
  `op1` (1 point, every `'@'` vector active, no `op2`). Confirmed by
  reading both plots back with `silk.signals.rawfile.read_rawfile` and
  counting `'@'`-prefixed vector names directly. This is the same fix
  `op_only_via_sequence.sp` below proves for the `Op()`-only case —
  applied uniformly, a pre-existing `.op` card is simply never emitted
  when `save_device_params` is on, so there is nothing left to race the
  interactive `op` for the `op1` slot.
- `op_only_via_sequence.sp`: proves the fix — when `Op()` is the ONLY
  declared analysis, dropping the netlist-level `.op` card ENTIRELY and
  using the exact same uniform post-run sequence (`save all` / `run` /
  `'@'` saves / `op` / `write ... op1.all`) produces a single, correctly
  populated `op1` — no dual-plot ambiguity, and no special-casing needed
  for "`Op()` is the only analysis" the way the ruling anticipated.

  **Revised recommendation** (supersedes the implementer proposal's
  placement (b)): whenever `save_device_params` is on, NEVER emit a
  netlist-level `.op` card — not the auto-inserted one for AC/DC/tran
  benches, and not the one `Op()` itself would normally emit when the
  user declares it directly (alone or alongside another analysis). The
  interactive post-run `save`+`op` sequence is the SOLE producer of the
  op-point plot in every case, uniformly. This keeps `select_plot(raw,
  'op')` unambiguous (condition (2) of ruling 1) by construction — there
  is only ever one `'Operating Point'`-named plot in the file, `op1`,
  regardless of how many analyses (including `Op()` itself) the test
  declares.

## TASK-330.3 probe: ammeter_wrapper combination

`tran_ammeter_noSC_withparams.sp` reuses the exact `amm_sky130_fd_pr__
nfet_01v8` wrapper subckt and `xdut.xmmm1.xmdev` hierarchy from
`op_probe_ammeter.sp` (TASK-330.1 check 7), on a real `.tran` bench, with
the uniform post-run save+op sequence and NO `.options savecurrents`.
Result (both ngspice versions): `tran1` clean (0 `'@'` vectors, 18
vectors total incl. the wrapper's ammeter branch currents), `op1` has
all 3 saved params (`id`/`gm`/`vth`) at the CORRECT path
`@m.xdut.xmmm1.xmdev.msky130_fd_pr__nfet_01v8[<param>]` — confirming the
extra `.xmdev` segment resolves through this seam with no ammeter-
specific branch needed, condition (1) of ruling 1.

**Critical finding: `save_device_params` and `save_device_currents` are
structurally incompatible, regardless of `terminal_current_method`.**
`tran_ammeter_op_after.sp` and `tran_ammeter_baseline_off.sp` (identical
device topology, WITH `.options savecurrents`, one with device-param
saves and one WITHOUT any `'@'` save at all) both FAIL to produce a
rawfile on both ngspice versions:
`Warning from checkvalid: vector @m.xdut.xmmm1.xmdev.msky130_fd_pr__
nfet_01v8[ib] is not available or has zero length.`, followed by `write`
producing no file. This is not new: it is the EXACT bug
`spice_netlister.py`'s `generate_output_block` docstring already
documents (`"tolerates the zero-length @m…[ib] vectors that savecurrents
registers and that the .control write aborts on"`) as the reason
`save_device_currents=True` unconditionally forces **batch-write mode**
(deck-level `.save all`, no `.control` block at all) — confirmed by
reading that function: the batch-write branch is gated purely on the
`save_device_currents` boolean, independent of `terminal_current_method`.
Batch-write mode has NO `.control` block, so there is no insertion point
for `'@'` saves or an interactive `op` at all. Since `.options
savecurrents` (gated on the same boolean, `generate_complete_netlist`
around line 812) is what registers the zero-length `[ib]` vector that
breaks a `.control`-block `write` in the first place, forcing a
`.control` block back on for this combination would not fix anything —
it would reintroduce the exact bug batch-write mode exists to dodge.

**Recommendation**: `save_device_params=True` combined with
`save_device_currents=True` (any `terminal_current_method`) should
RAISE at generation time with a clear error, mirroring the EXISTING
`set_save_list` + `save_device_currents` incompatibility
(`spice_netlister.py` `generate_output_block`, `save_list is not None
and save_device_currents`). `terminal_current_method='ammeter_wrapper'`
alone (without `save_device_currents=True`) is unaffected by this and
already proven to work via `tran_ammeter_noSC_withparams.sp` above — the
`.xmdev` path segment comes from the wrapper's structural presence in
the netlist, not from `save_device_currents`/`.options savecurrents`.

## TASK-330.5 probe: gf180mcu on Xyce (ruling C)

`gf180mcu/xyce_probe_gf180mcu_nmos3p3.cir/.log` and
`gf180mcu/xyce_probe_gf180mcu_pmos3p3.cir/.log` — real Xyce 7.10 runs
testing whether gf180mcu's `nmos3p3`/`pmos3p3` op-point columns are
reachable via `N(<path>:PARAM)`/`ID(<path>)`, using the flat-instance
form (`XMM1`, no `dut` wrapper, mirroring `gf180mcu/nmos3p3_op_params.sp`)
with `op_internal_name`'s declared leaf `m0` (i.e. `N(XMM1:M0:GM)`,
`ID(XMM1:M0)`), same 8-name known-good set as the sky130 probe.

**Result: gf180mcu cannot be simulated on Xyce 7.10 at all — a
PDK-library parse failure, not an op-column gap.** Both runs abort
identically (same error set, before any device-specific processing) at
`.lib "sm141064.ngspice" typical`: Xyce's netlist parser cannot handle
the file's nested `.LIB`/`.ENDL` structure (`.ENDL encountered without
library name`, `.ENDL encountered with name DIODE, which does not match
.LIB name DIO`) and two of its nested sections `.lib` in turn missing
include files ngspice tolerates (`Could not find include file
statistical`, `Could not find include file noise_corner`), ending in
unresolved `.PARAM` references that depend on those missing sections
(`RSH_PPLUS_U_M`, `RSH_NPLUS_U_M`). No `.raw` is produced by either run
(exit 1, `*** Xyce Abort ***`) — same "one bad symbol/one bad file
aborts the whole run, no partial output" behavior TASK-330.1 already
found for a single undefined `.print` column, just triggered earlier, at
library-load time. No existing silk test exercises `simulator='xyce'`
with `pdk='gf180mcu'` (grepped; none found), so this is a pre-existing,
previously-unverified gf180mcu/Xyce incompatibility at the whole-PDK
level, not something introduced by or fixable within TASK-330.5's scope.

**Conclusion for TASK-330.5**: gf180mcu's `op_backends` stays
`(ngspice,)` — `xyce` is not added. sky130 is the only PDK verified on
Xyce (per the pre-existing `xyce_probe_available.cir` evidence above);
sg13g2 was never declared for op capture on any backend.
