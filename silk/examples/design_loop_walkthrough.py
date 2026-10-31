"""The silk design loop, end to end — system intent to layout-aware netlist.

This is the one walkthrough to read if you want to know why silk is more
than Python syntax for SPICE. One design object — a sampling buffer built
from a 5-transistor OTA plus its bias mirror — carries the whole loop:

  1. **Intent → sizing.** A system-level statement of what the buffer must
     do (two load contexts, settling, gain, bandwidth) is turned into a
     tail-current budget and device widths by a first-order design rule.
  2. **Build.** A parameterized factory produces the circuit with typed
     ports and roles, and registers four uniquely named characterization
     tests on it — each with its own metrics and first-class specs.
  3. **Characterize.** Three of those tests run against ngspice under two
     distinct *characterization contexts* (different declared load, different
     temperature and supply) plus an open-loop AC sweep.
  4. **Report + persist.** Metrics, spec measurements and per-net currents
     are stored in silk's SQLite database, scoped to the profile and PVT point
     that produced them; a pre-layout report is written as JSON and HTML.
  5. **Reopen.** The database is opened as a bare file — no live circuit —
     and asked which profiles exist, which current sets they selected, and
     which circuit version originated each.
  6. **Rebuild + restore.** The same factory rebuilds the design in a fresh
     process-equivalent session (never simulated); the rebuild resolves to the
     same sealed version and one exact context is restored onto it, with
     profile, current-set and origin provenance in hand.
  7. **Layout feedback.** The restored currents size the routing (EM width
     per net on met1) through silk's layout *estimator* — a placement/routing
     model, not extracted GDS — the estimated parasitics are injected back
     into the design, a post-layout AC test runs against them, and the
     before/after reports are diffed so the bandwidth cost of routing is
     visible. (Real-GDS extraction is the post-0.2.0 arc; "post-layout"
     below means post-*estimated*-layout throughout.)
  8. **Fail loud.** A rebuilt design with a changed knob, and the post-layout
     design itself, both ask for the pre-layout evidence and are refused with
     the documented typed error — nothing is guessed, nothing is half-applied.
  9. **Netlist.** The final SPICE netlist is generated from that same design
     object, parasitics included.

Every step uses only shipped public APIs; the example is the glue.

Run::

    python -m silk.examples.design_loop_walkthrough [OUT_DIR]

Needs ngspice on ``PATH`` and the sky130 PDK at ``$PDK_ROOT``. Artifacts land
in ``OUT_DIR`` (default: ``silk-output/design_loop/`` under the working directory,
overridden by the ``SILK_EXAMPLES_OUT`` setting).

Nothing below is sky130-specific except one record: :class:`Technology`
names the PDK, its core devices, its corner spelling and its width grid, and
every step takes it as ``tech``. ``design_loop_walkthrough_gf180mcu.py`` runs
this same code on GF180MCU — a PDK whose models want W/L in *metres* — by
passing a different ``Technology`` and a 3.3 V intent, which is the proof that
no step assumes sky130 microns.
"""
import math
import os
import sys
from dataclasses import dataclass

from silk import load_pdk, sources
from silk.circuit.modules import Module
from silk.circuit.ports import Port
from silk.circuit.report import generate_report
from silk.circuit.report_diff import diff_reports
from silk.db import Database, EvidenceMismatchError
from silk.elements import elements
from silk.layout import RoutingModel, estimate_layout, inject_parasitics
from silk.metrics.waveform import final_value
from silk.metrics.timing import settling_time
from silk.metrics import bandwidth_3dB, dc_gain_dB
from silk.simulation import AC, Transient
from silk.examples import example_output_dir

_PDK_CACHE = {}


@dataclass(frozen=True)
class Technology:
    """The facts about one PDK this walkthrough needs — and nothing else.

    Everything a step does with the PDK goes through this record, so running
    the loop on another technology is a matter of data, not code. ``pdk`` is
    the registry name ``load_pdk`` resolves; ``nmos``/``pmos`` are the factory
    attributes of the core devices; ``corner`` is *the PDK's own spelling* of
    its typical corner (sky130 says ``tt``, GF180MCU says ``typical``);
    ``width_grid_um`` is the manufacturing grid device widths snap to. Units
    are deliberately absent: the author writes microns on every silk PDK
    (manifest ``user_units: u``) and the PDK converts.
    """
    pdk: str
    label: str            # the Module's ``pdk_name`` tag
    nmos: str
    pmos: str
    corner: str
    width_grid_um: float


def get_pdk(tech=None):
    """The technology's PDK, loaded on first use — never at import (see
    ``ota_5t_guide``)."""
    tech = tech or SKY130
    if tech.pdk not in _PDK_CACHE:
        _PDK_CACHE[tech.pdk] = load_pdk(tech.pdk)
    return _PDK_CACHE[tech.pdk]


# --------------------------------------------------------------------------- #
# 1. System intent — what the block must do, in the system's own terms
# --------------------------------------------------------------------------- #
#: The system owns these numbers. Nothing below is a transistor size.
INTENT = dict(
    vdd_v=1.8,
    step_v=0.05,                 # small-signal input step for settling tests
    light_load_f=20e-15,         # driving the next on-chip gate
    heavy_load_f=1e-12,          # driving a sampling capacitor
    gain_db_min=26.0,            # open-loop DC gain, light load
    bw_hz_min=100e6,             # open-loop -3 dB bandwidth, light load
    settle_light_s_max=2e-9,     # 1 % settling, light load, nominal PVT
    settle_heavy_s_max=6e-9,     # 1 % settling, heavy load, hot + low supply
    hot_temperature_c=85.0,
    hot_vdd_v=1.62,              # -10 % supply
    nominal_temperature_c=27.0,
)

#: First-order design rules that turn intent into sizes. These are stated
#: assumptions, not silk API — the characterization below is what checks them.
GM_OVER_ID = 10.0          # 1/V — moderate inversion for the input pair
SIZING_MARGIN = 2.0        # covers the hot/low-supply context and mirror loss
J_INPUT_UA_PER_UM = 10.0   # drain current density for M1/M2 at L=0.5 µm
J_LOAD_UA_PER_UM = 10.0    # for the PMOS mirror load M3/M4 at L=0.5 µm
J_TAIL_UA_PER_UM = 12.5    # for the tail device M5 at L=1.0 µm
MIRROR_RATIO = 8           # M5 : M6 width ratio (tail : reference)
L_INPUT_UM, L_LOAD_UM, L_TAIL_UM = 0.5, 0.5, 1.0
WIDTH_GRID_UM = 0.005      # sky130 manufacturing grid for w/l

#: The technology this file runs on by default. See ``Technology``.
SKY130 = Technology(pdk='sky130', label='PDK_skywater130', nmos='nmos1p8',
                    pmos='pmos1p8', corner='tt', width_grid_um=WIDTH_GRID_UM)

#: met1 constants for EM/routing sizing (same convention as
#: ``net_current_persistence_example.py``; cap density is the fringe-inclusive
#: near-min-width value argued in ``tests/layout/test_parasitics_effect.py``).
MET1 = RoutingModel(sheet_resistance=0.125, cap_per_um2=1.0,
                    j_max=0.78e-3, min_width=0.14, max_width=10.0)

#: Test names (unique on the module) and the metric/spec names they declare.
T_STEP_LIGHT = 'step_light_load'
T_STEP_HEAVY = 'step_heavy_load'
T_AC_PRE = 'ac_open_loop'
T_AC_POST = 'ac_post_layout'


def _snap(width_um, grid=WIDTH_GRID_UM):
    return round(round(width_um / grid) * grid, 3)


def size_from_intent(intent=INTENT, tech=SKY130):
    """Turn the system intent into a tail-current budget and device widths.

    A unity-gain follower settles to 1 % in ``ln(100) ≈ 4.6`` closed-loop time
    constants, and its closed-loop bandwidth is ``gm / (2π C_load)`` for a
    single-pole OTA. The heavy-load settling requirement therefore fixes the
    minimum transconductance; ``GM_OVER_ID`` turns that into a per-side drain
    current, and ``SIZING_MARGIN`` covers what the first-order model ignores
    (the hot/low-supply context and current-mirror loss — both of which the
    characterization then measures). Widths follow from current-density rules.
    """
    tau_s = intent['settle_heavy_s_max'] / math.log(100.0)
    gm_min = intent['heavy_load_f'] / tau_s                  # 2π f_cl C_load
    id_side = gm_min / GM_OVER_ID
    i_tail = SIZING_MARGIN * 2.0 * id_side
    grid = tech.width_grid_um
    w_tail = _snap(i_tail * 1e6 / J_TAIL_UA_PER_UM, grid)
    return dict(
        gm_min_s=gm_min,
        i_tail_a=i_tail,
        i_ref_a=i_tail / MIRROR_RATIO,
        w_in_um=_snap((i_tail / 2.0) * 1e6 / J_INPUT_UA_PER_UM, grid),
        w_load_um=_snap((i_tail / 2.0) * 1e6 / J_LOAD_UA_PER_UM, grid),
        w_tail_um=w_tail,
        w_ref_um=_snap(w_tail / MIRROR_RATIO, grid),
    )


# --------------------------------------------------------------------------- #
# 2. The parameterized design: factory + embedded characterization suite
# --------------------------------------------------------------------------- #
def _settling_metric(step_v):
    """1 %-of-step settling time of ``vout`` after the input step.

    The band is 1 % of the *step*'s swing, not 1 % of the output level — a
    9 mV band on a 0.9 V output would be 18 % of a 50 mV step and would call
    almost anything "settled". ``settling_time`` normalises by the swing
    itself (``|target - initial|``), so this reads the step's own size off
    the closure rather than working around a level-relative band. Bound as
    a default argument (not a closure over a loop variable) — see the agent
    guide's late-binding gotcha.

    This one stays a ``ckt`` extractor rather than moving to the probe form:
    it reads the step's final value to derive ``initial``, which is more
    than loading a signal and calling a metric, and the function form has no
    slot for that. It is also deliberately SHARED by the two step
    benches — a function can be, where a probe cannot, because a probe
    belongs to the bench it was named on.
    """
    def _settle(ckt, _step=step_v):
        sig = ckt.load_signal('vout')
        return settling_time(sig, initial=final_value(sig) - _step, tol=0.01,
                             start=_STEP_T)
    return _settle


_STEP_T = 100e-9          # input-step edge time inside the transient tests
_STEP_STOP = 300e-9
_CURRENT_TSTART = 50e-9   # currents averaged from here (past the DC settle)


def build_sampling_buffer(knobs, *, tech=SKY130, intent=INTENT, db_path=None,
                          work_dir=None, specs=None):
    """Build the sampling buffer and register its characterization suite.

    Returns ``(module, tests)``. The module carries typed ports with roles,
    six transistors of ``tech`` sized from ``knobs`` (see ``size_from_intent``),
    and four uniquely named tests — each declaring its metrics and specs —
    that are **declared but not yet wired**. Tests retain the live DUT;
    declare the full spec set before sealing its database identity.
    ``wire_pre_layout_tests`` wires three tests; the fourth
    (``ac_post_layout``) is wired after parasitics are injected.
    Pass a list as ``specs`` to retain the declared Spec handles.
    """
    specs = [] if specs is None else specs
    work_dir = work_dir or example_output_dir('design_loop', 'work')
    pdk = get_pdk(tech)
    nmos, pmos = getattr(pdk, tech.nmos), getattr(pdk, tech.pmos)
    mod = Module(tech.label, work_dir, 'sampling_buffer.cir',
                 pdk_details=pdk, module_name='sampling_buffer', db_path=db_path)
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VINP': Port.analog_input(), 'VINM': Port.analog_input(),
        'VOUT': Port.analog_output(),
        'IBIAS': Port.analog_input(),        # reference current in
    })
    for net in ('vint', 'tail'):
        mod.add_net(net)
    with mod:
        # Input pair — VINP drains into the mirror diode, VINM into VOUT.
        nmos('M1', d='vint', g='VINP', s='tail', b='VSS',
                    w=knobs['w_in_um'], l=L_INPUT_UM)
        nmos('M2', d='VOUT', g='VINM', s='tail', b='VSS',
                    w=knobs['w_in_um'], l=L_INPUT_UM)
        # PMOS current-mirror load.
        pmos('M3', d='vint', g='vint', s='VDD', b='VDD',
                    w=knobs['w_load_um'], l=L_LOAD_UM)
        pmos('M4', d='VOUT', g='vint', s='VDD', b='VDD',
                    w=knobs['w_load_um'], l=L_LOAD_UM)
        # Tail source mirrored from the IBIAS reference diode (ratio 8:1).
        nmos('M5', d='tail', g='IBIAS', s='VSS', b='VSS',
                    w=knobs['w_tail_um'], l=L_TAIL_UM)
        nmos('M6', d='IBIAS', g='IBIAS', s='VSS', b='VSS',
                    w=knobs['w_ref_um'], l=L_TAIL_UM)
    # Read all four MOS terminals through 0 V ammeters when a transient test
    # opts into device currents (``run_test`` below toggles the opt-in).
    mod.terminal_current_method = 'ammeter_wrapper'

    tests = {}
    settle = _settling_metric(intent['step_v'])

    t = mod.create_test(T_STEP_LIGHT, parameters={
        'cload_f': intent['light_load_f'], 'step_v': intent['step_v']})
    t.add_metric('t_settle_light_s', settle)
    specs.append(t.add_spec(f"settles in {intent['settle_light_s_max'] * 1e9:.0f} ns into the "
               "light load", 't_settle_light_s', '<',
               intent['settle_light_s_max'], key='settle_light'))
    tests[T_STEP_LIGHT] = t

    t = mod.create_test(T_STEP_HEAVY, parameters={
        'cload_f': intent['heavy_load_f'], 'step_v': intent['step_v']})
    t.add_metric('t_settle_heavy_s', settle)
    specs.append(t.add_spec(f"settles in {intent['settle_heavy_s_max'] * 1e9:.0f} ns into the "
               "sampling cap (hot, -10 % VDD)",
               't_settle_heavy_s', '<', intent['settle_heavy_s_max'],
               key='settle_heavy'))
    tests[T_STEP_HEAVY] = t

    t = mod.create_test(T_AC_PRE, parameters={'cload_f': intent['light_load_f']})
    # The taught program (docs/probes.md): name the
    # signal once with tb.probe, hand that Signal to each registered metric,
    # run, then plot the same probe. A probe belongs to ONE bench, so the
    # post-layout test below declares its own even though both measure
    # 'vout' -- what two benches can share is the metric FUNCTION, never the
    # signal.
    vout_pre = t.probe('vout')
    t.add_metric('gain_dB', dc_gain_dB, sig=vout_pre)
    t.add_metric('bw_hz', bandwidth_3dB, sig=vout_pre)
    specs.append(t.add_spec(f"DC gain >= {intent['gain_db_min']:.0f} dB", 'gain_dB', '>=',
               intent['gain_db_min'], key='gain'))
    specs.append(t.add_spec(f"open-loop BW >= {intent['bw_hz_min'] / 1e6:.0f} MHz (pre-layout)",
               'bw_hz', '>=', intent['bw_hz_min'], key='bw_pre_layout'))
    tests[T_AC_PRE] = t

    t = mod.create_test(T_AC_POST, parameters={'cload_f': intent['light_load_f']})
    t.add_metric('bw_post_layout_hz', bandwidth_3dB, sig=t.probe('vout'))
    specs.append(t.add_spec(f"open-loop BW >= {intent['bw_hz_min'] / 1e6:.0f} MHz (post-layout)",
               'bw_post_layout_hz', '>=', intent['bw_hz_min'], key='bw_post_layout'))
    tests[T_AC_POST] = t
    return mod, tests


_PORT_MAP_OPEN_LOOP = {'VDD': 'VDD', 'VSS': '0', 'VINP': 'vinp', 'VINM': 'vinm',
                       'VOUT': 'vout', 'IBIAS': 'ibias'}
_PORT_MAP_FOLLOWER = {'VDD': 'VDD', 'VSS': '0', 'VINP': 'vinp', 'VINM': 'vout',
                      'VOUT': 'vout', 'IBIAS': 'ibias'}      # VINM ↔ VOUT


def _rig(tb, *, vdd, iref):
    """Supplies shared by every testbench: VDD and the IBIAS reference."""
    tb.add_source(sources.VoltageSource('vdd', positive='VDD', negative='0', dc=vdd))
    tb.add_source(sources.CurrentSource('iref', positive='VDD', negative='ibias',
                                        dc=iref))


def wire_ac_test(tb, *, cload, vdd, iref):
    """Open-loop AC sweep with a 1 V AC drive on VINP and ``cload`` on VOUT.

    200 points per decade: ``bandwidth_3dB`` reports the last grid point
    above -3 dB without interpolating, so the sweep grid *is* the metric's
    resolution (1.2 % per point here; 20/decade would be 12 % and would hide
    the routing-parasitic effect measured in step 7).
    """
    tb.add_module_under_test(_PORT_MAP_OPEN_LOOP)
    _rig(tb, vdd=vdd, iref=iref)
    tb.add_source(sources.VoltageSource('vinp', positive='vinp', negative='0',
                                        dc=vdd / 2, ac=1.0))
    tb.add_source(sources.VoltageSource('vinm', positive='vinm', negative='0',
                                        dc=vdd / 2))
    tb.add(elements.capacitor('cload', p='vout', n='0', value=cload))
    tb.add_sim_type(AC(variation='dec', points=200, fstart=1, fstop=1e10))


def wire_step_test(tb, *, cload, vdd, iref, step_v):
    """Unity-gain follower (VINM tied to VOUT) with a ``step_v`` input step."""
    tb.add_module_under_test(_PORT_MAP_FOLLOWER)
    _rig(tb, vdd=vdd, iref=iref)
    lo, hi = vdd / 2, vdd / 2 + step_v
    tb.add_source(sources.VoltageSource(
        'vinp', positive='vinp', negative='0',
        pulse=sources.Pulse(v1=lo, v2=hi, td=_STEP_T, tr=0.2e-9, tf=0.2e-9,
                            pw=2 * _STEP_STOP, per=4 * _STEP_STOP)))
    tb.add(elements.capacitor('cload', p='vout', n='0', value=cload))
    tb.add_sim_type(Transient(step=0.05e-9, stop=_STEP_STOP))


def wire_pre_layout_tests(tests, knobs, intent=INTENT):
    """Wire the three pre-layout tests (declared by the factory)."""
    wire_step_test(tests[T_STEP_LIGHT], cload=intent['light_load_f'],
                   vdd=intent['vdd_v'], iref=knobs['i_ref_a'],
                   step_v=intent['step_v'])
    wire_step_test(tests[T_STEP_HEAVY], cload=intent['heavy_load_f'],
                   vdd=intent['hot_vdd_v'], iref=knobs['i_ref_a'],
                   step_v=intent['step_v'])
    wire_ac_test(tests[T_AC_PRE], cload=intent['light_load_f'],
                 vdd=intent['vdd_v'], iref=knobs['i_ref_a'])


def run_test(test, *, corner, temperature, voltage, currents):
    """Run one test at an explicit characterization point.

    ``corner`` is the PDK's own spelling (``Technology.corner``).
    ``temperature`` and ``voltage`` are applied with the testbench's
    ``set_temperature()`` and ``set_supply_voltage()``, replacing the previous
    values on repeated runs. Supply sources are wired from the same intent by
    ``wire_pre_layout_tests`` (or ``wire_ac_test`` after layout); keep that
    wiring consistent with ``voltage``, and do not add a conflicting custom
    ``.temp`` line.

    The core derives verified temperature and supply from execution, so the run
    is scoped with a ``CharacterizationContext`` and everything stored from it
    lands under the profile (owner, test name, declared parameters) at exactly
    that PVT point.
    Device-current capture is a transient-only opt-in on the design object
    (silk refuses it on an AC deck), so it is switched per run.
    """
    test.module.save_device_currents = currents
    test.set_temperature(temperature)
    test.set_supply_voltage(voltage)
    return test.run(corner=corner)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _h(title):
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


def _short(uuid):
    return f'{uuid[:8]}…'


def _ua(a):
    return '—' if a is None else f'{a * 1e6:7.1f} µA'


# --------------------------------------------------------------------------- #
# The walkthrough
# --------------------------------------------------------------------------- #
def characterize_and_store(mod, tests, knobs, db_path, out_dir, *, tech=SKY130,
                           intent=INTENT):
    """Session 1: seal, characterize under two contexts + AC, store, report."""
    corner = tech.corner
    version_id = mod.create_circuit_version('sampling buffer, pre-layout')
    print(f'sealed circuit version {_short(version_id)}')

    contexts, metrics = {}, {}
    for name, temperature, voltage in (
            (T_STEP_LIGHT, intent['nominal_temperature_c'], intent['vdd_v']),
            (T_STEP_HEAVY, intent['hot_temperature_c'], intent['hot_vdd_v'])):
        test = tests[name]
        print(f'\n[{name}] transient at ({corner}, {temperature} °C, {voltage} V), '
              f'params={test.parameters}')
        report = run_test(test, corner=corner, temperature=temperature,
                          voltage=voltage, currents=True)
        contexts[name] = test.netlist.characterization_context
        metrics.update(report.metrics)
        stored = mod.store_report(report, tool_name='ngspice')
        current_report = test.netlist.annotate_currents(tstart=_CURRENT_TSTART)
        versions = test.netlist.store_net_currents(
            current_report, corner=corner, db_path=db_path,
            temperature=temperature, voltage=voltage, tool_name='ngspice')
        assert versions['sampling_buffer'] == version_id
        dut = test.netlist.elem_dict['dut']['device']
        print(f'  {report.summary()}')
        print(f"  stored: {stored['metrics_stored']} metric(s), "
              f"{stored['specs_stored']} spec measurement(s), and per-net "
              f"currents (tail {_ua(dut.net_dict['tail'].peak_current)} peak, "
              f"vint {_ua(dut.net_dict['vint'].peak_current)} peak)")

    test = tests[T_AC_PRE]
    print(f"\n[{T_AC_PRE}] AC at ({corner}, {intent['nominal_temperature_c']} °C, "
          f"{intent['vdd_v']} V), params={test.parameters}")
    report = run_test(test, corner=corner, temperature=intent['nominal_temperature_c'],
                      voltage=intent['vdd_v'], currents=False)
    contexts[T_AC_PRE] = test.netlist.characterization_context
    metrics.update(report.metrics)
    stored = mod.store_report(report, tool_name='ngspice')
    print(f'  {report.summary()}')
    print(f"  stored: {stored['metrics_stored']} metric(s), "
          f"{stored['specs_stored']} spec measurement(s)")

    recon = mod.reconcile_specs()
    print('\nSpec reconciliation (pre-layout):')
    print(recon.summary())
    report_pre = generate_report(mod, auto_discover_config=False)
    report_pre.save_json(os.path.join(out_dir, 'report_pre_layout.json'))
    report_pre.save_html(os.path.join(out_dir, 'report_pre_layout.html'))
    print(f"report -> {os.path.join(out_dir, 'report_pre_layout.html')}")

    for name in (T_STEP_LIGHT, T_STEP_HEAVY, T_AC_PRE):
        tests[name].netlist.close()
    mod.close()
    return version_id, contexts, metrics, report_pre


def reopen_and_query(db_path, version_id, contexts, *, tech=SKY130, intent=INTENT):
    """Session 2: provenance from the SQLite file alone — no live circuit."""
    corner = tech.corner
    provenance = {}
    with Database(db_path) as history:
        profiles = history.list_characterization_profiles(version_id)
        print(f'{len(profiles)} characterization profile(s) on version '
              f'{_short(version_id)}:')
        for p in profiles:
            print(f'  profile {_short(p.profile_id)} test={p.test_name!r} '
                  f'params={dict(p.parameters)} origin={_short(p.origin_version_id)}')
        pvt = {T_STEP_LIGHT: (intent['nominal_temperature_c'], intent['vdd_v']),
               T_STEP_HEAVY: (intent['hot_temperature_c'], intent['hot_vdd_v'])}
        for name, (temperature, voltage) in pvt.items():
            profile = history.get_characterization_profile(
                version_id, context=contexts[name])
            origin, rows = history.get_net_current_set(
                version_id, corner, profile_id=profile.profile_id,
                temperature=temperature, voltage=voltage)
            tail_peak = max(abs(r['peak_a']) for r in rows if r['net_name'] == 'tail')
            provenance[name] = dict(profile_id=profile.profile_id,
                                    origin_version_id=origin,
                                    tail_peak_a=tail_peak, rows=len(rows))
            print(f'  selected current set for {name!r} at ({corner}, {temperature} °C, '
                  f'{voltage} V): {len(rows)} pin rows, origin '
                  f'{_short(origin)}, largest tail pin peak {_ua(tail_peak)}')
        bw_rows = history.get_simulation_results(version_id, parameter='bw_hz',
                                                 corner=corner)
        provenance['bw_pre_layout_hz'] = bw_rows[0].value
        print(f"  stored pre-layout bw_hz = {bw_rows[0].value / 1e6:.2f} MHz "
              f"(source={bw_rows[0].source!r}, tool={bw_rows[0].tool_name!r})")
    return provenance


def rebuild_and_restore(knobs, db_path, version_id, contexts, work_dir,
                        provenance, *, tech=SKY130, intent=INTENT, specs=None):
    """Session 3: rebuild from the same factory, restore one exact context.

    Returns ``(module, tests, restored, context_name)``.
    """
    fresh, tests = build_sampling_buffer(knobs, tech=tech, intent=intent,
                                         db_path=db_path, work_dir=work_dir, specs=specs)
    rebuilt_id = fresh.create_circuit_version()
    assert rebuilt_id == version_id, 'the rebuild resolves to the stored snapshot'
    print(f'rebuilt design resolves to the same version {_short(rebuilt_id)} — '
          'no simulation has run in this session')
    assert fresh.net_dict['tail'].peak_current is None

    # EM sizing wants the highest-current context. Session 2 read every
    # profile's selected current set back from the file; the context whose
    # tail pin peaked highest is the one exact context restored here (on
    # sky130 that is the nominal-supply light-load run, which draws more tail
    # current than the hot, low-supply one — the data decides, not this file).
    pvt = {T_STEP_LIGHT: (intent['nominal_temperature_c'], intent['vdd_v']),
           T_STEP_HEAVY: (intent['hot_temperature_c'], intent['hot_vdd_v'])}
    picked = max(pvt, key=lambda name: provenance[name]['tail_peak_a'])
    temperature, voltage = pvt[picked]
    print(f'highest-current context is {picked!r} '
          f'({_ua(provenance[picked]["tail_peak_a"]).strip()} tail peak) — restoring it')
    restored = fresh.restore_characterization_evidence(
        context=contexts[picked], corner=tech.corner,
        temperature=temperature, voltage=voltage)
    print(f'restored profile {_short(restored.profile_id)} '
          f'current set {_short(restored.current_set_id)} '
          f'origin {_short(restored.origin_version_id)} '
          f'onto target {_short(restored.target_version_id)}: '
          f'{restored.measurement_count} spec measurement(s), '
          f'{restored.current_count} current row(s)')
    assert fresh.net_dict['tail'].peak_current is not None
    print(f"  tail carries {_ua(fresh.net_dict['tail'].peak_current)} peak, "
          f"vint {_ua(fresh.net_dict['vint'].peak_current)} peak — from the "
          "database, not a simulator")

    # The other stored spec measurements (heavy-load settling, AC gain/BW)
    # attach to the live specs the same way, so the report can show them.
    fresh.restore_spec_measurements(corner=tech.corner)
    return fresh, tests, restored, picked


def layout_feedback(fresh, tests, knobs, version_id, report_pre, bw_pre,
                    out_dir, *, tech=SKY130, intent=INTENT):
    """Step 7: size routing from restored currents, inject, re-measure, diff."""
    est = estimate_layout(fresh, MET1)
    print(f'{"net":<6} {"restored peak":>14} {"width":>8} {"R":>8} {"C":>8}  sized by')
    em = {}
    for name, nl in sorted(est.nets.items()):
        net = fresh.net_dict[name]
        by = 'restored current' if net.peak_current is not None else 'min width'
        if net.peak_current is not None and nl.width > MET1.min_width:
            by += ' (EM: above min width)'
        em[name] = dict(width_um=nl.width, r_ohm=nl.r, c_ff=nl.c,
                        peak_a=net.peak_current)
        print(f'{name:<6} {_ua(net.peak_current):>14} {nl.width:7.3f}µ '
              f'{nl.r:7.2f}Ω {nl.c:7.2f}f  {by}')
    print('(port nets carry no net-level current by design — a port\'s boundary '
          'current is not one of its pins — so they route at min width)')

    inject_parasitics(fresh, est, mode='rc')
    n_par = sum(1 for k in fresh.passives_dict if k.startswith('__par_'))
    post_version_id = fresh.create_circuit_version(
        'post-layout: met1 routing parasitics injected',
        parent_version_id=version_id)
    assert post_version_id != version_id
    print(f'\ninjected {n_par} parasitic element(s); sealed post-layout version '
          f'{_short(post_version_id)} (parent {_short(version_id)})')

    test = tests[T_AC_POST]
    wire_ac_test(test, cload=intent['light_load_f'], vdd=intent['vdd_v'],
                 iref=knobs['i_ref_a'])
    report = run_test(test, corner=tech.corner,
                      temperature=intent['nominal_temperature_c'],
                      voltage=intent['vdd_v'], currents=False)
    fresh.store_report(report, tool_name='ngspice')
    bw_post = report.metrics['bw_post_layout_hz']
    print(f'  {report.summary()}')

    drop = 100.0 * (bw_pre - bw_post) / bw_pre
    print('\nBefore/after routing parasitics (open-loop, light load):')
    print(f'  {"":<22}{"pre-layout":>14}{"post-layout":>14}{"change":>10}')
    print(f'  {"-3 dB bandwidth":<22}{bw_pre / 1e6:>11.2f} MHz{bw_post / 1e6:>11.2f} MHz'
          f'{-drop:>+9.1f} %')
    print(f"  {'VOUT routing C':<22}{'—':>14}{est.nets['VOUT'].c:>11.2f} fF")
    print('  (sweep grid 200 pts/decade → bandwidth resolution ±1.2 %; the '
          'pre-layout number is the stored evidence, not a re-simulation)')

    report_post = generate_report(fresh, layout_estimate=est,
                                  auto_discover_config=False)
    report_post.save_json(os.path.join(out_dir, 'report_post_layout.json'))
    report_post.save_html(os.path.join(out_dir, 'report_post_layout.html'))
    diff = diff_reports(report_pre, report_post, label_a='pre-layout (session 1)',
                        label_b='post-layout (parasitics injected)')
    diff.save(os.path.join(out_dir, 'pre_vs_post_layout.md'))
    print()
    print(diff)
    print(f"report -> {os.path.join(out_dir, 'report_post_layout.html')}")
    print(f"diff   -> {os.path.join(out_dir, 'pre_vs_post_layout.md')}")
    return est, em, post_version_id, bw_post, report_post


def plot_pre_vs_post(tests, fresh_tests, png_path):
    """Step 7b: the open-loop magnitude before and after routing parasitics.

    Drawn entirely from probes, which is the last step of the taught program
    (``docs/probes.md``): probe -> ``add_metric(fn, sig=...)``
    -> ``run`` -> plot. Two benches means two probes: ``probe('vout')`` on the
    pre-layout AC test and on the post-layout one are different Signals
    naming the same net, because a probe belongs to the bench it was named
    on, and each resolves against the results ITS bench holds. A probe stores
    no samples, so one minted here reads runs that finished several steps
    ago. Both signals are complex, so ``Signal.plot()`` draws magnitude in dB
    on a log-frequency axis without being told.
    """
    import matplotlib
    matplotlib.use('Agg')                    # headless-safe; no display needed
    import matplotlib.pyplot as plt

    pre = tests[T_AC_PRE].probe('vout')
    post = fresh_tests[T_AC_POST].probe('vout')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pre.plot(ax=ax, color='steelblue', lw=1.6, label='pre-layout')
    post.plot(ax=ax, color='crimson', lw=1.4, ls='--',
              label='post-layout (met1 RC injected)')
    ax.set_title('Sampling buffer — open-loop |VOUT|, before and after routing '
                 'parasitics')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return png_path


def demonstrate_mismatch(fresh, knobs, db_path, contexts, work_dir, *,
                         tech=SKY130, intent=INTENT, specs=None):
    """Step 8: evidence that does not match the live structure is refused."""
    outcomes = {}
    if specs is None:
        specs = [outcome.spec for outcome in fresh.reconcile_specs().outcomes]
    corner = tech.corner
    nominal = (intent['nominal_temperature_c'], intent['vdd_v'])

    # (a) A rebuild with one knob changed is a different circuit.
    edited_knobs = dict(knobs, w_in_um=_snap(knobs['w_in_um'] + 1.0,
                                             tech.width_grid_um))
    edited_specs = []
    edited, _ = build_sampling_buffer(edited_knobs, tech=tech, intent=intent,
                                      db_path=db_path,
                                      work_dir=os.path.join(work_dir, 'edited'),
                                      specs=edited_specs)
    edited.create_circuit_version('edited: input pair 1 µm wider')
    try:
        edited.select_characterization_evidence(
            context=contexts[T_STEP_LIGHT], corner=corner,
            temperature=nominal[0], voltage=nominal[1])
    except EvidenceMismatchError as err:
        outcomes['edited'] = type(err).__name__
        print(f'edited design -> {type(err).__name__}: {err}')
    assert edited.net_dict['tail'].peak_current is None, 'nothing was applied'
    assert all(not s.measurements for s in edited_specs), 'nothing was applied'
    edited.close()

    # (b) The post-layout design cannot silently reuse pre-layout evidence.
    before = (fresh.net_dict['tail'].peak_current,
              sum(len(s.measurements) for s in specs))
    try:
        fresh.restore_characterization_evidence(
            context=contexts[T_STEP_LIGHT], corner=corner,
            temperature=nominal[0], voltage=nominal[1])
    except EvidenceMismatchError as err:
        outcomes['post_layout'] = type(err).__name__
        print(f'post-layout design -> {type(err).__name__}: {err}')
    after = (fresh.net_dict['tail'].peak_current,
             sum(len(s.measurements) for s in specs))
    assert before == after, 'a refused restore mutates nothing'
    print('  -> currents and spec measurements on the live design are untouched')
    return outcomes


def main(out_dir=None, *, tech=SKY130, intent=INTENT):
    """Run the whole loop on ``tech`` against ``intent``; returns a summary."""
    out_dir = out_dir or example_output_dir('design_loop')
    work_dir = os.path.join(out_dir, 'work')
    os.makedirs(work_dir, exist_ok=True)
    db_path = os.path.join(out_dir, 'design_loop.db')
    if os.path.exists(db_path):
        os.remove(db_path)                       # deterministic demo

    _h('1. System intent -> sizing')
    print(f"technology {tech.pdk} (corner {tech.corner!r}, VDD {intent['vdd_v']} V)")
    knobs = size_from_intent(intent, tech)
    print(f"heavy-load settling {intent['settle_heavy_s_max'] * 1e9:.0f} ns into "
          f"{intent['heavy_load_f'] * 1e12:.0f} pF -> gm >= {knobs['gm_min_s'] * 1e3:.2f} mS")
    print(f"gm/Id = {GM_OVER_ID:.0f}, margin x{SIZING_MARGIN:.0f} -> "
          f"I_tail = {knobs['i_tail_a'] * 1e6:.0f} µA (I_ref = {knobs['i_ref_a'] * 1e6:.1f} µA, "
          f"mirror {MIRROR_RATIO}:1)")
    print(f"widths: input {knobs['w_in_um']} µm, load {knobs['w_load_um']} µm, "
          f"tail {knobs['w_tail_um']} µm, reference {knobs['w_ref_um']} µm")

    _h('2. Build: parameterized design with embedded characterization suite')
    specs = []
    mod, tests = build_sampling_buffer(knobs, tech=tech, intent=intent,
                                       db_path=db_path, work_dir=work_dir, specs=specs)
    ports = {name: f'{pin.direction.value}/{pin.role.value}'
             for name, pin in mod.pin_list.items()}
    print(f'ports  : {ports}')
    print(f'tests  : {mod.list_tests()}')
    print(f'specs  : {[s.key for s in specs]}')
    wire_pre_layout_tests(tests, knobs, intent)

    _h('3+4. Characterize under two contexts, report, persist')
    version_id, contexts, metrics, report_pre = characterize_and_store(
        mod, tests, knobs, db_path, out_dir, tech=tech, intent=intent)

    _h('5. Reopen the database file — provenance with no live circuit')
    provenance = reopen_and_query(db_path, version_id, contexts, tech=tech,
                                  intent=intent)
    bw_pre = provenance['bw_pre_layout_hz']

    _h('6. Rebuild fresh, restore one exact context')
    fresh_specs = []
    fresh, fresh_tests, restored, restored_context = rebuild_and_restore(
        knobs, db_path, version_id, contexts, os.path.join(work_dir, 'rebuild'),
        provenance, tech=tech, intent=intent, specs=fresh_specs)

    _h('7. Layout feedback: restored currents -> EM widths -> parasitics -> re-measure')
    est, em, post_version_id, bw_post, report_post = layout_feedback(
        fresh, fresh_tests, knobs, version_id, report_pre, bw_pre, out_dir,
        tech=tech, intent=intent)

    bode_path = plot_pre_vs_post(tests, fresh_tests,
                                 os.path.join(out_dir, 'open_loop_pre_vs_post.png'))
    print(f'probe plot -> {bode_path}')

    _h('8. A mismatch fails loudly, with nothing half-applied')
    mismatch = demonstrate_mismatch(fresh, knobs, db_path, contexts, work_dir,
                                    tech=tech, intent=intent, specs=fresh_specs)

    _h('9. The netlist comes from the same design object')
    netlist = fresh.generate_netlist()
    netlist_path = os.path.join(out_dir, 'sampling_buffer_post_layout.cir')
    with open(netlist_path, 'w') as f:
        f.write(netlist)
    lines = netlist.splitlines()
    print('\n'.join(lines[:3] + ['  ...']
                    + [l for l in lines if l.startswith(('XMM2', 'XMM4'))]
                    + [l for l in lines if '__par_' in l and 'VOUT' in l]
                    + ['  ...'] + lines[-1:]))
    print(f'({len(lines)} lines) -> {netlist_path}')
    fresh_tests[T_AC_POST].netlist.close()
    fresh.close()

    print('\nDone. Guides: docs/agent-guide.md (loop), docs/database.md '
          '(evidence contract), docs/api-reference.md#silklayout (layout).')
    return dict(
        tech=tech, intent=intent,
        knobs=knobs, version_id=version_id, post_version_id=post_version_id,
        contexts=contexts, metrics=metrics, provenance=provenance,
        restored=restored, restored_context=restored_context,
        layout=est, em=em, bw_pre_hz=bw_pre, bw_post_hz=bw_post,
        mismatch=mismatch, netlist_path=netlist_path, out_dir=out_dir,
        report_post=report_post,
    )


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
