"""IHP SG13G2 simulates end to end through silk — the OSDI bring-up proof
(TASK-18, 2026-09-02).

Three things this machine must have, none of them a Python dependency: the
PDK install, its four Verilog-A models compiled to ``.osdi`` (openvaf-r +
the PDK's own install.py — silk never compiles), and an ngspice that speaks
the plugins' OSDI version (openvaf-r emits v0.4; Ubuntu's ngspice-42 stops
at v0.3 and refuses them, 47 loads them). ``needs_osdi_runnable`` asks the
ngspice silk would run, so the module skips with the reason rather than
failing on a box that lacks any of the three. Point ``$SILK_NGSPICE_BIN`` at
a new enough ngspice to run it; see docs/installing-the-pdk.md.

Every assertion is a *physical band*, never an exit code: currents in the
range a 130 nm process must produce, monotonic transfer curves, corner and
temperature effects with the right sign. The bands are wide on purpose — a
model revision may move Ion by 20 %, but a unit error moves it by 10^6 and
a swapped device type flips the n:p ratio — so they catch the failures that
matter without pinning IHP's numbers. The values measured at bring-up are
recorded beside each band.
"""
import numpy as np
import pytest

from tests._env import needs_ngspice, needs_osdi_runnable

pytestmark = [
    pytest.mark.requires_sg13g2,
    needs_ngspice,
    needs_osdi_runnable('sg13g2'),
]

VDD_LV, VDD_HV = 1.2, 3.3

# (device, VDD, L_um, Ion band [A], Ioff ceiling [A]) — W = 1 um throughout, so
# Ion is per micron. Measured 2026-09-02, typical, 27 C, ngspice-47:
#   nmos_lv 400 uA / 26 pA   pmos_lv 212 uA / 46 pA
#   nmos_hv 533 uA / 0.14 pA pmos_hv 219 uA / 0.26 pA
DEVICES = [
    ('nmos_lv', VDD_LV, 0.13, (200e-6, 800e-6), 1e-9),
    ('pmos_lv', VDD_LV, 0.13, (100e-6, 450e-6), 1e-9),
    ('nmos_hv', VDD_HV, 0.45, (250e-6, 1000e-6), 100e-12),
    ('pmos_hv', VDD_HV, 0.45, (100e-6, 500e-6), 100e-12),
]


@pytest.fixture(scope='module')
def pdk():
    from silk import load_pdk
    return load_pdk('sg13g2')


def _id_vg(pdk, tmp_path, device, vdd, l_um):
    """Drain current vs Vgs at |Vds| = VDD for a W = 1 um device, through the
    standard runner. Returns (vg, id) with id > 0 for both polarities."""
    from silk import sources
    from silk.circuit.modules import Module
    from silk.simulation import DC
    sign = -1 if device.startswith('p') else 1
    tb = Module('sg13g2', str(tmp_path), f'{device}.cir', pdk_details=pdk)
    for n in ('d', 'g'):
        tb.add_net(n)
    with tb:
        getattr(pdk, device)('M1', d='d', g='g', s='0', b='0', w=1.0, l=l_um)
    tb.add_source(sources.VoltageSource('D', positive='d', negative='0', dc=sign * vdd))
    tb.add_source(sources.VoltageSource('G', positive='g', negative='0', dc=0))
    tb.add_sim_type(DC('VG', 0, sign * vdd, sign * 0.05))
    tb.simulate(timeout=120)
    sig = tb.load_signal('vd', 'I')
    return np.asarray(sig.x), -sign * np.asarray(sig.y)


# --------------------------------------------------------------------------- #
# AC1 — a device simulation with physically sensible values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('device, vdd, l_um, ion_band, ioff_max', DEVICES,
                         ids=[d[0] for d in DEVICES])
def test_mosfet_id_vg_is_physical(pdk, tmp_path, device, vdd, l_um, ion_band,
                                  ioff_max):
    vg, idr = _id_vg(pdk, tmp_path, device, vdd, l_um)
    assert len(vg) == round(vdd / 0.05) + 1 and abs(vg[-1]) == pytest.approx(vdd)
    ioff, ion = idr[0], idr[-1]
    assert ioff < ioff_max, f'{device}: Ioff {ioff:.3e} A'
    assert ion_band[0] < ion < ion_band[1], f'{device}: Ion {ion:.3e} A/um'
    # Subthreshold -> strong inversion: strictly increasing, and mid-swing
    # sits between the two (not saturated at either end).
    assert np.all(np.diff(idr) > 0), f'{device}: Id not monotonic in Vgs'
    assert ioff < idr[len(idr) // 2] < ion


def test_nmos_beats_pmos_by_the_mobility_ratio(pdk, tmp_path):
    """Same W and L: electron mobility puts NMOS Ion 1.3-3x the PMOS's
    (measured 1.9x LV, 2.4x HV). A swapped type= in the model map, or the
    wrong section file, fails here while each device alone could pass."""
    for n, p, vdd, l_um in (('nmos_lv', 'pmos_lv', VDD_LV, 0.13),
                            ('nmos_hv', 'pmos_hv', VDD_HV, 0.45)):
        ion_n = _id_vg(pdk, tmp_path, n, vdd, l_um)[1][-1]
        ion_p = _id_vg(pdk, tmp_path, p, vdd, l_um)[1][-1]
        assert 1.3 < ion_n / ion_p < 3.0, (n, p, ion_n / ion_p)


# --------------------------------------------------------------------------- #
# AC3 — a silk-native testbench: factory circuit + ModuleTestbench + corners
# --------------------------------------------------------------------------- #
def _inverter(pdk, work_dir):
    from silk.circuit.modules import Module
    from silk.circuit.ports import Port
    mod = Module('PDK_sg13g2', work_dir, 'inv.cir', pdk_details=pdk,
                 module_name='inv')
    mod.define_module_ports({'VDD': Port.supply(), 'VSS': Port.ground(),
                             'VIN': Port.analog_input(),
                             'VOUT': Port.analog_output()})
    with mod:
        pdk.nmos_lv('MN', d='VOUT', g='VIN', s='VSS', b='VSS', w=1.0, l=0.13)
        pdk.pmos_lv('MP', d='VOUT', g='VIN', s='VDD', b='VDD', w=2.0, l=0.13)
    return mod


def _switching_threshold(ckt):
    s = ckt.load_signal('vout')
    x, y = np.asarray(s.x), np.asarray(s.y)
    k = np.where(np.diff(np.sign(y - VDD_LV / 2)))[0][0]
    return float(x[k] + (VDD_LV / 2 - y[k]) * (x[k + 1] - x[k]) / (y[k + 1] - y[k]))


def _supply_leakage(ckt):
    """Supply current with VIN = 0: the PMOS is on, so this is NMOS Ioff."""
    return float(abs(np.asarray(ckt.load_signal('vvdd', 'I').y)[0]))


def _vtc_test(mod, temp_c):
    """A DC transfer sweep at ``.temp temp_c`` — the deck directive is the
    only way temperature reaches the simulator; ``run(temperature=)`` is
    characterization metadata and never changes the deck."""
    from silk import sources
    from silk.simulation import DC
    t = mod.create_test(f'vtc_{temp_c}c')
    t.add_module_under_test({'VDD': 'vdd', 'VSS': '0', 'VIN': 'vin', 'VOUT': 'vout'})
    t.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=VDD_LV))
    t.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=0))
    t.add_custom_line(f'.temp {temp_c}')
    t.add_sim_type(DC('Vvin', 0, VDD_LV, 0.01))
    t.add_metric('vm', _switching_threshold)
    t.add_metric('ioff', _supply_leakage)
    t.add_spec(f'Vm above 0.45 V at {temp_c} C', 'vm', '>', 0.45)
    return t


def test_testbench_deck_is_the_documented_sg13g2_form(pdk, tmp_path):
    """The bytes docs/installing-the-pdk.md promises: per-corner section
    lines, metre-native device lines, the pre_osdi loads at the top of the
    .control block, and the .temp line from add_custom_line."""
    t = _vtc_test(_inverter(pdk, str(tmp_path)), 85)
    t.netlist.generate_netlist(corner='ss')            # writes the deck
    import pathlib
    text = pathlib.Path(t.netlist._run_files.netlist_path).read_text()
    assert '/cornerMOSlv.lib" mos_ss\n' in text
    assert '/cornerRES.lib" res_wcs\n' in text
    assert 'XMMN VOUT VIN VSS VSS sg13_lv_nmos m=1 w=1u l=130n ng=1\n' in text
    assert 'XMMP VOUT VIN VDD VDD sg13_lv_pmos m=1 w=2u l=130n ng=1\n' in text
    assert '.temp 85\n' in text
    control = text[text.index('.control'):]
    for name in ('psp103', 'psp103_nqs', 'r3_cmc', 'mosvar'):
        assert f'pre_osdi {pdk.osdi_files[0].rsplit("/", 1)[0]}/{name}.osdi\n' in control
    assert control.index('pre_osdi') < control.index('\nrun\n')


@pytest.mark.slow
def test_inverter_across_corners_and_temperature(pdk, tmp_path):
    """Five process corners at 27 C and 85 C through ModuleTestbench.run_corners.

    Measured 2026-09-02 (Vm in V; Ioff in A, VIN = 0):
        27 C  typical 0.618/6.0e-11  ss 0.611/1.2e-11  ff 0.622/2.8e-10
              sf 0.650/6.6e-11       fs 0.585/8.7e-11
        85 C  typical 0.620/4.0e-10  ss 0.613/4.5e-11  ff 0.624/1.9e-9
              sf 0.653/1.9e-10       fs 0.585/8.2e-10
    """
    corners = ['typical', 'ss', 'ff', 'sf', 'fs']
    mod = _inverter(pdk, str(tmp_path))
    vm, ioff = {}, {}
    for temp_c in (27, 85):
        t = _vtc_test(mod, temp_c)
        results = t.run_corners(corners, temperature=temp_c, voltage=VDD_LV)
        assert results.passed
        for c in corners:
            vm[temp_c, c] = results[c].metrics['vm']
            ioff[temp_c, c] = results[c].metrics['ioff']
            assert results[c].characterization_context.temperature == temp_c
    for key, v in vm.items():
        # A 2:1 P:N inverter switches near mid-rail: 0.45..0.75 of a 1.2 V rail.
        assert 0.45 < v < 0.75, (key, v)
    for temp_c in (27, 85):
        # Skewed corners move Vm with the right sign; balanced ones barely do.
        assert vm[temp_c, 'fs'] < vm[temp_c, 'typical'] < vm[temp_c, 'sf']
        assert vm[temp_c, 'sf'] - vm[temp_c, 'fs'] > 0.03
        for c in ('ss', 'ff'):
            assert abs(vm[temp_c, c] - vm[temp_c, 'typical']) < 0.03, c
        # Leakage orders with the corner...
        assert ioff[temp_c, 'ss'] < ioff[temp_c, 'typical'] < ioff[temp_c, 'ff']
    for c in corners:
        # ...and grows with temperature at every corner (measured 3-10x).
        assert ioff[85, c] > 2 * ioff[27, c], (c, ioff[27, c], ioff[85, c])
