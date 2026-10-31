"""Reproducer for TASK-301.8.

A circuit composed from two blocks built under SEPARATE ``load_pdk('sky130')``
calls could not be pickled: ``silk.elements.factory.make_device_class``
registers each PDK device class by bare name, last-registration-wins, so an
earlier block's devices hold a class object no longer in that registry slot.
Pickle resolves a class by module + qualname and demands the exact object
back, so it raised ``PicklingError: Can't pickle <class
'silk.elements.factory.nmos1p8'>: it's not the same object as
silk.elements.factory.nmos1p8``.

Composing blocks from separate files (each with its own top-level
``load_pdk()`` call) is the normal designer workflow — see
``silk.examples.charge_pump_example``, which imports ``build_inverter()``
from ``silk.examples.ring_oscillator_example``; both modules call
``load_pdk('sky130')`` at import time. This is the smallest circuit that
reproduces the same failure without any example machinery.
"""
import hashlib
import pickle

from silk import Port, load_pdk, sources
from silk.circuit.circuit_elements import _module_structure_hash
from silk.circuit.modules import Module
from silk.elements.base import AnalogElement
from silk.simulation import DC
from tests._env import needs_ngspice

#: Netlist text and structure hash for :func:`_build_composed_circuit`,
#: captured on main (before TASK-301.8's fix) and unchanged after it — the
#: serial composed path (no pickling at all) must render and hash exactly
#: as it did before, since the fix only changes what happens when an
#: instance or a ``BasePDK`` gets PICKLED.
_GOLDEN_NETLIST = (
    "\n*** subckt definition for block_a\n"
    ".subckt block_a D G S \n"
    "*** m1.  Connections = ['d', 'g', 's', 'b'].\n"
    "XMm1 D G S S sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.5\n"
    ".ends\n"
    ".subckt top D G S \n"
    "*** m2.  Connections = ['d', 'g', 's', 'b'].\n"
    "XMm2 D2 G2 S2 S2 sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.5\n"
    "*** u_a.  Connections = ['D', 'G', 'S'].\n"
    "xu_a D G S block_a\n"
    ".ends\n"
)
_GOLDEN_NETLIST_SHA256 = (
    'ab7e07568126aa832635aa94895bdac0d50d7429aa76c679a43b3a420fda3470')
_GOLDEN_STRUCTURE_HASH = (
    '9161e7162f0fa3353edb4b5e0c3685dd0f7f64c4323cfa4d2afec8bf3891b65c')


def _build_block(pdk, tmp_dir, name):
    """A one-transistor subcircuit built under *pdk*'s own ``load_pdk()`` call."""
    m = Module('PDK_skywater130', tmp_dir, f'{name}.cir', pdk_details=pdk,
               module_name=name)
    m.define_module_ports({'D': Port.analog_output(), 'G': Port.analog_input(),
                           'S': Port.ground()})
    with m:
        pdk.nmos1p8('m1', d='D', g='G', s='S', b='S', w=1.0, l=0.5)
    return m


def _build_composed_circuit(tmp_path):
    """Two blocks, each built under its OWN ``load_pdk('sky130')`` call,
    composed into one circuit — the normal designer workflow (see
    ``silk.examples.charge_pump_example`` importing
    ``ring_oscillator_example.build_inverter()``). Returns ``(block_a, top)``.
    """
    pdk_a = load_pdk('sky130')
    block_a = _build_block(pdk_a, str(tmp_path / 'a'), 'block_a')

    pdk_b = load_pdk('sky130')  # a SEPARATE load -- re-registers nmos1p8
    top = Module('PDK_skywater130', str(tmp_path / 'top'), 'top.cir',
                 pdk_details=pdk_b, module_name='top')
    top.define_module_ports({'D': Port.analog_output(), 'G': Port.analog_input(),
                             'S': Port.ground()})
    with top:
        pdk_b.nmos1p8('m2', d='D2', g='G2', s='S2', b='S2', w=1.0, l=0.5)
    top.add_module(block_a, 'u_a', nets={'D': 'D', 'G': 'G', 'S': 'S'})
    return block_a, top


def _device_of(mod, device_type_name='nmos1p8'):
    for entry in mod.elem_dict.values():
        device = entry.get('device')
        if device is not None and getattr(device, '_device_type_name', None) == device_type_name:
            return device
    raise AssertionError(f'no {device_type_name!r} device in {mod.module_name!r}')


def test_circuit_composed_from_two_separate_pdk_loads_pickles(tmp_path):
    """Two blocks, each built under its OWN ``load_pdk('sky130')`` call,
    composed into one circuit, must pickle and restore to the same netlist.

    On main this raised a ``PicklingError`` naming ``nmos1p8``; after the
    fix it round-trips.
    """
    _, top = _build_composed_circuit(tmp_path)
    expected_netlist = top.generate_netlist(corner='tt')

    restored = pickle.loads(pickle.dumps(top))

    assert restored.generate_netlist(corner='tt') == expected_netlist


# ---------------------------------------------------------------------------
# AC #4: the SERIAL composed path (no pickling at all) is unchanged.
# ---------------------------------------------------------------------------

def test_serial_composed_netlist_and_structure_hash_unchanged(tmp_path):
    """Netlist bytes and structure hash for the composed circuit match the
    golden values captured on main -- the fix touches pickling only."""
    _, top = _build_composed_circuit(tmp_path)

    netlist = top.generate_netlist(corner='tt')
    assert netlist == _GOLDEN_NETLIST
    assert hashlib.sha256(netlist.encode()).hexdigest() == _GOLDEN_NETLIST_SHA256
    assert _module_structure_hash(top) == _GOLDEN_STRUCTURE_HASH


def test_serial_composed_isinstance_and_type_checks_unchanged(tmp_path):
    """Devices from the two separate ``load_pdk()`` calls stay distinct
    class objects in a single process -- this is the TASK-312.3
    last-registered-wins policy for PDK devices, and the fix must not
    change it (it only changes what happens when an instance is pickled).
    """
    block_a, top = _build_composed_circuit(tmp_path)
    block_a_device = _device_of(block_a)
    top_device = _device_of(top)

    assert type(block_a_device) is not type(top_device)
    assert not isinstance(block_a_device, type(top_device))
    assert not isinstance(top_device, type(block_a_device))
    assert isinstance(block_a_device, AnalogElement)
    assert isinstance(top_device, AnalogElement)


@needs_ngspice
def test_serial_composed_tb_run_succeeds(tmp_path):
    """A testbench built on the composed circuit still runs (serial,
    in-process, no pool -- the path every shipped example uses)."""
    _, top = _build_composed_circuit(tmp_path)
    tb = top.create_test('smoke')
    tb.add_module_under_test({'D': 'vd', 'G': 'vg', 'S': '0'})
    tb.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.8))
    tb.add_sim_type(DC(source='vvg', start=0.0, stop=1.8, step=0.2))
    tb.add_metric('vd_final', lambda ckt: float(ckt.load_signal('vd', 'V').y[-1]))
    tb.add_spec('nonzero', 'vd_final', '>', 0.0)

    report = tb.run(corner='tt')

    assert report.passed is True
