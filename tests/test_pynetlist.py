"""
Tests for Module — the main circuit class integrating elements, sources,
netlist generation, and simulation invocation.
"""

import os
from collections.abc import Mapping
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from silk.circuit.modules import Module
from silk import sources
from silk.exceptions import CircuitError
from silk.simulation.analysis import Transient, AC, DC, Noise
from tests._env import needs_ngspice


def _make_fake_circuit_db_module():
    """Return a MagicMock module with a DesignStage enum for test injection."""
    from unittest.mock import MagicMock
    from enum import Enum

    class DesignStage(Enum):
        DEV = 'dev'
        STABLE = 'stable'
        POST_LAYOUT = 'post_layout'
        TAPEOUT = 'tapeout'

    fake = MagicMock()
    fake.DesignStage = DesignStage
    return fake


# ---------------------------------------------------------------------------
# Fixtures that include the 'm' parameter (required by generate_netlist_entry)
# ---------------------------------------------------------------------------

@pytest.fixture
def full_param_netlist(pdk, tmp_dir):
    """PyNetlist with devices that have all params (w, l, m) set."""
    ckt = Module('PDK_skywater130', tmp_dir, 'full.cir', pdk_details=pdk)
    ckt.add_net('vdd')
    ckt.add_net('gnd')
    ckt.add_net('in')
    ckt.add_net('out')

    ckt.add(pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15, m=1))
    ckt.add(pdk.pmos1p8('mp1', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15, m=1))

    ckt.add_source(sources.VoltageSource('vdd_supply', positive='vdd', negative='gnd', dc=1.8))
    ckt.add_source(sources.VoltageSource('vin', positive='in', negative='gnd', dc=0.9))
    ckt.add_sim_type(Transient(step=1e-9, stop=1e-7))

    return ckt



class TestPyNetlistConstructor:

    def test_instance_attributes(self, pdk, tmp_dir):
        """Constructor sets dir_folder, netlistname, pdk_name, and empty dicts."""
        ckt = Module('PDK_skywater130', tmp_dir, 'ctor.cir', pdk_details=pdk)

        assert ckt.dir_folder == tmp_dir
        assert ckt.netlistname == 'ctor.cir'
        assert ckt.pdk_name == 'PDK_skywater130'
        assert isinstance(ckt.elem_dict, Mapping) and len(ckt.elem_dict) == 0
        assert isinstance(ckt.net_dict, Mapping)
        assert isinstance(ckt.sig_sources_dict, dict) and len(ckt.sig_sources_dict) == 0
        assert isinstance(ckt.sim_analysis_dict, dict) and len(ckt.sim_analysis_dict) == 0
        assert isinstance(ckt.custom_lines, list) and len(ckt.custom_lines) == 0

    def test_pdk_details_attached(self, empty_netlist, pdk):
        """PDK details object is stored on the instance."""
        assert empty_netlist.pdk_details is pdk

    def test_no_tracker_by_default(self, empty_netlist):
        """No tracker created when no db_path given."""
        assert empty_netlist._tracker is None



class TestAddElement:

    def test_element_appears_in_elem_dict(self, empty_netlist, pdk):
        """After add_element, the element name exists in elem_dict."""
        empty_netlist.add_element(
            pdk.nmos1p8, 'mn1',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters={'w': 1, 'l': 0.15, 'm': 1},
        )
        assert 'mn1' in empty_netlist.elem_dict

    def test_element_has_correct_type(self, empty_netlist, pdk):
        """Device elements have type 'd'."""
        empty_netlist.add_element(
            pdk.nmos1p8, 'mn1',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters={'w': 1, 'l': 0.15, 'm': 1},
        )
        assert empty_netlist.elem_dict['mn1']['type'] == 'd'

    def test_element_has_correct_typename(self, empty_netlist, pdk):
        """Element typeName matches the PDK device name."""
        empty_netlist.add_element(
            pdk.pmos1p8, 'mp1',
            nets={'d': 'out', 'g': 'in', 's': 'vdd', 'b': 'vdd'},
            parameters={'w': 2, 'l': 0.15, 'm': 1},
        )
        assert empty_netlist.elem_dict['mp1']['typeName'] == pdk.pmos1p8.device_name

    def test_duplicate_name_exits(self, basic_netlist, pdk):
        """Adding an element with a duplicate name raises CircuitError."""
        with pytest.raises(CircuitError):
            basic_netlist.add_element(
                pdk.nmos1p8, 'mn1',
                nets={'d': 'x', 'g': 'x', 's': 'x', 'b': 'x'},
            )

    def test_nets_auto_created(self, empty_netlist, pdk):
        """Nets passed in add_element are auto-added to net_dict."""
        empty_netlist.add_element(
            pdk.nmos1p8, 'mn1',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
        )
        for net_name in ('out', 'in', 'gnd'):
            assert net_name in empty_netlist.net_dict



class TestAddSource:

    def test_source_appears_in_dict(self, basic_netlist):
        """After add_source, source name exists in sig_sources_dict."""
        basic_netlist.add_source(
            sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8)
        )
        assert 'vsup' in basic_netlist.sig_sources_dict

    def test_source_type_stored(self, basic_netlist):
        """Source entry records the correct type."""
        basic_netlist.add_source(
            sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8)
        )
        assert isinstance(basic_netlist.sig_sources_dict['vsup'], sources.VoltageSource)

    def test_duplicate_source_exits(self, basic_netlist):
        """Adding a source with a duplicate name raises CircuitError."""
        basic_netlist.add_source(sources.VoltageSource('v1', positive='vdd', negative='gnd', dc=1.8))
        with pytest.raises(CircuitError):
            basic_netlist.add_source(sources.VoltageSource('v1', positive='vdd', negative='gnd', dc=1.0))

    def test_invalid_source_type_exits(self, basic_netlist):
        """Passing a non-source object raises CircuitError."""
        with pytest.raises(CircuitError):
            basic_netlist.add_source('not_a_source')

    def test_source_dc_parameter(self, basic_netlist):
        """DC parameter value is stored correctly."""
        basic_netlist.add_source(
            sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=3.3)
        )
        src = basic_netlist.sig_sources_dict['vsup']
        assert src.dc == 3.3



class TestAddNet:

    def test_net_appears_in_dict(self, empty_netlist):
        """add_net creates an entry in net_dict."""
        empty_netlist.add_net('vdd')
        assert 'vdd' in empty_netlist.net_dict

    def test_duplicate_net_no_error(self, empty_netlist):
        """Adding the same net twice does not raise; it's a no-op."""
        empty_netlist.add_net('vdd')
        empty_netlist.add_net('vdd')  # should not error
        assert 'vdd' in empty_netlist.net_dict



class TestAddSimType:

    def test_transient_sim_added(self, empty_netlist):
        """add_sim_type(Transient(...)) populates sim_analysis_dict."""
        empty_netlist.add_sim_type(Transient(step=1e-9, stop=1e-7))
        assert 'transient' in empty_netlist.sim_analysis_dict
        assert empty_netlist.sim_analysis_dict['transient'].step == 1e-9
        assert empty_netlist.sim_analysis_dict['transient'].stop == 1e-7



class TestAddCustomLine:

    def test_custom_line_stored(self, empty_netlist):
        """add_custom_line appends to custom_lines."""
        empty_netlist.add_custom_line('cload out 0 10p')
        assert 'cload out 0 10p' in empty_netlist.custom_lines

    def test_multiple_custom_lines(self, empty_netlist):
        """Multiple custom lines are all stored in order."""
        empty_netlist.add_custom_line('rload out vdd 1k')
        empty_netlist.add_custom_line('cload out 0 10p')
        assert len(empty_netlist.custom_lines) == 2
        assert empty_netlist.custom_lines[0] == 'rload out vdd 1k'



class TestGenerateNetlist:

    def test_file_written_to_disk(self, full_param_netlist):
        """generate_netlist() writes a file at the expected path."""
        ckt = full_param_netlist
        ckt.set_output_options(mode='overwrite')
        ckt.generate_netlist(corner='tt')
        assert os.path.isfile(os.path.join(ckt.dir_folder, ckt.netlistname))

    def test_file_contains_device_names(self, full_param_netlist):
        """Generated netlist contains the device instance names."""
        ckt = full_param_netlist
        ckt_deck_path = ckt.generate_netlist(corner='tt')
        content = open(ckt_deck_path).read()
        assert 'mn1' in content
        assert 'mp1' in content

    def test_file_contains_lib_include(self, full_param_netlist):
        """Generated netlist contains a .lib include directive."""
        ckt = full_param_netlist
        ckt_deck_path = ckt.generate_netlist(corner='tt')
        content = open(ckt_deck_path).read()
        assert '.lib' in content.lower()

    def test_file_contains_simulation_command(self, full_param_netlist):
        """Generated netlist contains the transient simulation command."""
        ckt = full_param_netlist
        ckt_deck_path = ckt.generate_netlist(corner='tt')
        content = open(ckt_deck_path).read()
        assert '.tran' in content.lower() or 'tran' in content.lower()

    def test_corner_tt_in_lib_line(self, full_param_netlist):
        """Corner 'tt' appears in the .lib line."""
        ckt = full_param_netlist
        ckt_deck_path = ckt.generate_netlist(corner='tt')
        content = open(ckt_deck_path).read()
        assert 'tt' in content

    def test_corner_ff_in_lib_line(self, full_param_netlist):
        """Corner 'ff' appears in the .lib line when specified."""
        ckt = full_param_netlist
        ckt_deck_path = ckt.generate_netlist(corner='ff')
        content = open(ckt_deck_path).read()
        assert 'ff' in content

    def test_corner_ss_in_lib_line(self, full_param_netlist):
        """Corner 'ss' appears in the .lib line when specified."""
        ckt = full_param_netlist
        ckt_deck_path = ckt.generate_netlist(corner='ss')
        content = open(ckt_deck_path).read()
        assert 'ss' in content

    def test_file_contains_end_directive(self, full_param_netlist):
        """Generated netlist contains an .end directive."""
        ckt = full_param_netlist
        ckt_deck_path = ckt.generate_netlist(corner='tt')
        content = open(ckt_deck_path).read().lower()
        assert '.end' in content

    # ------------------------------------------------------------------
    # simulate() convenience method
    # ------------------------------------------------------------------

    def test_simulate_writes_netlist_file(self, full_param_netlist, monkeypatch):
        """simulate() writes the netlist file (generate_netlist step fires)."""
        ckt = full_param_netlist
        ckt.set_output_options(mode='overwrite')

        # Stub out the parts that need ngspice
        monkeypatch.setattr(ckt, 'run_simulator', lambda **kw: None)
        monkeypatch.setattr(ckt, 'load_file',     lambda path: None)
        monkeypatch.setattr(ckt, 'read_raw_file',  lambda: None)

        ckt.simulate(corner='tt')
        assert os.path.isfile(os.path.join(ckt.dir_folder, ckt.netlistname))

    def test_simulate_returns_self(self, full_param_netlist, monkeypatch):
        """simulate() returns the same PyNetlist instance for chaining."""
        ckt = full_param_netlist
        monkeypatch.setattr(ckt, 'run_simulator', lambda **kw: None)
        monkeypatch.setattr(ckt, 'load_file',     lambda path: None)
        monkeypatch.setattr(ckt, 'read_raw_file',  lambda: None)

        result = ckt.simulate(corner='tt')
        assert result is ckt

    def test_simulate_corner_forwarded(self, full_param_netlist, monkeypatch):
        """simulate() forwards the corner argument to generate_netlist."""
        ckt = full_param_netlist
        captured = {}

        original_generate = ckt.generate_netlist
        def capturing_generate(corner='tt'):
            captured['corner'] = corner
            original_generate(corner=corner)
        monkeypatch.setattr(ckt, 'generate_netlist', capturing_generate)
        monkeypatch.setattr(ckt, 'run_simulator',    lambda **kw: None)
        monkeypatch.setattr(ckt, 'load_file',        lambda path: None)
        monkeypatch.setattr(ckt, 'read_raw_file',     lambda: None)

        ckt.simulate(corner='ss')
        assert captured['corner'] == 'ss'

    def test_simulate_consumes_run_without_loading_a_guessed_file(self, full_param_netlist, monkeypatch):
        """simulate() reads the run result without selecting another disk file."""
        ckt = full_param_netlist
        calls = []

        original_gen = ckt.generate_netlist
        monkeypatch.setattr(ckt, 'generate_netlist', lambda corner='tt': (calls.append('generate'), original_gen(corner=corner)))
        monkeypatch.setattr(ckt, 'run_simulator',    lambda **kw:        calls.append('run'))
        monkeypatch.setattr(ckt, 'load_file',        lambda path:        calls.append('load'))
        monkeypatch.setattr(ckt, 'read_raw_file',     lambda:             calls.append('read'))

        ckt.simulate(corner='tt')
        assert calls == ['generate', 'run', 'read']

    def test_custom_line_in_output(self, full_param_netlist):
        """Custom lines appear in the generated netlist."""
        ckt = full_param_netlist
        ckt.add_custom_line('cload out 0 10p')
        ckt_deck_path = ckt.generate_netlist(corner='tt')
        content = open(ckt_deck_path).read()
        assert 'cload out 0 10p' in content

    def test_source_names_in_output(self, full_param_netlist):
        """Source names appear in the generated netlist."""
        ckt = full_param_netlist
        ckt_deck_path = ckt.generate_netlist(corner='tt')
        content = open(ckt_deck_path).read()
        assert 'vdd_supply' in content
        assert 'vin' in content



class TestRunSimulator:

    @needs_ngspice

    def test_run_calls_ngspice(self, full_param_netlist, mock_ngspice):
        """run_simulator() calls os.system with a command containing 'ngspice'."""
        ckt = full_param_netlist
        ckt.generate_netlist(corner='tt')
        ckt.run_simulator()
        assert mock_ngspice['called']
        # argv[0] may be a bare 'ngspice' or the full $SILK_NGSPICE_BIN path
        assert 'ngspice' in mock_ngspice['command'][0]

    @needs_ngspice

    def test_run_command_contains_netlist_path(self, full_param_netlist, mock_ngspice):
        """The ngspice command includes the netlist file path."""
        ckt = full_param_netlist
        ckt.set_output_options(mode='overwrite')
        ckt.generate_netlist(corner='tt')
        ckt.run_simulator()
        generation = ckt._raw._generation
        assert generation.staged.netlist_path == mock_ngspice['command'][-1]
        assert generation.public.netlist_path == os.path.join(ckt.dir_folder, ckt.netlistname)

    @needs_ngspice

    def test_run_sets_results_file(self, full_param_netlist, mock_ngspice):
        """After run_simulator(), results_file attribute is set."""
        ckt = full_param_netlist
        ckt.generate_netlist(corner='tt')
        ckt.run_simulator()
        assert hasattr(ckt, 'results_file')
        assert 'rawfile.raw' in ckt.results_file

    @needs_ngspice

    def test_run_command_contains_raw_flag(self, full_param_netlist, mock_ngspice):
        """The ngspice command includes -r for raw file output."""
        ckt = full_param_netlist
        ckt.generate_netlist(corner='tt')
        ckt.run_simulator()
        assert '-r' in mock_ngspice['command']

    @needs_ngspice

    def test_run_command_contains_batch_flag(self, full_param_netlist, mock_ngspice):
        """The ngspice command includes -b for batch mode."""
        ckt = full_param_netlist
        ckt.generate_netlist(corner='tt')
        ckt.run_simulator()
        assert '-b' in mock_ngspice['command']



class TestPrintMethods:

    def test_print_elements_no_error(self, basic_netlist, capsys):
        """print_elements() runs without error and produces output."""
        basic_netlist.print_elements()
        captured = capsys.readouterr()
        assert 'mn1' in captured.out
        assert 'mp1' in captured.out

    def test_print_nets_no_error(self, basic_netlist, capsys):
        """print_nets() runs without error and produces output."""
        basic_netlist.print_nets()
        captured = capsys.readouterr()
        assert 'vdd' in captured.out
        assert 'gnd' in captured.out



class TestFullFlow:

    @needs_ngspice

    def test_add_generate_run(self, pdk, tmp_dir, mock_ngspice):
        """Full flow: create netlist → add elements → sources → sim → generate → run."""
        ckt = Module('PDK_skywater130', tmp_dir, 'flow.cir', pdk_details=pdk)
        ckt.set_output_options(mode='overwrite')

        ckt.add_net('vdd')
        ckt.add_net('gnd')
        ckt.add_net('in')
        ckt.add_net('out')

        ckt.add(pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15, m=1))

        ckt.add(pdk.pmos1p8('mp1', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15, m=1))

        ckt.add_source(sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8))

        ckt.add_source(sources.VoltageSource('vin', positive='in', negative='gnd', dc=0.9))

        ckt.add_sim_type(Transient(step=1e-9, stop=1e-7))

        ckt.generate_netlist(corner='tt')
        assert os.path.isfile(os.path.join(ckt.dir_folder, ckt.netlistname))

        ckt.run_simulator()
        assert mock_ngspice['called']
        assert mock_ngspice['call_count'] == 1

    @needs_ngspice

    def test_multiple_simulations_increment_count(self, full_param_netlist, mock_ngspice):
        """Running the simulator multiple times increments the call count."""
        ckt = full_param_netlist
        ckt.generate_netlist(corner='tt')
        ckt.run_simulator()
        ckt.generate_netlist(corner='tt')
        ckt.run_simulator()
        assert mock_ngspice['call_count'] == 2



class TestNetlistHeader:

    def test_set_header(self, empty_netlist):
        """netlist_header() sets the header attribute."""
        empty_netlist.netlist_header('My Test Circuit')
        assert empty_netlist.netlist_header == 'My Test Circuit'



class TestAddSourceBranches:

    def test_source_with_no_params_defaults_dc_to_zero(self, basic_netlist):
        """Adding a VoltageSource with default dc=0."""
        basic_netlist.add_source(sources.VoltageSource('vsup', positive='vdd', negative='gnd'))
        src = basic_netlist.sig_sources_dict['vsup']
        assert isinstance(src, sources.VoltageSource)
        assert src.dc == 0

    def test_source_with_empty_terminals_is_rejected(self, basic_netlist):
        """TASK-76: ``positive=''`` used to be accepted here and emitted
        ``Vvsup  <neg> DC 0`` — one node short, which ngspice mis-parses into
        a phantom node. It now raises at construction, naming the source and
        the terminal, so nothing reaches the module: no source, no ``''`` net."""
        with pytest.raises(ValueError, match=r"VoltageSource 'vsup' terminal 'positive'"):
            basic_netlist.add_source(sources.VoltageSource('vsup', positive='', negative=''))
        assert 'vsup' not in basic_netlist.sig_sources_dict
        assert '' not in basic_netlist.net_dict

    def test_source_with_nets(self, basic_netlist):
        """Adding a VoltageSource with explicit nets."""
        basic_netlist.add_source(
            sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8)
        )
        src = basic_netlist.sig_sources_dict['vsup']
        assert src.positive == 'vdd'
        assert src.negative == 'gnd'

    def test_source_current_type(self, basic_netlist):
        """Adding a CurrentSource stores correctly."""
        basic_netlist.add_source(
            sources.CurrentSource('ibias', positive='vb', negative='gnd', dc=100e-6)
        )
        src = basic_netlist.sig_sources_dict['ibias']
        assert isinstance(src, sources.CurrentSource)
        assert src.dc == 100e-6

    def test_source_debug_mode(self, pdk, tmp_dir, capsys):
        """Adding a source with debug=True prints debug info."""
        ckt = Module('PDK_skywater130', tmp_dir, 'dbg.cir',
                           pdk_details=pdk, debug=True)
        ckt.add_source(sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8))
        captured = capsys.readouterr()
        assert 'Added source' in captured.out



class TestUpdateSourceErrors:

    def test_update_nonexistent_source(self, basic_netlist):
        """update_source raises CircuitError for nonexistent source."""
        with pytest.raises(CircuitError):
            basic_netlist.update_source('nonexistent', ['dc'], [1.0])

    def test_connect_nonexistent_source(self, basic_netlist):
        """connect_source raises CircuitError for nonexistent source."""
        with pytest.raises(CircuitError):
            basic_netlist.connect_source('nonexistent', ['positive'], ['vdd'])

    def test_connect_source_undefined_net(self, basic_netlist):
        """connect_source raises an error when net is undefined or source format is incompatible."""
        basic_netlist.add_source(sources.VoltageSource('vsup', positive='vdd', negative='gnd'))
        with pytest.raises(Exception):
            basic_netlist.connect_source('vsup', ['positive'], ['undefined_net'])



class TestUpdateNet:

    def test_update_existing_net(self, empty_netlist):
        """update_net updates parameters of an existing net."""
        empty_netlist.add_net('vdd')
        empty_netlist.update_net('vdd', ['max_voltage=1.8'])
        assert empty_netlist.net_dict['vdd'].parameters == ['max_voltage=1.8']

    def test_update_nonexistent_net(self, empty_netlist):
        """update_net raises CircuitError for a nonexistent net."""
        with pytest.raises(CircuitError):
            empty_netlist.update_net('nonexistent', ['param=1'])



class TestLoadAndSignalErrors:

    def test_load_file_sets_datafile(self, empty_netlist):
        """load_file stores the dataFile attribute."""
        empty_netlist.load_file('/some/path/data.raw')
        assert empty_netlist.dataFile == '/some/path/data.raw'

    def test_get_signal_names_without_results(self, empty_netlist):
        """get_signal_names raises CircuitError when results not loaded."""
        with pytest.raises(CircuitError, match='not loaded'):
            empty_netlist.get_signal_names()

    def test_load_signal_without_results(self, empty_netlist):
        """load_signal raises CircuitError when results not loaded."""
        with pytest.raises(CircuitError, match='not Loaded'):
            empty_netlist.load_signal('out')





class TestElementDelegation:


    def test_update_element(self, basic_netlist):
        """update_element delegates to _element_mgr."""
        basic_netlist.update_element('mn1', ['w'], [2.0])
        elem = basic_netlist.elem_dict['mn1']
        w_param = elem['device'].param_list['w']
        # param_list values are ParameterConstraint objects; .value is a Variable with .value
        assert w_param.value.value == 2.0



class TestAddNetOwn:

    def test_add_net_duplicate_is_noop(self, empty_netlist):
        """Calling add_net twice on the same net is a silent no-op.

        Module.add_net delegates to PyCircuitElements.add_net, which skips
        silently (or prints only in debug mode) on duplicate names.
        The net should appear exactly once in net_dict.
        """
        empty_netlist.add_net('vdd')
        empty_netlist.add_net('vdd')
        assert list(empty_netlist.net_dict.keys()).count('vdd') == 1

    def test_add_net_stores_parameters(self, empty_netlist):
        """add_net stores parameters on Net.parameters."""
        empty_netlist.add_net('vdd', ['max=1.8'])
        assert empty_netlist.net_dict['vdd'].parameters == ['max=1.8']



class TestPrintElementsWithSources:

    def test_print_elements_includes_sources(self, basic_netlist, capsys):
        """print_elements also prints sources."""
        basic_netlist.add_source(sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8))
        basic_netlist.print_elements()
        captured = capsys.readouterr()
        assert 'vsup' in captured.out


# ---------------------------------------------------------------------------
# generateNetlist_OLD tests
# ---------------------------------------------------------------------------
# _setup_tracking with tracking enabled (mocked)
# ---------------------------------------------------------------------------
class TestReadRawFile:

    def test_read_raw_file_sets_results(self, empty_netlist, monkeypatch):
        """read_raw_file loads simulation results."""
        from unittest.mock import MagicMock

        mock_results = MagicMock()
        mock_results.nvars = 2
        mock_results.vectors = [MagicMock(), MagicMock()]
        # The loader gates on data presence (TASK-116), so the stand-in result
        # has to look like a real analysis plot: two data rows, and a plot name
        # that is not ngspice's synthetic 'constants'.
        mock_results.npoints = 2
        mock_results.current_plot.plotname = 'Transient Analysis'

        import silk.signals.rawfile as rawfile
        def fake_read_rawfile(path, *, lazy):
            assert lazy is True
            return mock_results
        monkeypatch.setattr(rawfile, 'read_rawfile', fake_read_rawfile)

        empty_netlist.dataFile = '/fake/data.raw'
        empty_netlist.read_raw_file()

        assert empty_netlist.sim_results_loaded is True
        assert empty_netlist.simResultsAll is mock_results



class TestLoadSignalWithResults:

    def test_load_signal_finds_signal(self, empty_netlist):
        """load_signal returns a NgspiceSignal when signal exists."""
        from unittest.mock import MagicMock
        import numpy as np

        # Create mock results
        mock_results = MagicMock()
        mock_results.nvars = 2

        time_vec = MagicMock()
        time_vec.name = 'time'
        time_vec.data = np.linspace(0, 1e-6, 10)
        time_vec.type = 'time'

        vout_vec = MagicMock()
        vout_vec.name = 'v(out)'
        vout_vec.data = np.ones(10) * 0.9
        vout_vec.type = 'voltage'

        mock_results.vectors = [time_vec, vout_vec]

        empty_netlist.sim_results_loaded = True
        empty_netlist.simResultsAll = mock_results

        sig = empty_netlist.load_signal('out', 'v')
        assert sig.name == 'out'

    def test_load_signal_not_found(self, empty_netlist):
        """load_signal raises CircuitError when signal not found."""
        from unittest.mock import MagicMock
        import numpy as np

        mock_results = MagicMock()
        mock_results.nvars = 2

        time_vec = MagicMock()
        time_vec.name = 'time'
        time_vec.data = np.linspace(0, 1e-6, 10)
        time_vec.type = 'time'

        other_vec = MagicMock()
        other_vec.name = 'v(other)'
        other_vec.data = np.ones(10)
        other_vec.type = 'voltage'

        mock_results.vectors = [time_vec, other_vec]

        empty_netlist.sim_results_loaded = True
        empty_netlist.simResultsAll = mock_results

        with pytest.raises(CircuitError, match='not found'):
            empty_netlist.load_signal('out', 'v')



class TestGetSignalNames:

    def test_get_signal_names(self, empty_netlist, capsys):
        """get_signal_names returns list of signal names."""
        from unittest.mock import MagicMock

        mock_results = MagicMock()
        mock_results.nvars = 2

        vec1 = MagicMock()
        vec1.name = 'time'
        vec2 = MagicMock()
        vec2.name = 'v(out)'

        mock_results.vectors = [vec1, vec2]

        empty_netlist.sim_results_loaded = True
        empty_netlist.simResultsAll = mock_results

        names = empty_netlist.get_signal_names()
        assert 'time' in names
        assert 'v(out)' in names



class TestGetOutputSignalList:

    def test_raises_when_file_not_loaded(self, empty_netlist):
        """get_output_signal_list raises CircuitError when no file loaded."""
        with pytest.raises(CircuitError, match='not loaded'):
            empty_netlist.get_output_signal_list()

    def test_prints_when_file_available(self, empty_netlist, capsys):
        """get_output_signal_list prints when file is available."""
        empty_netlist.output_file_available = True
        empty_netlist.get_output_signal_list()
        captured = capsys.readouterr()
        assert 'List of signals' in captured.out


# ---------------------------------------------------------------------------
# Tracking-enabled paths with mocked database
# ---------------------------------------------------------------------------
class TestRunSimulatorFailure:

    def test_ngspice_nonzero_raises_simulation_error(self, full_param_netlist, monkeypatch):
        """When ngspice exits with non-zero code, run_simulator raises SimulationError."""
        import subprocess
        from unittest.mock import MagicMock
        from silk.exceptions import SimulationError

        def fake_run_fail(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1
            try:
                o_idx = cmd.index('-o')
                log_path = cmd[o_idx + 1]
                import os
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'w') as f:
                    f.write('Error: something went wrong\nother line\n')
            except (ValueError, IndexError):
                pass
            return mock_result

        monkeypatch.setattr(subprocess, 'run', fake_run_fail)

        full_param_netlist.generate_netlist(corner='tt')
        with pytest.raises(SimulationError, match='ngspice simulation failed'):
            full_param_netlist.run_simulator()

    def test_ngspice_nonzero_no_log_file(self, full_param_netlist, monkeypatch):
        """When ngspice fails and no log file exists, OSError is swallowed and SimulationError raised."""
        import subprocess
        from unittest.mock import MagicMock
        from silk.exceptions import SimulationError

        def fake_run_fail(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 2
            # Do NOT create the log file — exercises the OSError branch
            return mock_result

        monkeypatch.setattr(subprocess, 'run', fake_run_fail)

        full_param_netlist.generate_netlist(corner='tt')
        with pytest.raises(SimulationError, match='ngspice simulation failed'):
            full_param_netlist.run_simulator()

    def test_ngspice_failure_error_lines_in_message(self, full_param_netlist, monkeypatch):
        """Error lines from the log file appear in the SimulationError message."""
        import subprocess
        from unittest.mock import MagicMock
        import os
        from silk.exceptions import SimulationError

        def fake_run_fail(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1
            try:
                o_idx = cmd.index('-o')
                log_path = cmd[o_idx + 1]
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'w') as f:
                    f.write('ERROR: undefined device\nfatal error\n')
            except (ValueError, IndexError):
                pass
            return mock_result

        monkeypatch.setattr(subprocess, 'run', fake_run_fail)

        full_param_netlist.generate_netlist(corner='tt')
        with pytest.raises(SimulationError) as exc_info:
            full_param_netlist.run_simulator()
        assert 'undefined device' in str(exc_info.value)


# ---------------------------------------------------------------------------
# Lines 615–633: store_results() — detailed coverage
# ---------------------------------------------------------------------------
class TestPrintMethodsFormat:

    def test_print_elements_shows_count(self, basic_netlist, capsys):
        """print_elements() header includes device and source counts."""
        basic_netlist.print_elements()
        captured = capsys.readouterr()
        assert 'Elements' in captured.out
        assert 'devices' in captured.out
        assert 'sources' in captured.out

    def test_print_elements_empty_netlist(self, empty_netlist, capsys):
        """print_elements() on empty netlist shows zero counts."""
        empty_netlist.print_elements()
        captured = capsys.readouterr()
        assert '0 devices' in captured.out
        assert '0 sources' in captured.out

    def test_print_nets_shows_count(self, basic_netlist, capsys):
        """print_nets() header includes net count."""
        basic_netlist.print_nets()
        captured = capsys.readouterr()
        assert 'Nets' in captured.out
        assert '4' in captured.out  # basic_netlist has vdd, gnd, in, out

    def test_print_nets_empty_netlist(self, empty_netlist, capsys):
        """print_nets() on empty netlist shows zero count."""
        empty_netlist.print_nets()
        captured = capsys.readouterr()
        assert 'Nets (0)' in captured.out

    def test_print_elements_shows_source_nets(self, basic_netlist, capsys):
        """print_elements() shows net connections for sources."""
        basic_netlist.add_source(sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8))
        basic_netlist.print_elements()
        captured = capsys.readouterr()
        assert 'vsup' in captured.out



# ---------------------------------------------------------------------------
# print_structure — lines 750–830
# ---------------------------------------------------------------------------
class TestSourceNetsControlledSources:

    def test_vcvs_nets_registered(self, empty_netlist):
        """VCVS source registers all four net names (out_pos, out_neg, ctrl_pos, ctrl_neg)."""
        src = sources.VoltageControlledVoltageSource(
            'eamp', out_pos='vout', out_neg='gnd', ctrl_pos='vin', ctrl_neg='gnd', gain=10
        )
        empty_netlist.add_source(src)
        # _source_nets should return the four nets, and add_source registers them
        assert 'eamp' in empty_netlist.sig_sources_dict

    def test_vccs_nets_registered(self, empty_netlist):
        """VCCS source registers all four net names."""
        src = sources.VoltageControlledCurrentSource(
            'gm1', out_pos='iout', out_neg='gnd', ctrl_pos='vin', ctrl_neg='gnd', gm=0.01
        )
        empty_netlist.add_source(src)
        assert 'gm1' in empty_netlist.sig_sources_dict

    def test_ccvs_nets_registered(self, empty_netlist):
        """CCVS source registers out_pos and out_neg (two-net branch, line 237)."""
        src = sources.CurrentControlledVoltageSource(
            'htrans', out_pos='vout', out_neg='gnd', transresistance=1000, ctrl_vname='vmeas'
        )
        empty_netlist.add_source(src)
        assert 'htrans' in empty_netlist.sig_sources_dict

    def test_cccs_nets_registered(self, empty_netlist):
        """CCCS source registers out_pos and out_neg."""
        src = sources.CurrentControlledCurrentSource(
            'fmir', out_pos='icopy', out_neg='gnd', gain=2, ctrl_vname='vmeas'
        )
        empty_netlist.add_source(src)
        assert 'fmir' in empty_netlist.sig_sources_dict

    def test_vcvs_four_nets_in_net_dict(self, empty_netlist):
        """Adding a VCVS registers all four referenced nets into net_dict."""
        src = sources.VoltageControlledVoltageSource(
            'etest', out_pos='out_p', out_neg='out_n', ctrl_pos='ctrl_p', ctrl_neg='ctrl_n', gain=5
        )
        empty_netlist.add_source(src)
        for net in ('out_p', 'out_n', 'ctrl_p', 'ctrl_n'):
            assert net in empty_netlist.net_dict

    def test_ccvs_two_nets_in_net_dict(self, empty_netlist):
        """Adding a CCVS registers out_pos and out_neg into net_dict."""
        src = sources.CurrentControlledVoltageSource(
            'h1', out_pos='vo_p', out_neg='vo_n', transresistance=500, ctrl_vname='vmeas'
        )
        empty_netlist.add_source(src)
        assert 'vo_p' in empty_netlist.net_dict
        assert 'vo_n' in empty_netlist.net_dict



class TestAddNetDelegate:

    def test_addnet_via_netlist(self, empty_netlist):
        """PyNetlist.add_net() delegates to element manager and adds to net_dict."""
        empty_netlist.add_net('mynet')
        assert 'mynet' in empty_netlist.net_dict

    def test_addnet_returns_none_for_duplicate(self, empty_netlist):
        """PyNetlist.add_net() for a duplicate net does not raise."""
        empty_netlist.add_net('dupnet')
        empty_netlist.add_net('dupnet')  # should not raise
        assert 'dupnet' in empty_netlist.net_dict



class TestUpdateSource:

    def _inject_legacy_source(self, ckt, name):
        """Inject a legacy dict-format source entry for testing old API paths."""
        # sigParams order: ['dc', 'ac', 'pwl', 'pulse', 'sine']
        ckt.sig_sources_dict[name] = {
            'type': 'VoltageSource',
            'pinNets': ['vdd', 'gnd'],
            'params': [1.8, None, None, None, None],
        }

    def test_update_source_modifies_params(self, empty_netlist):
        """update_source() modifies a parameter value in a legacy dict-format source (lines 271-287)."""
        self._inject_legacy_source(empty_netlist, 'vsrc_legacy')
        empty_netlist.update_source('vsrc_legacy', ['dc'], [3.3])
        assert empty_netlist.sig_sources_dict['vsrc_legacy']['params'][0] == 3.3

    def test_update_source_missing_raises_exit(self, empty_netlist):
        """update_source() raises CircuitError when source does not exist."""
        with pytest.raises(CircuitError):
            empty_netlist.update_source('nonexistent', ['dc'], [1.8])



class TestConnectSource:

    def _inject_legacy_source(self, ckt, name):
        """Inject a legacy dict-format source entry for testing old API paths."""
        ckt.sig_sources_dict[name] = {
            'type': 'VoltageSource',
            'pinNets': ['', ''],
            'params': [1.8, None, None, None, None],
        }

    def test_connect_source_updates_pin_nets(self, empty_netlist):
        """connect_source() updates pin nets in a legacy dict-format source (lines 302-307)."""
        empty_netlist.add_net('vdd')
        empty_netlist.add_net('gnd')
        self._inject_legacy_source(empty_netlist, 'vsrc_conn')
        empty_netlist.connect_source('vsrc_conn', ['positive', 'negative'], ['vdd', 'gnd'])
        assert empty_netlist.sig_sources_dict['vsrc_conn']['pinNets'][0] == 'vdd'
        assert empty_netlist.sig_sources_dict['vsrc_conn']['pinNets'][1] == 'gnd'

    def test_connect_source_missing_source_raises_exit(self, empty_netlist):
        """connect_source() raises CircuitError when source does not exist."""
        with pytest.raises(CircuitError):
            empty_netlist.connect_source('nonexistent', ['positive'], ['vdd'])

    def test_connect_source_undefined_net_raises_exit(self, empty_netlist):
        """connect_source() raises CircuitError when net is not in net_dict."""
        self._inject_legacy_source(empty_netlist, 'vsrc_nonet')
        with pytest.raises(CircuitError):
            empty_netlist.connect_source('vsrc_nonet', ['positive'], ['undefined_net'])



class TestPrintElementsControlledSources:

    def test_print_elements_with_vcvs_source(self, empty_netlist, capsys):
        """print_elements() handles a VCVS source (uses out_pos/out_neg branch, line 338-339)."""
        src = sources.VoltageControlledVoltageSource(
            'eprint', out_pos='outp', out_neg='outn', ctrl_pos='ctrlp', ctrl_neg='ctrln', gain=2
        )
        empty_netlist.add_source(src)
        empty_netlist.print_elements()
        out = capsys.readouterr().out
        assert 'eprint' in out

    def test_print_elements_with_ccvs_source(self, empty_netlist, capsys):
        """print_elements() handles a CCVS source (uses out_pos/out_neg branch)."""
        src = sources.CurrentControlledVoltageSource(
            'hprint', out_pos='vp', out_neg='vn', transresistance=100, ctrl_vname='vctrl'
        )
        empty_netlist.add_source(src)
        empty_netlist.print_elements()
        out = capsys.readouterr().out
        assert 'hprint' in out


# ---------------------------------------------------------------------------
# Coverage gap: run_simulator() debug=True branch — line 490
# ---------------------------------------------------------------------------
class TestStrictNets:

    def test_default_is_false_auto_registers_nets(self, pdk, tmp_dir):
        """Default strict_nets=False: add_element with undeclared nets does not raise."""
        ckt = Module('PDK_skywater130', tmp_dir, 'default.cir', pdk_details=pdk)
        # No add_net calls — should succeed silently
        ckt.add_element(
            pdk.nmos1p8, 'mn1',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters={'w': 1, 'l': 0.15},
        )
        assert 'mn1' in ckt.elem_dict
        for net in ('out', 'in', 'gnd'):
            assert net in ckt.net_dict

    def test_explicit_false_auto_registers_nets(self, pdk, tmp_dir):
        """Explicit strict_nets=False: add_element with undeclared nets does not raise."""
        ckt = Module('PDK_skywater130', tmp_dir, 'explicit_false.cir',
                           pdk_details=pdk, strict_nets=False)
        ckt.add_element(
            pdk.nmos1p8, 'mn1',
            nets={'d': 'undeclared_net', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters={'w': 1, 'l': 0.15},
        )
        assert 'undeclared_net' in ckt.net_dict

    def test_strict_true_addelement_undeclared_net_raises(self, pdk, tmp_dir):
        """strict_nets=True: add_element with undeclared net raises CircuitError."""
        ckt = Module('PDK_skywater130', tmp_dir, 'strict.cir',
                           pdk_details=pdk, strict_nets=True)
        with pytest.raises(CircuitError, match='mystery_net'):
            ckt.add_element(
                pdk.nmos1p8, 'mn1',
                nets={'d': 'mystery_net', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                parameters={'w': 1, 'l': 0.15},
            )

    def test_strict_true_addelement_predeclared_nets_succeeds(self, pdk, tmp_dir):
        """strict_nets=True: add_element succeeds when all nets are pre-declared."""
        ckt = Module('PDK_skywater130', tmp_dir, 'strict_ok.cir',
                           pdk_details=pdk, strict_nets=True)
        for net in ('out', 'in', 'gnd'):
            ckt.add_net(net)
        ckt.add_element(
            pdk.nmos1p8, 'mn1',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters={'w': 1, 'l': 0.15},
        )
        assert 'mn1' in ckt.elem_dict

    def test_strict_true_addsource_undeclared_net_raises(self, pdk, tmp_dir):
        """strict_nets=True: add_source with undeclared net raises CircuitError."""
        ckt = Module('PDK_skywater130', tmp_dir, 'strict_src.cir',
                           pdk_details=pdk, strict_nets=True)
        with pytest.raises(CircuitError, match='undeclared_vdd'):
            ckt.add_source(
                sources.VoltageSource('vsup', positive='undeclared_vdd', negative='gnd', dc=1.8)
            )

    def test_strict_true_addsource_predeclared_nets_succeeds(self, pdk, tmp_dir):
        """strict_nets=True: add_source succeeds when all nets are pre-declared."""
        ckt = Module('PDK_skywater130', tmp_dir, 'strict_src_ok.cir',
                           pdk_details=pdk, strict_nets=True)
        ckt.add_net('vdd')
        ckt.add_net('gnd')
        ckt.add_source(
            sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8)
        )
        assert 'vsup' in ckt.sig_sources_dict

    def test_strict_true_addnet_then_addelement_succeeds(self, pdk, tmp_dir):
        """strict_nets=True: round-trip add_net then add_element works."""
        ckt = Module('PDK_skywater130', tmp_dir, 'strict_rt.cir',
                           pdk_details=pdk, strict_nets=True)
        for net in ('vdd', 'gnd', 'in', 'out'):
            ckt.add_net(net)
        ckt.add_element(
            pdk.pmos1p8, 'mp1',
            nets={'d': 'out', 'g': 'in', 's': 'vdd', 'b': 'vdd'},
            parameters={'w': 2, 'l': 0.15},
        )
        assert 'mp1' in ckt.elem_dict

    def test_strict_nets_flag_stored_true(self, pdk, tmp_dir):
        """strict_nets=True is stored on the instance."""
        ckt = Module('PDK_skywater130', tmp_dir, 'flag_true.cir',
                           pdk_details=pdk, strict_nets=True)
        assert ckt.strict_nets is True

    def test_strict_nets_flag_stored_false(self, pdk, tmp_dir):
        """strict_nets=False is stored on the instance."""
        ckt = Module('PDK_skywater130', tmp_dir, 'flag_false.cir',
                           pdk_details=pdk, strict_nets=False)
        assert ckt.strict_nets is False

    def test_error_message_contains_net_name(self, pdk, tmp_dir):
        """CircuitError message contains the offending net name."""
        ckt = Module('PDK_skywater130', tmp_dir, 'strict_msg.cir',
                           pdk_details=pdk, strict_nets=True)
        with pytest.raises(CircuitError, match='net_alpha'):
            ckt.add_element(
                pdk.nmos1p8, 'mn1',
                nets={'d': 'net_alpha', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                parameters={'w': 1, 'l': 0.15},
            )


class TestSimulateCornerValidation:
    """Task 5: simulate() defaults corner from pdk.default_corner and
    validates it against pdk.corners, but ONLY when pdk.corners is
    non-empty (legacy/manifest-less PDKs must keep pass-through behavior).
    """

    def test_simulate_rejects_unknown_corner_when_corners_declared(self, tmp_path):
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK('silk/pdk/data/skywater_130_device_details.yaml',
                              library_location='/dev/null',
                              corners={'tt': {}, 'ff': {}}, default_corner='tt')
        m = Module('PDK_skywater130', str(tmp_path), 'x.cir', pdk_details=pdk, module_name=None)
        with pytest.raises(ValueError) as e:
            m.simulate(corner='zz')
        assert 'zz' in str(e.value) and 'tt' in str(e.value)

    def test_simulate_empty_corners_skips_validation(self, tmp_path, monkeypatch):
        # legacy path: corners={} -> any corner passes validation (still fails later
        # for other reasons, but NOT with 'unknown corner').
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK('silk/pdk/data/skywater_130_device_details.yaml',
                              library_location='/dev/null')   # corners == {}
        m = Module('PDK_skywater130', str(tmp_path), 'x.cir', pdk_details=pdk, module_name=None)
        # stub the actual run so we only exercise the validation gate
        monkeypatch.setattr(m, 'generate_netlist', lambda corner=None: None)
        monkeypatch.setattr(m, 'run_simulator', lambda **k: None)
        monkeypatch.setattr(m, 'load_file', lambda *a: None)
        monkeypatch.setattr(m, 'read_raw_file', lambda: None)
        m.simulate(corner='anything')   # must NOT raise 'unknown corner'
