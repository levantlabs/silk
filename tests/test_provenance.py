"""Provenance on every result (TASK-306).

Pins the record's contract (JSON-plain fields, round trip, digest/diff that
ignore the timestamp), its privacy rule (no hostname, username or absolute
home path -- including inside option values), the ``.lib`` hashes, the
circuit-report block and diff section, the schema-v4 storage and the
v3-to-v4 copy migration, and -- through real ngspice -- the run_origin rule:
a silk launch names the binary it launched, a handed-in rawfile names none.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import socket
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from silk.simulation import provenance
from silk.simulation.provenance import FIELDS, RunProvenance, SCHEME
from tests._env import needs_ngspice, needs_pdk


def _record(**overrides):
    base = dict(
        run_origin='silk-launch', silk_version='0.2.0', silk_git='abc1234',
        python_version='3.12.3', numpy_version='2.5.3', simulator='ngspice',
        simulator_binary='/usr/bin/ngspice', simulator_binary_source='PATH',
        simulator_version='ngspice-42', pdk_name='sky130A',
        pdk_libraries=[{'file': 'sky130A/libs.tech/ngspice/sky130.lib.spice',
                        'section': 'tt', 'file_sha256': 'a' * 64,
                        'section_text_sha256': 'b' * 64}],
        corner='tt', temperature_c=27.0, temperature_kind='fixed',
        supply_voltage_v=1.8, simulator_options={'reltol': 1e-4},
        save_list=None, deck_sha256='c' * 64, seed=None, seedinfo=None,
        timestamp_utc='2026-09-29T00:00:00Z',
    )
    base.update(overrides)
    return RunProvenance(**base)


# ---------------------------------------------------------------------------
# Record contract
# ---------------------------------------------------------------------------

def _assert_json_plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for v in value:
            _assert_json_plain(v)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str)
            _assert_json_plain(v)
        return
    raise AssertionError(f'not JSON-plain: {value!r} ({type(value).__name__})')


def test_to_dict_is_json_plain_and_round_trips_through_json():
    rec = _record(simulator_options={'reltol': np.float64(1e-4),
                                     'steps': (1, 2)}, seed=np.int64(7))
    d = rec.to_dict()
    assert list(d) == list(FIELDS)
    assert d['scheme'] == SCHEME == 'silk.provenance/v3'
    _assert_json_plain(d)
    back = RunProvenance.from_dict(json.loads(json.dumps(d)))
    assert back.to_dict() == d
    assert back.digest() == rec.digest()


def test_from_dict_ignores_unknown_keys_and_defaults_missing_ones():
    rec = RunProvenance.from_dict({'run_origin': 'silk-launch', 'future_field': 1})
    assert rec.run_origin == 'silk-launch'
    assert rec.deck_sha256 is None and rec.scheme == SCHEME


def test_digest_and_diff_ignore_the_timestamp_only():
    a = _record()
    b = _record(timestamp_utc='2027-01-01T00:00:00Z')
    assert a.digest() == b.digest()
    assert a.diff(b) == []
    c = _record(simulator_version='ngspice-47', deck_sha256='d' * 64)
    assert a.diff(c) == [('simulator_version', 'ngspice-42', 'ngspice-47'),
                         ('deck_sha256', 'c' * 64, 'd' * 64)]
    assert a.digest() != c.digest()


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

class _PreparedStub:
    def __init__(self, options, save_list=None):
        self.simulator_options = options
        self._save_list = save_list


def test_no_hostname_username_or_home_path_anywhere_in_the_record(tmp_path, monkeypatch):
    home = os.path.expanduser('~')
    pdk_root = tmp_path / 'pdkroot'
    lib_dir = pdk_root / 'sky130A' / 'libs.tech' / 'ngspice'
    lib_dir.mkdir(parents=True)
    lib = lib_dir / 'sky130.lib.spice'
    lib.write_text('.lib tt\n.include "x.spice"\n.endl tt\n')
    monkeypatch.setenv('PDK_ROOT', str(pdk_root))
    home_lib = f'{home}/private-models/extra.lib'
    deck = (f'* deck\n.lib "{lib}" tt\n.lib "{home_lib}" ss\n'
            f'.lib "/scratch/{getpass.getuser()}/models.lib" ff\n')
    prepared = provenance.prepared_deck_record(
        _PreparedStub({'reltol': 1e-4,
                       'include_dir': f'{home}/work/models',
                       'nested': {'paths': [f'{home}/a.spice', '/usr/share/b']}},
                      save_list={'signals': [f'{home}/sig'], 'exclude': None,
                                 'max_depth': None}),
        deck)
    launch = {'simulator': 'ngspice',
              'simulator_binary': provenance.share_path(f'{home}/ngspice-47/bin/ngspice'),
              'simulator_binary_source': 'SILK_NGSPICE_BIN',
              'simulator_version': 'ngspice-47'}
    rec = provenance.build(run_origin='silk-launch', launch=launch,
                           prepared=prepared, pdk_name='sky130A', conditions=None)
    text = json.dumps(rec.to_dict())

    assert home not in text
    assert getpass.getuser() not in text
    hostname = socket.gethostname()
    if len(hostname) > 3:
        assert hostname not in text
    assert rec.simulator_binary == '~/ngspice-47/bin/ngspice'
    assert rec.simulator_options['include_dir'] == '~/work/models'
    assert rec.simulator_options['nested']['paths'] == ['~/a.spice', '/usr/share/b']
    assert rec.save_list['signals'] == ['~/sig']
    files = [e['file'] for e in rec.pdk_libraries]
    assert files == ['sky130A/libs.tech/ngspice/sky130.lib.spice',
                     '~/private-models/extra.lib', 'models.lib']


def test_home_path_is_rewritten_mid_string_and_in_nested_lists():
    """Every occurrence, not just a leading one: include flags and key=value
    strings embed the home path after a prefix."""
    home = os.path.expanduser('~')
    user = getpass.getuser()
    prepared = provenance.prepared_deck_record(
        _PreparedStub({
            'flags': f'-I{home}/inc -L{home}/lib',
            'osdi': f'lib={home}/a.osdi',
            'includes': [[f'-I{home}/x', ('-I/usr/include', f'{home}')],
                         {'deep': [f'x:{home}/y']}],
            f'{home}/key': 1,
        }, save_list={'signals': [f'v({home}/n)'], 'exclude': [[f'{home}']],
                      'max_depth': None}),
        '')
    opts, save = prepared['simulator_options'], prepared['save_list']
    assert opts['flags'] == '-I~/inc -L~/lib'
    assert opts['osdi'] == 'lib=~/a.osdi'
    assert opts['includes'] == [['-I~/x', ['-I/usr/include', '~']],
                                {'deep': ['x:~/y']}]
    assert opts['~/key'] == 1
    assert save['signals'] == ['v(~/n)'] and save['exclude'] == [['~']]
    text = json.dumps(prepared)
    assert home not in text and user not in text


def test_home_rewrite_respects_the_path_boundary(monkeypatch):
    monkeypatch.setenv('HOME', '/home/user')
    assert provenance.home_relative('/home/userX/a') == '/home/userX/a'
    assert provenance.home_relative('-I/home/user -I/home/userX') == '-I~ -I/home/userX'
    assert provenance.home_relative('a=/home/user/b;c=/home/user') == 'a=~/b;c=~'
    for tail in ('X', '2', '.bak', '_old', '-x'):
        assert provenance.home_relative(f'/home/user{tail}/f') == f'/home/user{tail}/f'
    assert provenance.home_relative('"/home/user",/home/user:x') == '"~",~:x'
    assert provenance.scrub({'k': ['/home/users/x', '/home/user/x']}) == {
        'k': ['/home/users/x', '~/x']}


def test_share_path_rules(monkeypatch, tmp_path):
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    home = os.path.expanduser('~')
    assert provenance.share_path('/usr/bin/ngspice') == '/usr/bin/ngspice'
    assert provenance.share_path(f'{home}/x/y') == '~/x/y'
    assert provenance.share_path('/data/someone/models.lib') == 'models.lib'
    assert provenance.share_path(str(tmp_path / 'a' / 'b.lib'),
                                 pdk_relative=True) == 'a/b.lib'


# ---------------------------------------------------------------------------
# .lib hashes
# ---------------------------------------------------------------------------

def test_lib_hashes_name_one_file_and_one_section_text(tmp_path, monkeypatch):
    monkeypatch.setenv('PDK_ROOT', str(tmp_path))
    body_tt = '.param a = 1\n.include "tt_models.spice"\n'
    content = ('* header\n.lib tt\n' + body_tt + '.endl tt\n'
               '.lib ss\n.param a = 2\n.endl ss\n')
    lib = tmp_path / 'models.lib'
    lib.write_text(content)
    (entry,) = provenance.deck_libraries(f'.lib "{lib}" tt\n.include "other"\n')
    assert entry == {
        'file': 'models.lib', 'section': 'tt',
        'file_sha256': hashlib.sha256(content.encode()).hexdigest(),
        'section_text_sha256': hashlib.sha256(body_tt.encode()).hexdigest(),
    }
    # The section hash covers only that section's own text: editing the
    # file it includes changes neither hash (documented limitation).
    (tmp_path / 'tt_models.spice').write_text('changed')
    (again,) = provenance.deck_libraries(f'.lib "{lib}" tt\n')
    assert again == entry


def test_missing_lib_file_records_the_line_with_no_hashes():
    (entry,) = provenance.deck_libraries('.lib "/nonexistent/x.lib" tt\n')
    assert entry['section'] == 'tt'
    assert entry['file_sha256'] is None and entry['section_text_sha256'] is None


def test_seed_comes_from_the_seed_option_and_seedinfo_stays_none():
    prepared = provenance.prepared_deck_record(_PreparedStub({'seed': 42}), '')
    rec = provenance.build(run_origin='silk-launch', launch=None,
                           prepared=prepared, pdk_name=None, conditions=None)
    assert rec.seed == 42 and rec.seedinfo is None


# ---------------------------------------------------------------------------
# Cached environment facts never raise
# ---------------------------------------------------------------------------

def test_git_describe_uses_the_checkout_root(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(provenance.shutil, 'which', lambda name: '/usr/bin/git')
    calls = []

    def output(argv, timeout):
        calls.append(argv)
        return 'abc1234-dirty\n'

    monkeypatch.setattr(provenance, '_first_output', output)
    assert provenance._git_describe() == 'abc1234-dirty'
    assert calls == [['/usr/bin/git', '-C', str(root),
                      'describe', '--always', '--dirty']]


def test_git_describe_is_none_when_git_is_missing(monkeypatch):
    monkeypatch.setattr(provenance.shutil, 'which', lambda name: None)
    assert provenance._git_describe() is None


def test_git_describe_is_none_when_git_fails(monkeypatch):
    monkeypatch.setattr(provenance, '_first_output', lambda *a, **k: None)
    assert provenance._git_describe() is None


def test_launch_record_for_a_missing_binary_is_not_an_error(monkeypatch):
    monkeypatch.setenv('SILK_NGSPICE_BIN', '/nonexistent/ngspice')
    rec = provenance.launch_record('ngspice')
    assert rec['simulator_binary_source'] == 'SILK_NGSPICE_BIN'
    assert rec['simulator_binary'] is None and rec['simulator_version'] is None


# ---------------------------------------------------------------------------
# SpecReport, circuit report and diff
# ---------------------------------------------------------------------------

def test_spec_report_provenance_defaults_to_none_and_summary_shows_it():
    from silk.circuit.spec_report import SpecReport
    bare = SpecReport('t', 'm', {'x': 1.0})
    assert bare.provenance is None
    assert 'Run details:' not in bare.summary()
    rep = SpecReport('t', 'm', {'x': 1.0}, provenance=_record())
    lines = rep.summary().splitlines()
    line = lines[2]
    assert line == f'  Run details: {rep.provenance.summary_line()}'
    assert lines[:2] + lines[3:] == bare.summary().splitlines()
    assert 'silk 0.2.0 (abc1234)' in line and 'ngspice-42' in line
    assert 'sky130.lib.spice[tt]' in line


def _circuit_report(records):
    from silk.circuit.interface_check import AnnotationCoverage
    from silk.circuit.report import CircuitReport, TopologySummary
    return CircuitReport(
        module_name='m', generated_at='2026-09-29T00:00:00',
        topology=TopologySummary(module_name='m', instances=[], total_nets=0),
        annotation=AnnotationCoverage(total=0, annotated=0, unannotated_ports=[]),
        violations=[], simulation=None, provenance=list(records))


def test_circuit_report_json_round_trip_and_html_markdown_blocks(tmp_path):
    from silk.circuit.report import CircuitReport
    report = _circuit_report([_record()])
    path = tmp_path / 'r.json'
    report.save_json(str(path))
    on_disk = json.loads(path.read_text())
    assert on_disk['provenance'][0]['deck_sha256'] == 'c' * 64
    back = CircuitReport.load_json(str(path))
    assert back.provenance == report.provenance
    html = report._to_html()
    line = 'Run details: silk 0.2.0, ngspice-42, sky130A (sky130.lib.spice aaaaaaaaaa)'
    for output in (html, report._to_markdown()):
        assert line in output
        assert 'Provenance' not in output
        assert 'section_text_sha256' not in output and 'deck_sha256' not in output
    assert on_disk['provenance'][0] == _record().to_dict()


def test_generate_report_collects_distinct_records_from_spec_reports(pdk, tmp_path):
    from silk import Module
    from silk.circuit.report import generate_report
    from silk.circuit.spec_report import SpecReport
    mod = Module('sky130', str(tmp_path), 'm.cir', pdk_details=pdk, module_name='m')
    mod.add_net('out')
    a, b = _record(), _record(corner='ss')
    reports = [SpecReport('t', 'm', {}, provenance=rec)
               for rec in (a, _record(timestamp_utc='later'), b)]
    reports.append(SpecReport('t', 'm', {}))  # hand-built: no record
    built = generate_report(mod, auto_discover_config=False, spec_reports=reports)
    assert built.provenance == [a, b]
    for output in (built._to_html(), built._to_markdown()):
        assert output.count('Run details:') == 1
        assert 'silk 0.2.0, ngspice-42, sky130A (sky130.lib.spice aaaaaaaaaa)' in output
    path = tmp_path / 'distinct.json'
    built.save_json(path)
    assert json.loads(path.read_text())['provenance'] == [a.to_dict(), b.to_dict()]
    bare = generate_report(mod, auto_discover_config=False)
    assert bare.provenance == []
    assert 'Run details:' not in bare._to_html()


def test_older_report_json_without_provenance_still_loads(tmp_path):
    from silk.circuit.report import CircuitReport
    path = tmp_path / 'r.json'
    _circuit_report([_record()]).save_json(str(path))
    d = json.loads(path.read_text())
    del d['provenance']
    path.write_text(json.dumps(d))
    assert CircuitReport.load_json(str(path)).provenance == []


def test_diff_reports_lists_differing_provenance_fields_by_default():
    from silk.circuit.report_diff import diff_reports
    a = _circuit_report([_record()])
    b = _circuit_report([_record(simulator_version='ngspice-47',
                                 timestamp_utc='2027-01-01T00:00:00Z')])
    diff = diff_reports(a, b)
    assert 'provenance' in diff.sections_compared
    assert diff.provenance_diff.changed == [
        {'record': 0, 'field': 'simulator_version',
         'a': 'ngspice-42', 'b': 'ngspice-47'}]
    assert "simulator_version: 'ngspice-42' -> 'ngspice-47'" in str(diff)
    assert '## Provenance' in diff._to_markdown()
    same = diff_reports(a, _circuit_report([_record(timestamp_utc='x')]))
    assert same.provenance_diff.changed == []
    skipped = diff_reports(a, b, sections={'topology'})
    assert skipped.provenance_diff is None


# ---------------------------------------------------------------------------
# Database: schema v4 storage and the v3-to-v4 copy migration
# ---------------------------------------------------------------------------

def test_store_report_stamps_every_row_with_one_provenance_row(pdk, tmp_path):
    from silk.circuit.spec_report import SpecReport
    from tests.db.test_unavailable_measurements import _build
    mod, (spec,) = _build(pdk, tmp_path, ':memory:')
    try:
        rec = _record()
        report = SpecReport('ac', 'review', {'ugf': 10.0}, corner='tt',
                            provenance=rec)
        report.check(spec.name, spec.metric, spec.op, spec.target,
                     spec_key=spec.key, owner_path='$')
        mod.store_report(report)
        db = mod._tracker.db
        (count,) = db.conn.execute('SELECT COUNT(*) FROM run_provenance').fetchone()
        assert count == 1
        ids = {r[0] for r in db.conn.execute(
            'SELECT provenance_id FROM simulation_results UNION '
            'SELECT provenance_id FROM spec_measurements')}
        assert len(ids) == 1 and None not in ids
        (pid,) = ids
        assert db.get_run_provenance(pid) == rec
        results = db.get_simulation_results(mod._tracker.circuit_id)
        assert {r.provenance_id for r in results} == {pid}
        # Same record (a later timestamp) reuses the same row.
        assert db.store_run_provenance(
            _record(timestamp_utc='2027-01-01T00:00:00Z')) == pid
    finally:
        mod.close()


def test_report_without_provenance_stores_null(pdk, tmp_path):
    from tests.db.test_unavailable_measurements import _build, _report
    mod, specs = _build(pdk, tmp_path, ':memory:')
    try:
        mod.store_report(_report(specs, [10.0]))
        db = mod._tracker.db
        assert db.conn.execute('SELECT COUNT(*) FROM run_provenance').fetchone()[0] == 0
        assert {r[0] for r in db.conn.execute(
            'SELECT provenance_id FROM simulation_results')} == {None}
    finally:
        mod.close()


class _V3Database:
    """A genuine schema-v3 file, created the way silk 0.2 created one."""

    @staticmethod
    def create(path, schema_version=3):
        from silk.db import Database

        class V3(Database):
            def _initialize_schema(self):
                super()._initialize_schema(schema_version=schema_version)

        db = V3(str(path))
        db.close()


def test_v3_open_is_refused_and_names_the_migration(tmp_path):
    from silk.db import Database
    from silk.exceptions import UnsupportedSchemaVersionError
    source = tmp_path / 'v3.db'
    _V3Database.create(source)
    before = source.read_bytes()
    with pytest.raises(UnsupportedSchemaVersionError, match='migrate_v3_to_v4'):
        Database(str(source))
    assert source.read_bytes() == before


def test_migrate_v3_to_v4_keeps_rows_and_adds_null_provenance(tmp_path):
    from silk.db import Database, migrate_v3_to_v4, migrate_v4_to_v5, migrate_v5_to_v6, migrate_v6_to_v7
    source = tmp_path / 'v3.db'
    _V3Database.create(source)
    with sqlite3.connect(source) as conn:
        conn.execute(
            "INSERT INTO circuit_versions (version_id, circuit_name, version, "
            "stage, state, structure_scheme, structure_hash, spec_scheme, "
            "spec_set_hash) VALUES ('v1', 'c', '1', 'dev', 'sealed', 's', 'h', "
            "'s', 'h')")
        conn.execute(
            "INSERT INTO simulation_results (sim_id, circuit_version_id, source, "
            "parameter, value) VALUES ('r1', 'v1', 'spice', 'gain', 3.0)")
    before = source.read_bytes()
    dest = tmp_path / 'v4.db'
    assert migrate_v3_to_v4(source, dest) == str(dest)
    assert source.read_bytes() == before
    # Current silk opens v7, so the v4 copy is read directly.
    with sqlite3.connect(dest) as db:
        row = db.execute(
            "SELECT value, provenance_id FROM simulation_results WHERE sim_id = 'r1'"
        ).fetchone()
        assert tuple(row) == (3.0, None)
        assert db.execute('SELECT COUNT(*) FROM run_provenance').fetchone()[0] == 0
        version = db.execute('SELECT schema_version FROM silk_metadata').fetchone()[0]
        assert version == 4
        # A migrated file and a fresh v4 one declare the same tables.
        fresh = tmp_path / 'fresh.db'
        _V3Database.create(fresh, schema_version=4)
        with sqlite3.connect(fresh) as f:
            fresh_sql = dict(f.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table'"))
        migrated_sql = dict(db.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall())
        assert migrated_sql == fresh_sql
    final = tmp_path / 'v5.db'
    migrate_v4_to_v5(dest, final)
    current = tmp_path / "v6.db"
    v6 = tmp_path / 'migration-v6.db'
    migrate_v5_to_v6(final, v6)
    migrate_v6_to_v7(v6, current)
    db = Database(str(current))
    try:
        assert tuple(db.conn.execute(
            "SELECT value, provenance_id FROM simulation_results WHERE sim_id = 'r1'"
        ).fetchone()) == (3.0, None)
    finally:
        db.close()
    with pytest.raises(FileExistsError):
        migrate_v3_to_v4(source, dest)


# ---------------------------------------------------------------------------
# run_origin through real ngspice + sky130
# ---------------------------------------------------------------------------

@needs_ngspice
@needs_pdk
def test_run_is_silk_launch_and_handed_in_rawfile_claims_no_binary(tmp_path, monkeypatch):
    from silk import sources
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.pdk import load_pdk
    from silk.simulation import analysis

    pdk = load_pdk('sky130')
    inv = Module('PDK_skywater130', str(tmp_path), 'inv.cir',
                 pdk_details=pdk, module_name='inv')
    inv.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with inv:
        pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1.0, l=0.15)
        pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2.0, l=0.15)
    tb = inv.create_test('tran')
    tb.add_module_under_test({'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'})
    tb.add(sources.vdc('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add(sources.vpulse('vin', positive='vin', negative='0', v1=0.0, v2=1.8,
                          td=1e-9, tr=100e-12, tf=100e-12, pw=2e-9, per=4e-9))
    tb.add_sim_type(analysis.Transient(step=10e-12, stop=4e-9))
    tb.set_simulator_options(reltol=1e-4)
    tb.add_metric('vout_max', lambda ckt: float(np.max(ckt.load_signal('vout').y)))

    launched = tb.run('tt').provenance
    deck_bytes = Path(tb.netlist._run_files.netlist_path).read_bytes()
    prepared_sha = hashlib.sha256(deck_bytes).hexdigest()
    assert launched.run_origin == 'silk-launch'
    assert launched.simulator == 'ngspice'
    assert launched.simulator_version and launched.simulator_version.startswith('ngspice-')
    assert launched.simulator_binary and launched.simulator_binary_source in (
        'PATH', 'SILK_NGSPICE_BIN')
    assert launched.deck_sha256 == prepared_sha
    assert launched.corner == 'tt' and launched.simulator_options == {'reltol': 1e-4}
    (lib,) = launched.pdk_libraries
    assert lib['section'] == 'tt' and lib['file'].endswith('sky130.lib.spice')
    assert not os.path.isabs(lib['file'])
    assert lib['file_sha256'] and lib['section_text_sha256']

    # The handed-in path must never probe a local binary's version.
    def _no_probe(*a, **k):
        raise AssertionError('external-rawfile read a local binary version')
    monkeypatch.setattr(provenance, '_binary_version', _no_probe)
    monkeypatch.setattr(provenance, 'launch_record', _no_probe)
    external = tb.evaluate('tt', rawfile=tb.results_file).provenance
    assert external.run_origin == 'external-rawfile'
    assert external.simulator_binary is None
    assert external.simulator_binary_source is None
    assert external.simulator_version is None
    assert external.deck_sha256 == prepared_sha
