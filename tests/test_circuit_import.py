"""
Tests for silk.db.circuit_import — CircuitImporter and CircuitExporter.
"""

import csv
import json
import math
import os
from pathlib import Path

import pytest

from silk.db.circuit_db import Database, DesignStage
from silk.db.circuit_import import (
    CircuitImporter, CircuitExporter, _import_identity, _import_structure,
    _parse_spice_value,
)
from silk.exceptions import DatabaseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """In-memory Database."""
    d = Database(':memory:')
    yield d
    d.close()


@pytest.fixture
def importer(db):
    return CircuitImporter(db)


@pytest.fixture
def exporter(db):
    return CircuitExporter(db)


def _manual_identity(transistors=()):
    return _import_identity({
        'transistors': list(transistors),
        'nets': [],
        'hierarchy': [],
        'parameters': [],
        'passives': [],
        'modules': {},
    })


# ---------------------------------------------------------------------------
# _parse_spice_value
# ---------------------------------------------------------------------------

class TestParseSpiceValue:
    """Thin adapter over ``silk.utils._utils._parse_si`` (TASK-26).

    The parse contract lives in ``tests/test_spice_number_contract.py``; these
    pin the importer-specific behaviour: same numbers as everywhere else,
    and ``None`` only with a warning attached.
    """

    def test_plain_int(self):
        assert _parse_spice_value('4') == 4.0

    def test_plain_float(self):
        assert _parse_spice_value('0.15') == 0.15

    def test_scientific(self):
        assert _parse_spice_value('1.5e-6') == 1.5e-6

    def test_suffix_m(self):
        assert _parse_spice_value('150m') == pytest.approx(0.15)

    def test_suffix_u(self):
        assert _parse_spice_value('1u') == pytest.approx(1e-6)

    def test_suffix_n(self):
        assert _parse_spice_value('10n') == pytest.approx(1e-8)

    def test_suffix_p(self):
        assert _parse_spice_value('100p') == pytest.approx(1e-10)

    def test_suffix_k(self):
        assert _parse_spice_value('10k') == pytest.approx(10e3)

    def test_suffix_meg(self):
        assert _parse_spice_value('2meg') == pytest.approx(2e6)

    def test_suffix_case_insensitive(self):
        assert _parse_spice_value('150M') == pytest.approx(0.15)

    def test_trailing_unit_letter_is_ignored_like_spice(self):
        """Was ``None`` before TASK-26 (``endswith`` never matched ``'us'``),
        so imported W/L carrying a unit letter came through null."""
        assert _parse_spice_value('2.56us') == pytest.approx(2.56e-6)
        assert _parse_spice_value('150nm') == pytest.approx(150e-9)

    @pytest.mark.parametrize('token', ['', 'abc', '{2*wn}', "'wn*2'"])
    def test_unparseable_is_none_with_a_warning(self, token):
        """Lenient for foreign netlists, but never silent."""
        with pytest.warns(UserWarning, match='stored as None'):
            assert _parse_spice_value(token) is None

    def test_whitespace(self):
        assert _parse_spice_value('  4  ') == 4.0


# ---------------------------------------------------------------------------
# parse_spice_netlist
# ---------------------------------------------------------------------------

class TestParseSpiceNetlist:

    def test_parse_bare_mosfet(self, importer, tmp_path):
        """Parse M... lines (classic SPICE format)."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text(
            "* test circuit\n"
            ".subckt amp vin vout vdd gnd\n"
            "M1 vout vin gnd gnd nmos_model w=2e-6 l=180e-9 m=2\n"
            "M2 vout vbias vdd vdd pmos_model w=4e-6 l=180e-9\n"
            ".ends\n"
        )
        result = importer.parse_spice_netlist(str(netlist))
        assert result['name'] == 'amp'
        assert len(result['transistors']) == 2
        t1 = result['transistors'][0]
        assert t1['name'] == 'M1'
        assert t1['width'] == pytest.approx(2e-6)
        assert t1['length'] == pytest.approx(180e-9)
        assert t1['multiplier'] == 2
        assert t1['type'] == 'nmos'

    def test_parse_xm_mosfet(self, importer, tmp_path):
        """Parse XM... lines (silk/sky130 subcircuit-wrapped format)."""
        netlist = tmp_path / 'sky130.cir'
        netlist.write_text(
            ".subckt ldo_nmos vdd gnd vin vout vnbias\n"
            "XMmn1 voutn vinp tail gnd sky130_fd_pr__nfet_01v8 m=2 w=4 l=150m\n"
            "XMmp1 voutp voutp vdd vdd sky130_fd_pr__pfet_01v8 m=1 w=4 l=150m\n"
            ".ends\n"
        )
        result = importer.parse_spice_netlist(str(netlist))
        assert result['name'] == 'ldo_nmos'
        assert len(result['transistors']) == 2
        mn = result['transistors'][0]
        assert mn['name'] == 'XMmn1'
        assert mn['width'] == pytest.approx(4.0)
        assert mn['length'] == pytest.approx(0.15)
        assert mn['multiplier'] == 2
        assert mn['type'] == 'nmos'
        mp = result['transistors'][1]
        assert mp['type'] == 'pmos'

    def test_parse_no_subckt(self, importer, tmp_path):
        """Without .subckt, name comes from filename stem."""
        netlist = tmp_path / 'my_design.cir'
        netlist.write_text(
            "M1 out in gnd gnd nmos w=1 l=0.15\n"
        )
        result = importer.parse_spice_netlist(str(netlist))
        assert result['name'] == 'my_design'

    def test_net_type_classification(self, importer, tmp_path):
        """Nets are classified by name (power, ground, signal)."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text(
            "M1 vout vin 0 vdd nmos w=1 l=0.15\n"
        )
        result = importer.parse_spice_netlist(str(netlist))
        nets = {n['name']: n['type'] for n in result['nets']}
        assert nets['vdd'] == 'power'
        assert nets['0'] == 'ground'
        assert nets['vout'] == 'signal'
        assert nets['vin'] == 'signal'

    def test_name_classified_nets_are_marked_inferred(self, importer, tmp_path):
        """TASK-97. Pre-fix, a foreign SPICE netlist's classified nets
        ('vdd' -> power, '0' -> ground) carried no provenance at all --
        indistinguishable, downstream, from a role a user declared. Since
        a bare name-pattern match is always a guess (the importer has no
        source of a *declared* net role), every classified net is now
        marked ``inferred``; the unmatched 'signal' default carries no
        such mark because nothing was inferred for it."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text(
            "M1 vout vin 0 vdd nmos w=1 l=0.15\n"
        )
        result = importer.parse_spice_netlist(str(netlist))
        nets = {n['name']: n for n in result['nets']}
        assert nets['vdd'].get('inferred') is True
        assert nets['0'].get('inferred') is True
        assert 'inferred' not in nets['vout']
        assert 'inferred' not in nets['vin']

    def test_skips_comments_and_dotcmds(self, importer, tmp_path):
        """Comments and .end lines are skipped."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text(
            "* this is a comment\n"
            ".param vdd=1.8\n"
            "M1 out in gnd gnd nmos w=1 l=0.15\n"
            ".end\n"
        )
        result = importer.parse_spice_netlist(str(netlist))
        assert len(result['transistors']) == 1


# ---------------------------------------------------------------------------
# import_from_spice
# ---------------------------------------------------------------------------

class TestImportFromSpice:

    def test_imports_transistors_and_nets(self, importer, db, tmp_path):
        """import_from_spice creates circuit with transistors and nets."""
        netlist = tmp_path / 'amp.cir'
        netlist.write_text(
            ".subckt amp vin vout vdd gnd\n"
            "XMmn1 vout vin gnd gnd sky130_fd_pr__nfet_01v8 w=2 l=150m m=2\n"
            ".ends\n"
        )
        cid = importer.import_from_spice(str(netlist))
        assert isinstance(cid, str)

        row = db.conn.execute(
            "SELECT * FROM circuit_versions WHERE version_id = ?", (cid,)
        ).fetchone()
        assert row['state'] == 'sealed'
        assert row['structure_scheme'] == 'silk.structure/v4'
        assert row['spec_scheme'] == 'silk.spec/v1'
        assert 'structure_hash' not in json.loads(row['metadata'] or '{}')

        summary = db.get_circuit_summary('amp')
        assert summary is not None
        assert len(summary['devices']) == 1
        assert len(summary['nets']) >= 1

    def test_refuses_transistors_without_dimensions(self, importer, db, tmp_path):
        """A transistor without w/l raises instead of being dropped (TASK-613)."""
        netlist = tmp_path / 'incomplete.cir'
        netlist.write_text(
            "M1 out in gnd gnd nmos\n"
        )
        with pytest.raises(DatabaseError, match=r"M1.*width"):
            importer.import_from_spice(str(netlist))
        assert db.get_circuit_summary('incomplete') is None

    def test_inferred_net_role_survives_store_readback_and_export(
        self, importer, exporter, db, tmp_path
    ):
        """TASK-112. parse_spice_netlist marks a name-classified net (e.g.
        'vdd' -> power) 'inferred': True (TASK-97), but the mark stopped
        there: store_structure's net loop passed only name/net_type to
        db.add_net, so it never reached the properties column, and
        get_circuit_summary's net SELECT never fetched that column even
        for a net whose flag *was* stored. Either gap alone drops an
        imported circuit's role provenance the moment it is stored. This
        exercises the full chain -- parse -> import (store) -> summary
        (read back) -> export -- for one inferred net ('vdd', power,
        pattern-matched) and one declared-by-default net ('vout', signal,
        nothing was inferred for it)."""
        netlist = tmp_path / 'amp.cir'
        netlist.write_text(
            "M1 vout vin gnd vdd nmos w=1 l=0.15\n"
        )
        cid = importer.import_from_spice(str(netlist))

        summary = db.get_circuit_summary('amp')
        nets = {n['name']: n for n in summary['nets']}
        assert nets['vdd']['net_type'] == 'power'
        assert nets['vdd']['role_inferred'] is True
        assert nets['vout']['net_type'] == 'signal'
        assert nets['vout']['role_inferred'] is False

        json_out = tmp_path / 'amp.json'
        exporter.export_to_json('amp', str(json_out))
        exported_nets = {
            n['name']: n for n in json.loads(json_out.read_text())['circuits'][0]['nets']
        }
        assert exported_nets['vdd']['role_inferred'] is True
        assert exported_nets['vout']['role_inferred'] is False

        # And the flag survives a further re-import into a fresh database,
        # even though the projection writes it back as 'role_inferred'
        # rather than parse_spice_netlist's original 'inferred' key.
        reimport_db = Database(':memory:')
        try:
            CircuitImporter(reimport_db).import_from_json(str(json_out))
            reimported = reimport_db.get_circuit_summary('amp')
            reimported_nets = {n['name']: n for n in reimported['nets']}
            assert reimported_nets['vdd']['role_inferred'] is True
            assert reimported_nets['vout']['role_inferred'] is False
        finally:
            reimport_db.close()


# ---------------------------------------------------------------------------
# import_from_json
# ---------------------------------------------------------------------------

class TestImportFromJson:

    def test_imports_circuit(self, importer, db, tmp_path):
        """import_from_json creates circuit with all components."""
        json_file = tmp_path / 'circuits.json'
        json_file.write_text(json.dumps({
            'circuits': [{
                'name': 'test_amp',
                'version': '2.0',
                'stage': 'dev',
                'description': 'test',
                'transistors': [
                    {'name': 'M1', 'type': 'nmos', 'width': 1e-6, 'length': 0.18e-6},
                ],
                'nets': [
                    {'name': 'vdd', 'type': 'power'},
                    {'name': 'gnd', 'type': 'ground'},
                ],
                'simulation_results': [
                    {'parameter': 'gain', 'value': 45.2, 'unit': 'dB'},
                ],
            }]
        }))
        ids = importer.import_from_json(str(json_file))
        assert len(ids) == 1
        summary = db.get_circuit_summary('test_amp')
        assert summary is not None
        assert len(summary['devices']) == 1
        assert len(summary['nets']) == 2
        assert len(summary['simulation_results']) == 1

    def test_imports_multiple_circuits(self, importer, db, tmp_path):
        """import_from_json handles multiple circuits in one file."""
        json_file = tmp_path / 'multi.json'
        json_file.write_text(json.dumps({
            'circuits': [
                {'name': 'amp1', 'transistors': [], 'nets': []},
                {'name': 'amp2', 'transistors': [], 'nets': []},
            ]
        }))
        ids = importer.import_from_json(str(json_file))
        assert len(ids) == 2

    def test_content_edit_creates_child_snapshot(self, importer, db, tmp_path):
        json_file = tmp_path / 'edit.json'

        def write(width):
            json_file.write_text(json.dumps({
                'circuits': [{
                    'name': 'edited_amp',
                    'version': 'draft',
                    'transistors': [{
                        'name': 'M1', 'type': 'nmos',
                        'width': width, 'length': 0.18e-6,
                    }],
                    'nets': [{'name': 'out', 'type': 'signal'}],
                }]
            }))

        write(1e-6)
        first = importer.import_from_json(str(json_file))[0]
        write(2e-6)
        second = importer.import_from_json(str(json_file))[0]

        assert second != first
        row = db.conn.execute(
            "SELECT parent_version_id, state FROM circuit_versions "
            "WHERE version_id = ?", (second,)
        ).fetchone()
        assert tuple(row) == (first, 'sealed')


# ---------------------------------------------------------------------------
# import_simulation_results_csv
# ---------------------------------------------------------------------------

class TestImportCSV:

    def test_imports_csv_results(self, importer, db, tmp_path):
        """import_simulation_results_csv stores all rows."""
        cid = db.create_circuit(
            name='test', version='1.0', stage=DesignStage.DEV,
            identity=_manual_identity(),
        )
        db.seal_circuit_version(cid)
        csv_file = tmp_path / 'results.csv'
        csv_file.write_text("parameter,value,unit\ngain,45.2,dB\npower,2.3,mW\n")
        count = importer.import_simulation_results_csv(str(csv_file), cid)
        assert count == 2
        results = db.get_simulation_results(cid)
        assert len(results) == 2


    @pytest.mark.parametrize('header,row,expected', [
        ('parameter,value,unit', 'gain,42,dB', (42, None)),
        ('parameter,value,value_text,unit', 'gain,42,,dB', (42, None)),
        ('parameter,value,value_text', 'gain,42', (42, None)),
        ('parameter,value,value_text', 'gain,,', (None, '')),
        ('parameter,value,value_text', 'gain,,  ', (None, '  ')),
        ('parameter,value,value_text', 'gain,,42', (None, '42')),
        ('parameter,value,value_text', 'gain,42,text', None),
        ('parameter,value', 'gain,', None),
        ('parameter,value,value_text', 'gain,', None),
        ('parameter,value,value_text', 'gain,bad,', None),
    ])
    def test_cell_presence(self, importer, db, tmp_path, header, row, expected):
        target = db.create_circuit('target', '1', identity=_manual_identity())
        db.seal_circuit_version(target)
        path = tmp_path / 'cells.csv'
        path.write_text(header + '\n' + row + '\n')
        if expected is None:
            with pytest.raises(ValueError):
                importer.import_simulation_results_csv(str(path), target)
            assert db.get_simulation_results(target) == []
        else:
            assert importer.import_simulation_results_csv(str(path), target) == 1
            result, = db.get_simulation_results(target)
            assert (result.value, result.value_text) == expected

    @pytest.mark.parametrize('transaction', [False, True])
    def test_caller_transaction(self, importer, db, tmp_path, transaction):
        from contextlib import nullcontext

        target = db.create_circuit('target', '1', identity=_manual_identity())
        db.seal_circuit_version(target)
        path = tmp_path / 'partial.csv'
        path.write_text('parameter,value,value_text\ngood,42,\nbad,7,both\n')
        with pytest.raises(ValueError, match='exactly one'):
            with db.transaction() if transaction else nullcontext():
                importer.import_simulation_results_csv(str(path), target)
        results = db.get_simulation_results(target)
        assert len(results) == (0 if transaction else 1)
        if results:
            assert results[0].parameter == 'good'
            assert results[0].value == 42

    @pytest.mark.parametrize('state', ['missing', 'building', 'locked'])
    def test_target_guards(self, importer, db, tmp_path, state):
        from silk.exceptions import LockedVersionError

        target = 'missing'
        if state != 'missing':
            target = db.create_circuit('target', '1', identity=_manual_identity())
            if state == 'locked':
                db.seal_circuit_version(target)
                db.lock_version(target)
        path = tmp_path / 'guard.csv'
        path.write_text('parameter,value,value_text\ngain,42,\n')
        error = LockedVersionError if state == 'locked' else DatabaseError
        with pytest.raises(error):
            importer.import_simulation_results_csv(str(path), target)

    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('metadata', [{}, {'sim_type': 'noise', 'tool_name': 'reader'}])
    def test_public_value_round_trip(self, importer, exporter, db, tmp_path,
                                     reverse, metadata):
        source = db.create_circuit('source', '1', identity=_manual_identity())
        target = db.create_circuit('target', '1', identity=_manual_identity())
        db.seal_circuit_version(source)
        db.seal_circuit_version(target)
        values = [42, 0, -7, 0.125, float('nan'),
                  'converged', '42', '', '  ', 'comma, "quote"\nnext line',
                  'first\r\nsecond', 'first\rsecond', ' \r\t ']
        if reverse:
            values.reverse()
        for index, value in enumerate(values):
            payload = {'value_text': value} if isinstance(value, str) else {'value': value}
            db.add_simulation_result(
                source, parameter='gain_dB' if value == 42 and not isinstance(value, str) else f'metric_{index}', unit='dB',
                sim_type='ac', tool_name='writer', **payload,
            )
        path = tmp_path / 'roundtrip.csv'
        exporter.export_results_to_csv('source', str(path))
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == [
                'parameter', 'value', 'value_text', 'unit',
                'sim_type', 'tool_name', 'timestamp',
            ]
            rows = list(reader)
        # Make timestamp restoration observably different from insertion time.
        for row in rows:
            row['timestamp'] = '2000-01-01 00:00:00'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        assert importer.import_simulation_results_csv(str(path), target, **metadata) == len(values)
        results = {r.parameter: r for r in db.get_simulation_results(target)}
        assert len(results) == len(values)
        for index, value in enumerate(values):
            parameter = 'gain_dB' if value == 42 and not isinstance(value, str) else f'metric_{index}'
            result = results[parameter]
            assert result.unit == 'dB'
            assert result.sim_type == metadata.get('sim_type', 'dc')
            assert result.tool_name == metadata.get('tool_name', 'unknown')
            assert str(result.timestamp) != '2000-01-01 00:00:00'
            if isinstance(value, str):
                assert result.value is None
                assert result.value_text == value
            else:
                assert result.value_text is None
                if math.isnan(value):
                    assert math.isnan(result.value)
                else:
                    assert result.value == value


# ---------------------------------------------------------------------------
# CircuitExporter
# ---------------------------------------------------------------------------

class TestExportToJson:

    def test_round_trip(self, importer, exporter, db, tmp_path):
        """Export then re-import produces equivalent data."""
        # Create circuit with data
        json_in = tmp_path / 'in.json'
        json_in.write_text(json.dumps({
            'circuits': [{
                'name': 'roundtrip',
                'version': '1.0',
                'stage': 'dev',
                'transistors': [
                    {'name': 'M1', 'type': 'nmos', 'width': 1e-6, 'length': 0.18e-6},
                ],
                'nets': [
                    {'name': 'vdd', 'type': 'power'},
                ],
                'simulation_results': [
                    {'parameter': 'gain', 'value': 45.2, 'unit': 'dB'},
                ],
            }]
        }))
        importer.import_from_json(str(json_in))

        # Export
        json_out = tmp_path / 'out.json'
        exporter.export_to_json('roundtrip', str(json_out))

        with open(json_out) as f:
            data = json.load(f)
        c = data['circuits'][0]
        assert c['name'] == 'roundtrip'
        assert len(c['transistors']) == 1
        assert c['transistors'][0]['name'] == 'M1'
        assert len(c['nets']) == 1
        assert len(c['simulation_results']) == 1
        assert c['simulation_results'][0]['value'] == 45.2

    def test_export_nonexistent_circuit(self, exporter, tmp_path):
        """export_to_json does nothing for nonexistent circuit."""
        out = tmp_path / 'empty.json'
        exporter.export_to_json('nonexistent', str(out))
        assert not out.exists()

    def test_projection_marks_scope_and_exports_complete_result_context(
        self, exporter, db, tmp_path
    ):
        circuit_id = db.create_circuit(
            name='projection_context',
            version='1.0',
            stage=DesignStage.DEV,
            identity=_manual_identity(),
        )
        db.seal_circuit_version(circuit_id)
        db.add_simulation_result(
            circuit_version_id=circuit_id,
            parameter='gain_dB',
            value=45.2,
            unit='dB',
            object_id='stage_1',
            object_type='net',
            source='spice',
            sim_type='ac',
            corner='ss',
            temperature=85.0,
            voltage=1.62,
            tool_name='ngspice-44',
            conditions={'load_pf': 2.0, 'mode': 'closed_loop'},
            timestamp='2026-08-29 12:34:56',
        )

        output = tmp_path / 'projection.json'
        exporter.export_to_json('projection_context', str(output))
        payload = json.loads(output.read_text())

        assert payload['projection_format'] == 'silk.circuit-projection/v2'
        assert payload['projection_scope']['is_database_backup'] is False
        assert payload['projection_scope']['is_circuit_archive'] is False
        assert set(payload['projection_scope']['omits']) == {
            'specs',
            'net_currents',
            'parasitics',
            'complete_module_connectivity',
            'raw_runs',
            'waveforms',
            'simulator_logs',
            'calibration',
        }
        result = payload['circuits'][0]['simulation_results'][0]
        assert set(result) == {
            'parameter', 'value', 'value_text', 'unit', 'object_id',
            'object_type', 'source', 'sim_type', 'corner', 'temperature',
            'voltage', 'tool_name', 'conditions', 'timestamp',
            'corner_state', 'temperature_state', 'voltage_state',
        }
        assert result['conditions'] == {
            'load_pf': 2.0,
            'mode': 'closed_loop',
        }
        assert 'result_id' not in result
        assert 'origin_version_id' not in result
        circuit = payload['circuits'][0]
        assert 'specs' not in circuit
        assert 'net_currents' not in circuit
        assert 'parasitics' not in circuit

    def test_projection_import_recomputes_identity_and_evidence_ids(
        self, exporter, db, tmp_path
    ):
        original_id = db.create_circuit(
            name='projection_identity',
            version='1.0',
            stage=DesignStage.DEV,
            identity=_manual_identity(),
        )
        db.seal_circuit_version(original_id)
        original_result_id = db.add_simulation_result(
            circuit_version_id=original_id,
            parameter='status',
            value_text='converged',
            source='user',
            corner='tt',
            conditions={'attempt': 1},
        )
        projection = tmp_path / 'identity-projection.json'
        exporter.export_to_json('projection_identity', str(projection))

        payload = json.loads(projection.read_text())
        circuit_payload = payload['circuits'][0]
        expected_identity = _import_identity(_import_structure(circuit_payload))
        # Even forged database identifiers in an input projection are data the
        # importer must ignore; applicability is recomputed from parsed content.
        circuit_payload['version_id'] = original_id
        circuit_payload['structure_hash'] = 'forged-structure'
        circuit_payload['spec_set_hash'] = 'forged-specs'
        circuit_payload['simulation_results'][0]['result_id'] = original_result_id
        circuit_payload['simulation_results'][0]['origin_version_id'] = original_id
        projection.write_text(json.dumps(payload))

        destination = Database(str(tmp_path / 'projection-import.db'))
        try:
            imported_id = CircuitImporter(destination).import_from_json(
                str(projection)
            )[0]
            imported = destination._snapshot_row(imported_id)
            imported_results = destination.get_simulation_results(imported_id)
        finally:
            destination.close()

        assert imported_id != original_id
        assert imported['structure_scheme'] == expected_identity.structure.scheme
        assert imported['structure_hash'] == expected_identity.structure.digest
        assert imported['spec_scheme'] == expected_identity.spec_set.scheme
        assert imported['spec_set_hash'] == expected_identity.spec_set.digest
        assert len(imported_results) == 1
        assert imported_results[0].result_id != original_result_id
        assert imported_results[0].origin_version_id == imported_id
        assert imported_results[0].conditions == {'attempt': 1}

    def test_projection_import_refuses_existing_snapshot_identity(
        self, exporter, db, tmp_path
    ):
        original_id = db.create_circuit(
            name='projection_collision',
            version='1.0',
            stage=DesignStage.DEV,
            identity=_manual_identity(),
        )
        db.seal_circuit_version(original_id)
        original_result_id = db.add_simulation_result(
            circuit_version_id=original_id,
            parameter='gain_dB',
            value=45.2,
            unit='dB',
            source='spice',
        )
        projection = tmp_path / 'collision-projection.json'
        exporter.export_to_json('projection_collision', str(projection))

        # An import projection is untrusted interchange data. It must never
        # resolve to an existing sealed UUID and append evidence to it.
        with pytest.raises(DatabaseError, match='existing sealed snapshot'):
            CircuitImporter(db).import_from_json(str(projection))

        results = db.get_simulation_results(original_id)
        assert [record.result_id for record in results] == [original_result_id]
        assert results[0].origin_version_id == original_id


class TestExportToCSV:

    def test_exports_csv(self, exporter, db, tmp_path):
        """export_results_to_csv writes correct CSV."""
        cid = db.create_circuit(
            name='csv_test', version='1.0', stage=DesignStage.DEV,
            identity=_manual_identity(),
        )
        db.seal_circuit_version(cid)
        db.add_simulation_result(circuit_version_id=cid, parameter='gain',
                                 value=45.0, unit='dB')
        out = tmp_path / 'results.csv'
        exporter.export_results_to_csv('csv_test', str(out))
        with open(out) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 row
        assert 'gain' in lines[1]

    def test_export_csv_nonexistent(self, exporter, tmp_path):
        """export_results_to_csv does nothing for nonexistent circuit."""
        out = tmp_path / 'empty.csv'
        exporter.export_results_to_csv('nonexistent', str(out))
        assert not out.exists()


# ---------------------------------------------------------------------------
# Extended parser: passives and sources
# ---------------------------------------------------------------------------

class TestParsePassives:

    def test_parse_resistor(self, importer, tmp_path):
        """Parser extracts R elements."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text("Rload vout vdd 10k\n")
        result = importer.parse_spice_netlist(str(netlist))
        assert len(result['passives']) == 1
        r = result['passives'][0]
        assert r['kind'] == 'resistor'
        assert r['name'] == 'Rload'
        assert r['pos'] == 'vout'
        assert r['neg'] == 'vdd'
        assert r['value'] == pytest.approx(10e3)

    def test_parse_capacitor(self, importer, tmp_path):
        """Parser extracts C elements."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text("Cload vout 0 100p\n")
        result = importer.parse_spice_netlist(str(netlist))
        assert len(result['passives']) == 1
        c = result['passives'][0]
        assert c['kind'] == 'capacitor'
        assert c['value'] == pytest.approx(1e-10)

    def test_parse_inductor(self, importer, tmp_path):
        """Parser extracts L elements."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text("L1 vdd node1 1u\n")
        result = importer.parse_spice_netlist(str(netlist))
        assert len(result['passives']) == 1
        assert result['passives'][0]['kind'] == 'inductor'
        assert result['passives'][0]['value'] == pytest.approx(1e-6)


class TestParseSources:

    def test_parse_voltage_source(self, importer, tmp_path):
        """Parser extracts V elements with DC value."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text("Vsupply vdd 0 DC 1.8V\n")
        result = importer.parse_spice_netlist(str(netlist))
        assert len(result['sources']) == 1
        v = result['sources'][0]
        assert v['kind'] == 'voltage'
        assert v['name'] == 'Vsupply'
        assert v['positive'] == 'vdd'
        assert v['negative'] == '0'
        assert v['dc'] == pytest.approx(1.8)

    def test_parse_voltage_source_dc_ac(self, importer, tmp_path):
        """Parser extracts both DC and AC values."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text("Vsupply vdd 0 DC 3.3V AC 1V\n")
        result = importer.parse_spice_netlist(str(netlist))
        v = result['sources'][0]
        assert v['dc'] == pytest.approx(3.3)
        assert v['ac'] == pytest.approx(1.0)

    def test_parse_current_source(self, importer, tmp_path):
        """Parser extracts I elements."""
        netlist = tmp_path / 'test.cir'
        netlist.write_text("Ibias vdd vnbias DC 0.0001A\n")
        result = importer.parse_spice_netlist(str(netlist))
        assert len(result['sources']) == 1
        i = result['sources'][0]
        assert i['kind'] == 'current'
        assert i['dc'] == pytest.approx(1e-4)

    def test_mixed_elements(self, importer, tmp_path):
        """Parser handles mixed transistors, passives, sources."""
        netlist = tmp_path / 'mixed.cir'
        netlist.write_text(
            ".subckt amp vin vout vdd gnd\n"
            "XMmn1 vout vin gnd gnd sky130_fd_pr__nfet_01v8 w=2 l=150m m=1\n"
            "Rload vdd vout 10k\n"
            "Cload vout gnd 1p\n"
            "Vsupply vdd gnd DC 1.8V\n"
            ".ends\n"
        )
        result = importer.parse_spice_netlist(str(netlist))
        assert len(result['transistors']) == 1
        assert len(result['passives']) == 2
        assert len(result['sources']) == 1


# ---------------------------------------------------------------------------
# parse_spice_netlist with PDK-declared device prefixes
# ---------------------------------------------------------------------------

class TestParseWithDeclaredPrefixes:
    """``device_prefixes=`` makes device-line detection follow what a PDK
    declares instead of the literal ``M``/``XM`` rule (TASK-27 AC #2). The
    default (``None``) is the legacy rule, byte-for-byte, because the
    database import path still relies on it."""

    def test_default_keeps_legacy_detection(self, importer, tmp_path):
        netlist = tmp_path / 'legacy.cir'
        netlist.write_text(
            "M1 out in gnd gnd nmos w=1 l=0.15\n"
            "XMmn1 vout vin gnd gnd sky130_fd_pr__nfet_01v8 w=2 l=150m m=2\n"
            "D1 a k diode_nd2ps_03v3 area=2e-12\n"
        )
        result = importer.parse_spice_netlist(str(netlist))
        names = [t['name'] for t in result['transistors']]
        assert names == ['M1', 'XMmn1'], 'D is not a legacy device prefix'
        legacy = result['transistors'][1]
        assert legacy['prefix'] is None
        assert legacy['nodes'] == ['vout', 'vin', 'gnd', 'gnd']
        assert legacy['model'] == 'sky130_fd_pr__nfet_01v8'
        assert legacy['params'] == {'w': 2.0, 'l': pytest.approx(0.15), 'm': 2.0}

    def test_only_declared_prefixes_are_device_lines(self, importer, tmp_path):
        netlist = tmp_path / 'xm_only.cir'
        netlist.write_text(
            "M1 out in gnd gnd nmos w=1 l=0.15\n"
            "XMmn1 vout vin gnd gnd sky130_fd_pr__nfet_01v8 w=2 l=150m\n"
        )
        result = importer.parse_spice_netlist(str(netlist), device_prefixes=('XM',))
        assert [t['name'] for t in result['transistors']] == ['XMmn1']
        assert result['transistors'][0]['prefix'] == 'XM'

    def test_prefix_match_is_case_insensitive_and_recorded_as_declared(
            self, importer, tmp_path):
        netlist = tmp_path / 'lower.cir'
        netlist.write_text("xmn1 d g s b nfet_03v3 w=1u l=0.5u\n")
        result = importer.parse_spice_netlist(str(netlist), device_prefixes=['XM'])
        t = result['transistors'][0]
        assert t['name'] == 'xmn1'
        assert t['prefix'] == 'XM', 'the declared spelling, so len() strips it'

    def test_two_terminal_device_parses_by_position(self, importer, tmp_path):
        """Model is the last non-assignment token; nodes are what precede it.
        Spaces around '=' do not split an assignment into positional tokens."""
        netlist = tmp_path / 'diode.cir'
        netlist.write_text("D1 a k diode_nd2ps_03v3 area = 2e-12 pj=4e-6 m=2\n")
        result = importer.parse_spice_netlist(str(netlist), device_prefixes=('D', 'XM'))
        t = result['transistors'][0]
        assert t['prefix'] == 'D'
        assert t['nodes'] == ['a', 'k']
        assert t['model'] == 'diode_nd2ps_03v3'
        assert t['params'] == {'area': pytest.approx(2e-12),
                               'pj': pytest.approx(4e-6), 'm': 2.0}
        assert t['multiplier'] == 2
        assert {'a', 'k'} <= {n['name'] for n in result['nets']}

    def test_longest_declared_prefix_wins(self, importer, tmp_path):
        netlist = tmp_path / 'xx.cir'
        netlist.write_text("XMn1 d g s b nfet w=1 l=1\nX1 a b sub\n")
        result = importer.parse_spice_netlist(str(netlist), device_prefixes=('X', 'XM'))
        by_name = {t['name']: t['prefix'] for t in result['transistors']}
        assert by_name == {'XMn1': 'XM', 'X1': 'X'}

    def test_unparseable_param_value_is_kept_verbatim(self, importer, tmp_path):
        netlist = tmp_path / 'ref.cir'
        netlist.write_text("XMn1 d g s b nfet w=wn l=0.5\n")
        result = importer.parse_spice_netlist(str(netlist), device_prefixes=('XM',))
        assert result['transistors'][0]['params'] == {'w': 'wn', 'l': 0.5}

    def test_line_without_a_model_token_is_skipped(self, importer, tmp_path):
        netlist = tmp_path / 'short.cir'
        netlist.write_text("D1 a\nD2 a k dmodel\n")
        result = importer.parse_spice_netlist(str(netlist), device_prefixes=('D',))
        assert [t['name'] for t in result['transistors']] == ['D2']


class TestExportPdkNativeKeys:
    """2026-08-24 results-file review, B4: the auto-tracker writes PDK-native
    'w'/'l'/'m' keys into devices.properties while the exporter read only
    'width'/'length'/'multiplier' with a default of 1 — a real m=4 device
    exported as multiplier=1 (actively wrong, not merely null)."""

    def test_auto_tracker_keys_export_correctly(self, exporter, db, tmp_path):
        device = {
            'name': 'mbig', 'type': 'nmos1p8', 'model': 'nmos1p8',
            'parameters': {'w': 4.0, 'l': 0.5, 'm': 4},
        }
        cid = db.create_circuit(
            'autockt', '1.0', identity=_manual_identity([device])
        )
        db.add_device(cid, 'mbig', 'nmos1p8', model='nmos1p8',
                      w=4.0, l=0.5, m=4)
        db.seal_circuit_version(cid)
        out = tmp_path / 'e.json'
        exporter.export_to_json('autockt', str(out))
        t = json.loads(out.read_text())['circuits'][0]['transistors'][0]
        assert t['width'] == 4.0
        assert t['length'] == 0.5
        assert t['multiplier'] == 4

    def test_unknown_multiplier_is_none_not_one(self, exporter, db, tmp_path):
        device = {
            'name': 'mx', 'type': 'nmos1p8', 'model': 'nmos1p8',
            'parameters': {},
        }
        cid = db.create_circuit(
            'bare', '1.0', identity=_manual_identity([device])
        )
        db.add_device(cid, 'mx', 'nmos1p8')
        db.seal_circuit_version(cid)
        out = tmp_path / 'b.json'
        exporter.export_to_json('bare', str(out))
        t = json.loads(out.read_text())['circuits'][0]['transistors'][0]
        assert t['multiplier'] is None
        assert t['width'] is None


# ---------------------------------------------------------------------------
# TASK-603: the circuit author survives the JSON projection
# ---------------------------------------------------------------------------

class TestCreatedByRoundTrip:
    def test_author_survives_create_seal_export_import(
        self, exporter, db, tmp_path
    ):
        circuit_id = db.create_circuit(
            name='authored', version='1.0', stage=DesignStage.DEV,
            created_by='jdoe', identity=_manual_identity(),
        )
        db.seal_circuit_version(circuit_id)

        assert db.get_latest('authored')['created_by'] == 'jdoe'
        assert db.get_version('authored', '1.0')['created_by'] == 'jdoe'

        json_out = tmp_path / 'authored.json'
        exporter.export_to_json('authored', str(json_out))
        exported = json.loads(json_out.read_text())['circuits'][0]
        assert exported['created_by'] == 'jdoe'

        other = Database(':memory:')
        try:
            CircuitImporter(other).import_from_json(str(json_out))
            assert other.get_latest('authored')['created_by'] == 'jdoe'
        finally:
            other.close()
