"""TASK-9 (DB-02-PROOF) AC2: a real v1 database file migrates and reads back
correctly through silk's PUBLIC readback surface.

TASK-4 (``tests/db/test_migration_v1_to_v2.py``) already proves the
migration's internal shape/byte/atomicity contract in detail. This file is
narrower and deliberately separate: it proves the *public, docs-facing*
promise from ``docs/database.md`` — build a real v1 file through the genuine
v1 write path, migrate it through the documented v1-through-v7 copy chain,
and read the result back with nothing but
``silk.db.Database`` — the same calls a user follows.

The v1 fixture is built through ``tests/db/_v1_fixture`` (a frozen snapshot
of the real v1 write path) rather than hand-authored, because
normal ``Database`` construction now creates schema v6 — this fixture
builder is the only real v1 code path left in the
repository. It is test-only (not shipped in the wheel), which is why this
proof lives here rather than inside ``silk/examples/``.
"""

import hashlib

from silk.db import (
    Database,
    migrate_v1_to_v2,
    migrate_v2_to_v3,
    migrate_v3_to_v4,
    migrate_v4_to_v5,
    migrate_v5_to_v6,
    migrate_v6_to_v7,
)

from tests.db._v1_fixture.build import build_v1_database


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_file_migrates_and_reads_back_through_the_public_surface(tmp_path):
    v1_path = tmp_path / 'legacy.db'
    dest_path = tmp_path / 'migrated.db'

    fixture = build_v1_database(str(v1_path))
    source_digest_before = _sha256(v1_path)

    result_path = migrate_v1_to_v2(str(v1_path), str(dest_path))
    assert result_path == str(dest_path)
    assert dest_path.exists()

    v2_digest = _sha256(dest_path)
    v3_path = tmp_path / "v3.db"
    assert migrate_v2_to_v3(dest_path, v3_path) == str(v3_path)
    assert _sha256(dest_path) == v2_digest
    v4_path = tmp_path / "v4.db"
    assert migrate_v3_to_v4(v3_path, v4_path) == str(v4_path)
    v5_path = tmp_path / "v5.db"
    assert migrate_v4_to_v5(v4_path, v5_path) == str(v5_path)
    current_path = tmp_path / "current.db"
    v6_path = tmp_path / "v6.db"
    assert migrate_v5_to_v6(v5_path, v6_path) == str(v6_path)
    assert migrate_v6_to_v7(v6_path, current_path) == str(current_path)

    with Database(str(current_path)) as db:
        # Parent version: metrics, spec measurements, and a current set all
        # read back, all honestly unscoped (profile_id is None; v1 has no
        # authoritative test identity to invent one from — design SS7 step 4).
        metrics = db.get_simulation_results(
            fixture['version_a'], corner='tt', include_ancestors=False
        )
        assert metrics
        assert {record.parameter for record in metrics} == {'gain'}
        assert metrics[0].value == 41.5
        assert all(record.profile_id is None for record in metrics)

        declarations = db.get_stored_specs(
            fixture['version_a'], include_ancestors=False
        )
        assert {d.spec_key for d in declarations} == {
            'system_gain', 'core_gain',
        }
        core_gain = next(
            d for d in declarations if d.spec_key == 'core_gain'
        )
        assert core_gain.measurements
        assert all(
            measurement.profile_id is None
            for measurement in core_gain.measurements
        )

        origin_version_id, current_rows = db.get_net_current_set(
            fixture['version_a'], 'tt'
        )
        assert origin_version_id == fixture['version_a']
        assert current_rows
        assert {row['net_name'] for row in current_rows} == {'out_a'}
        # No characterization_profiles rows exist anywhere in a migrated v1
        # database (design SS7 step 4) -- every current set/metric/spec
        # measurement above is explicitly unscoped, not guessed.
        assert db.list_characterization_profiles(
            fixture['version_a'], include_ancestors=False
        ) == []

        # Child version (lineage): its own directly-stored current set at
        # 'tt' reads back too, still unscoped.
        child_origin, child_rows = db.get_net_current_set(
            fixture['version_b'], 'tt', include_ancestors=False
        )
        assert child_origin == fixture['version_b']
        assert child_rows
        assert {row['net_name'] for row in child_rows} == {'out_b'}

    # The original v1 file is byte-identical after the migration, not merely
    # still present and non-empty.
    assert v1_path.exists()
    assert _sha256(v1_path) == source_digest_before
