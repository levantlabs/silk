BEGIN TRANSACTION;
CREATE TABLE characterization_profiles (
                profile_id         TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                owner_path         TEXT NOT NULL,
                test_name          TEXT NOT NULL,
                parameters_json    TEXT NOT NULL,
                profile_scheme     TEXT NOT NULL,
                profile_hash       TEXT NOT NULL,
                created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(circuit_version_id, profile_scheme, profile_hash),
                UNIQUE(circuit_version_id, profile_id),
                FOREIGN KEY (circuit_version_id)
                    REFERENCES circuit_versions(version_id) ON DELETE RESTRICT
            );
CREATE TABLE circuit_modules (
                circuit_version_id TEXT NOT NULL,
                module_id TEXT NOT NULL,
                instance_name TEXT,  -- Instance name in the circuit (e.g., 'ldo_output_1')
                position_in_hierarchy INTEGER,  -- Order/depth in hierarchy
                PRIMARY KEY (circuit_version_id, module_id, instance_name),
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id) ON DELETE CASCADE,
                FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
            );
CREATE TABLE circuit_versions (
                version_id TEXT PRIMARY KEY,
                circuit_name TEXT NOT NULL,
                version TEXT NOT NULL,
                stage TEXT NOT NULL CHECK(stage IN ('dev', 'stable', 'post_layout', 'tapeout')),
                state TEXT NOT NULL CHECK(state IN ('building', 'sealed')),
                is_locked BOOLEAN DEFAULT 0,
                parent_version_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                description TEXT,
                netlist_path TEXT,
                metadata TEXT,  -- JSON string for flexible data
                simulation_summary TEXT,  -- Auto-generated summary of all simulations
                last_summary_update TIMESTAMP,
                structure_scheme TEXT NOT NULL,
                structure_hash TEXT NOT NULL,
                spec_scheme TEXT NOT NULL,
                spec_set_hash TEXT NOT NULL,
                -- (circuit_name, version) is deliberately NOT unique at the
                -- schema level: CircuitTracker.resolve_version keys rows by
                -- structure_hash and reuses one version LABEL across them.
                -- The partial sealed-identity index below prevents duplicate
                -- snapshots without treating human labels as identifiers.
                FOREIGN KEY (parent_version_id) REFERENCES circuit_versions(version_id)
            );
CREATE TABLE devices (
                device_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                model TEXT,
                properties TEXT,  -- JSON for additional properties
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id)
            );
CREATE TABLE modules (
                module_id TEXT PRIMARY KEY,
                module_name TEXT NOT NULL,
                module_path TEXT NOT NULL,
                module_hash_scheme TEXT NOT NULL,
                module_hash TEXT NOT NULL,
                device_count INTEGER DEFAULT 0,
                structure TEXT,  -- JSON of module structure (devices, nets, submodules)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(module_hash_scheme, module_hash)
            );
CREATE TABLE net_current_sets (
                current_set_id             TEXT PRIMARY KEY,
                circuit_version_id         TEXT NOT NULL,
                profile_id                 TEXT,
                state                      TEXT NOT NULL
                                           CHECK(state IN ('current', 'superseded')),
                supersedes_current_set_id  TEXT UNIQUE,
                corner                     TEXT,
                temperature                REAL,
                voltage                    REAL,
                tool_name                  TEXT,
                analysis                   TEXT,
                tstart                     REAL,
                results_file               TEXT,
                structure_scheme           TEXT NOT NULL,
                structure_hash             TEXT NOT NULL,
                selected_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                superseded_at              TIMESTAMP,
                corner_state TEXT NOT NULL CHECK(corner_state IN ('fixed','none','unknown')),
                temperature_state TEXT NOT NULL CHECK(temperature_state IN ('fixed','none','unknown','sweep')),
                voltage_state TEXT NOT NULL CHECK(voltage_state IN ('fixed','none','unknown')), provenance_id TEXT REFERENCES run_provenance(provenance_id) ON DELETE RESTRICT,
                CHECK((corner_state = 'fixed') = (corner IS NOT NULL)),
                CHECK((temperature_state = 'fixed') = (temperature IS NOT NULL)),
                CHECK((voltage_state = 'fixed') = (voltage IS NOT NULL)),
                UNIQUE(circuit_version_id, current_set_id),
                CHECK(profile_id IS NULL OR
                      (corner_state IN ('fixed','none') AND
                       temperature_state IN ('fixed','none') AND
                       voltage_state IN ('fixed','none'))),
                CHECK((state = 'current' AND superseded_at IS NULL) OR
                      (state = 'superseded' AND superseded_at IS NOT NULL)),
                FOREIGN KEY (circuit_version_id)
                    REFERENCES circuit_versions(version_id) ON DELETE RESTRICT,
                FOREIGN KEY (circuit_version_id, profile_id)
                    REFERENCES characterization_profiles(circuit_version_id, profile_id)
                    ON DELETE RESTRICT,
                FOREIGN KEY (supersedes_current_set_id)
                    REFERENCES net_current_sets(current_set_id) ON DELETE RESTRICT
            );
CREATE TABLE net_currents (
                current_id     TEXT PRIMARY KEY,
                current_set_id TEXT NOT NULL,
                net_name       TEXT NOT NULL,
                pin_id         TEXT NOT NULL DEFAULT '',
                avg_a          REAL NOT NULL,
                rms_a          REAL,
                peak_a         REAL,
                source         TEXT NOT NULL CHECK(source IN ('spice', 'user')),
                current_vector TEXT,
                timestamp      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(current_set_id, net_name, pin_id),
                FOREIGN KEY (current_set_id)
                    REFERENCES net_current_sets(current_set_id) ON DELETE RESTRICT
            );
CREATE TABLE nets (
                net_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                name TEXT NOT NULL,
                net_type TEXT,  -- signal, power, ground, etc.
                properties TEXT,  -- JSON for additional properties
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id)
            );
CREATE TABLE parasitics (
                parasitic_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                net_id TEXT NOT NULL,
                parasitic_type TEXT CHECK(parasitic_type IN ('resistance', 'capacitance', 'inductance')),
                value REAL,
                unit TEXT,
                from_node TEXT,
                to_node TEXT,
                extraction_tool TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id),
                FOREIGN KEY (net_id) REFERENCES nets(net_id)
            );
CREATE TABLE release_contents (
                release_id TEXT NOT NULL,
                circuit_version_id TEXT NOT NULL,
                PRIMARY KEY (release_id, circuit_version_id),
                FOREIGN KEY (release_id) REFERENCES releases(release_id),
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id)
            );
CREATE TABLE releases (
                release_id TEXT PRIMARY KEY,
                release_name TEXT UNIQUE NOT NULL,
                stage TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                notes TEXT,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived'))
            );
CREATE TABLE run_provenance (
                provenance_id TEXT PRIMARY KEY,
                scheme        TEXT NOT NULL,
                digest        TEXT NOT NULL,
                record_json   TEXT NOT NULL,
                created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(scheme, digest)
            );
CREATE TABLE silk_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                format_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                creator_version TEXT
            );
INSERT INTO "silk_metadata" VALUES(1,'silk.db',5,'2026-10-09 00:24:53','0.2.0');
CREATE TABLE simulation_results (
                sim_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                object_id TEXT,  -- Can reference transistor, net, or circuit
                object_type TEXT CHECK(object_type IN ('circuit', 'transistor', 'net')),
                source TEXT NOT NULL,
                sim_type TEXT,  -- dc, ac, tran, noise, etc.
                corner TEXT,  -- Process corner: tt, ss, ff, sf, fs, etc.
                temperature REAL,  -- Temperature in Celsius
                voltage REAL,  -- Supply voltage
                parameter TEXT NOT NULL,
                value REAL,
                value_text TEXT,  -- For non-numeric results
                unit TEXT,
                tool_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                conditions TEXT,  -- JSON for additional simulation conditions
                profile_id TEXT,  -- NULL = explicitly unscoped/legacy/manual
                corner_state TEXT NOT NULL CHECK(corner_state IN ('fixed','none','unknown')),
                temperature_state TEXT NOT NULL CHECK(temperature_state IN ('fixed','none','unknown','sweep')),
                voltage_state TEXT NOT NULL CHECK(voltage_state IN ('fixed','none','unknown')), provenance_id TEXT REFERENCES run_provenance(provenance_id) ON DELETE RESTRICT,
                CHECK((corner_state = 'fixed') = (corner IS NOT NULL)),
                CHECK((temperature_state = 'fixed') = (temperature IS NOT NULL)),
                CHECK((voltage_state = 'fixed') = (voltage IS NOT NULL)),
                CHECK(profile_id IS NULL OR
                      (corner_state IN ('fixed','none') AND
                       temperature_state IN ('fixed','none') AND
                       voltage_state IN ('fixed','none'))),
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id),
                FOREIGN KEY (circuit_version_id, profile_id)
                    REFERENCES characterization_profiles(circuit_version_id, profile_id)
                    ON DELETE RESTRICT
            );
CREATE TABLE spec_measurements (
                measurement_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                spec_id TEXT NOT NULL,
                profile_id TEXT,  -- NULL = explicitly unscoped/legacy/manual
                source TEXT NOT NULL CHECK(source IN ('behavioral', 'spice', 'extraction', 'user')),
                value REAL ,
                corner TEXT,
                temperature REAL,
                voltage REAL,
                sim_type TEXT,
                tool_name TEXT,
                timestamp REAL,
                sim_run_id TEXT,
                notes TEXT,
                passes INTEGER,
                corner_state TEXT NOT NULL CHECK(corner_state IN ('fixed','none','unknown')),
                temperature_state TEXT NOT NULL CHECK(temperature_state IN ('fixed','none','unknown','sweep')),
                voltage_state TEXT NOT NULL CHECK(voltage_state IN ('fixed','none','unknown')), provenance_id TEXT REFERENCES run_provenance(provenance_id) ON DELETE RESTRICT,
                CHECK((corner_state = 'fixed') = (corner IS NOT NULL)),
                CHECK((temperature_state = 'fixed') = (temperature IS NOT NULL)),
                CHECK((voltage_state = 'fixed') = (voltage IS NOT NULL)),
                CHECK(profile_id IS NULL OR
                      (corner_state IN ('fixed','none') AND
                       temperature_state IN ('fixed','none') AND
                       voltage_state IN ('fixed','none'))),
                FOREIGN KEY (circuit_version_id, spec_id)
                    REFERENCES specs(circuit_version_id, spec_id) ON DELETE CASCADE,
                FOREIGN KEY (circuit_version_id, profile_id)
                    REFERENCES characterization_profiles(circuit_version_id, profile_id)
                    ON DELETE RESTRICT
            );
CREATE TABLE specs (
                spec_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                owner_path TEXT NOT NULL,
                spec_key TEXT NOT NULL,
                name TEXT NOT NULL,
                metric TEXT NOT NULL,
                op TEXT NOT NULL CHECK(op IN ('<', '<=', '>', '>=', '==')),
                target REAL NOT NULL,
                units TEXT,
                description TEXT,
                parent_id TEXT,
                declared_at TEXT,
                definition_scheme TEXT NOT NULL,
                definition_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(circuit_version_id, owner_path, spec_key),
                UNIQUE(circuit_version_id, spec_id),
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES specs(spec_id)
            );
CREATE INDEX idx_circuit_name_stage 
            ON circuit_versions(circuit_name, stage, created_at DESC)
        ;
CREATE UNIQUE INDEX idx_circuit_sealed_identity
            ON circuit_versions(
                circuit_name, stage, structure_scheme, structure_hash,
                spec_scheme, spec_set_hash
            )
            WHERE state = 'sealed'
        ;
CREATE INDEX idx_circuit_parent
            ON circuit_versions(parent_version_id)
        ;
CREATE INDEX idx_sim_results_circuit 
            ON simulation_results(circuit_version_id, parameter)
        ;
CREATE INDEX idx_sim_results_context
            ON simulation_results(
                circuit_version_id, profile_id, parameter, source, sim_type,
                corner, temperature, voltage, tool_name, timestamp
            )
        ;
CREATE INDEX idx_profiles_owner_test
            ON characterization_profiles(circuit_version_id, owner_path, test_name)
        ;
CREATE UNIQUE INDEX idx_current_sets_scoped_current
            ON net_current_sets(
                circuit_version_id, profile_id,
                corner_state, ifnull(corner, ''),
                temperature_state, ifnull(temperature, ''),
                voltage_state, ifnull(voltage, '')
            )
            WHERE state = 'current' AND profile_id IS NOT NULL
        ;
CREATE UNIQUE INDEX idx_current_sets_unscoped_current
            ON net_current_sets(
                circuit_version_id, corner_state, ifnull(corner, '')
            )
            WHERE state = 'current' AND profile_id IS NULL
        ;
CREATE INDEX idx_current_sets_profile_history
            ON net_current_sets(profile_id, corner, temperature, voltage, selected_at)
        ;
CREATE INDEX idx_net_currents_set
            ON net_currents(current_set_id, net_name, pin_id)
        ;
CREATE INDEX idx_modules_hash
            ON modules(module_hash_scheme, module_hash)
        ;
CREATE INDEX idx_modules_name 
            ON modules(module_name)
        ;
CREATE INDEX idx_circuit_modules_circuit 
            ON circuit_modules(circuit_version_id)
        ;
CREATE INDEX idx_circuit_modules_module
            ON circuit_modules(module_id)
        ;
CREATE UNIQUE INDEX idx_specs_owner_key
            ON specs(circuit_version_id, owner_path, spec_key)
        ;
CREATE INDEX idx_specs_definition
            ON specs(definition_scheme, definition_hash)
        ;
CREATE INDEX idx_spec_measurements_spec
            ON spec_measurements(spec_id)
        ;
CREATE INDEX idx_spec_measurements_context
            ON spec_measurements(
                circuit_version_id, profile_id, spec_id, source, corner,
                temperature, voltage, tool_name, timestamp
            )
        ;
COMMIT;
