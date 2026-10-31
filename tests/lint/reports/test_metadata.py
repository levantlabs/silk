from silk.lint.reports.metadata import ReportMetadata, SCHEMA_VERSION


def test_schema_version_is_1_0():
    assert SCHEMA_VERSION == "1.0"


def test_metadata_minimal_required_fields():
    md = ReportMetadata(
        silk_version="0.5.0",
        pdk_name="skywater130",
        pdk_version="1.0.0",
        module_name="ldo",
    )
    assert md.schema_version == "1.0"
    assert md.silk_version == "0.5.0"
    assert md.pdk_name == "skywater130"
    assert md.pdk_version == "1.0.0"
    assert md.generated_at  # auto-populated


def test_metadata_context_hash_stable_for_same_inputs():
    md1 = ReportMetadata(
        silk_version="0.5.0",
        pdk_name="skywater130",
        pdk_version="1.0.0",
        module_name="ldo",
    )
    md2 = ReportMetadata(
        silk_version="0.5.0",
        pdk_name="skywater130",
        pdk_version="1.0.0",
        module_name="ldo",
    )
    assert md1.context_hash == md2.context_hash


def test_metadata_context_hash_differs_on_pdk_change():
    md1 = ReportMetadata(
        silk_version="0.5.0",
        pdk_name="skywater130",
        pdk_version="1.0.0",
        module_name="ldo",
    )
    md2 = ReportMetadata(
        silk_version="0.5.0",
        pdk_name="skywater130",
        pdk_version="1.0.1",
        module_name="ldo",
    )
    assert md1.context_hash != md2.context_hash


def test_metadata_to_dict_and_from_dict_roundtrip():
    md = ReportMetadata(
        silk_version="0.5.0",
        pdk_name="skywater130",
        pdk_version="1.0.0",
        module_name="ldo",
    )
    d = md.to_dict()
    md2 = ReportMetadata.from_dict(d)
    assert md.context_hash == md2.context_hash
