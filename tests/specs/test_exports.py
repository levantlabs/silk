"""Top-level silk package exports for the design-stage bridge."""


def test_spec_exports_from_top_level():
    from silk import Spec, Measurement, CalibrationBinding
    from silk import SpecOutcome, SpecReconciliationReport
    assert Spec is not None
    assert Measurement is not None
    assert CalibrationBinding is not None
    assert SpecOutcome is not None
    assert SpecReconciliationReport is not None


def test_silk_namespace_has_specs_submodule():
    import silk.specs
    assert hasattr(silk.specs, 'Spec')
    assert hasattr(silk.specs, 'Measurement')
