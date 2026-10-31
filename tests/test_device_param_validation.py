"""Fail-loud device parameter validation on the factory path.

2026-08-23 review finding, verified: ``pdk.nmos1p8('m1', ..., m=1.5, w=-1,
l=0)`` was written verbatim into the simulation deck — ``AnalogElement.
validate()`` is a no-op and nothing on the factory emission path checked the
catalog's minval/maxval/integer constraints (only the legacy ``add_element``
path even warned). Decision (owner, 2026-08-23): raise at netlist-emission
time, i.e. in ``to_spice()``, the moment the deck line is produced.

Scope pins:
* only CALLER-supplied values are checked — a catalog default is the PDK
  author's own number;
* string values are SPICE literals passed through, never judged;
* comparison happens AFTER unit conversion, in the PDK's own unit (the unit
  minval/maxval are written in), with a 1e-9 relative tolerance so a
  boundary value that went through µm→m conversion does not false-positive;
* the legacy path keeps its historical warn-and-proceed
  (``ParameterConstraint.validate_value``) — separately tested.
"""
import pytest

from silk.exceptions import CircuitError


@pytest.fixture
def nmos(pdk):
    return pdk.nmos1p8


class TestFailLoudAtEmission:

    def test_negative_width_raises(self, nmos):
        dev = nmos('m1', d='d', g='g', s='s', b='s', w=-1, l=0.15, m=1)
        with pytest.raises(CircuitError, match=r'w=-1.*below the PDK minimum'):
            dev.to_spice()

    def test_zero_length_raises(self, nmos):
        dev = nmos('m1', d='d', g='g', s='s', b='s', w=1, l=0, m=1)
        with pytest.raises(CircuitError, match=r'l=0.*below the PDK minimum'):
            dev.to_spice()

    def test_fractional_multiplier_raises(self, nmos):
        dev = nmos('m1', d='d', g='g', s='s', b='s', w=1, l=0.15, m=1.5)
        with pytest.raises(CircuitError, match=r'm=1\.5 must be an integer'):
            dev.to_spice()

    def test_all_violations_reported_together(self, nmos):
        """The review's exact triple — every problem in ONE error, so the
        user fixes the device once, not three times."""
        dev = nmos('m1', d='d', g='g', s='s', b='s', w=-1, l=0, m=1.5)
        with pytest.raises(CircuitError) as e:
            dev.to_spice()
        msg = str(e.value)
        assert 'w=-1' in msg and 'l=0' in msg and 'm=1.5' in msg

    def test_valid_values_emit_unchanged(self, nmos):
        line = nmos('m1', d='d', g='g', s='s', b='s',
                    w=1, l=0.15, m=1).to_spice()
        assert 'w=1' in line and 'l=0.15' in line   # pdk fixture = user_units='pdk'

    def test_boundary_value_passes(self, nmos):
        """Exactly minval is legal — and must survive float unit-conversion
        noise (the 1e-9 rel_tol)."""
        line = nmos('m1', d='d', g='g', s='s', b='s',
                    w=0.42, l=0.15, m=1).to_spice()
        assert 'w=0.42' in line

    def test_defaults_never_checked(self, nmos):
        """An omitted parameter takes the catalog default — the PDK author's
        own number, exempt from caller validation."""
        nmos('m1', d='d', g='g', s='s', b='s').to_spice()   # must not raise

    def test_string_values_pass_through_unjudged(self, nmos):
        """A string is a SPICE literal (parameter reference, SI suffix) —
        silk cannot judge it and must not try."""
        nmos('m1', d='d', g='g', s='s', b='s',
             w='wparam', l=0.15, m=1).to_spice()             # must not raise
