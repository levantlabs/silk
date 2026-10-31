"""Integration tests for the Variable → VariableSpace → SPICE netlist pipeline.

These tests verify the chain:
  1. Variable created externally
  2. Passed to update_element() as paramValue → stored as ParameterConstraint.value
  3. Included in a VariableSpace
  4. space.apply(point) sets mv.value = x
  5. generate_netlist() reads constraint.value.value (i.e. mv.value) into SPICE line

No ngspice is invoked — only generate_netlist() and file reads.
"""

import os
import pytest

from silk.circuit.variable import Variable
from silk.optimize.parameter_space import VariableSpace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_netlist(deck_path):
    """Return the full content of the generated netlist file."""
    with open(deck_path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Class 1: Variable value propagation into SPICE
# ---------------------------------------------------------------------------

class TestVariableToSpice:
    """Tests that Variable.value changes propagate into the SPICE netlist."""

    def test_module_variable_value_appears_in_netlist(self, basic_netlist):
        """update_element with a Variable writes mv.value into the SPICE line."""
        ckt = basic_netlist
        w_var = Variable('w', value=3.0, bounds=(0.5, 20.0))

        ckt.update_element('mn1', ['w'], [w_var])
        deck_path = ckt.generate_netlist(corner='tt')

        content = _read_netlist(deck_path)
        # The basepdk uses f'{p}={param_val}'; for float 3.0 this writes 'w=3.0'
        assert 'w=3.0' in content or 'w=3' in content, (
            f"Expected 'w=3.0' or 'w=3' in netlist, got:\n{content}"
        )

    def test_module_variable_value_change_updates_netlist(self, basic_netlist):
        """Changing mv.value before generate_netlist reflects the new value."""
        ckt = basic_netlist
        w_var = Variable('w', value=1.0, bounds=(0.5, 20.0))
        ckt.update_element('mn1', ['w'], [w_var])

        # First generation with value=1.0
        w_var.value = 1.0
        deck_path = ckt.generate_netlist(corner='tt')
        content_first = _read_netlist(deck_path)
        assert 'w=1.0' in content_first or 'w=1' in content_first, (
            f"Expected 'w=1.0' in first netlist:\n{content_first}"
        )

        # Change value and regenerate
        w_var.value = 8.0
        deck_path = ckt.generate_netlist(corner='tt')
        content_second = _read_netlist(deck_path)

        # New value must appear
        assert 'w=8.0' in content_second or 'w=8' in content_second, (
            f"Expected 'w=8.0' in second netlist:\n{content_second}"
        )
        # Old value (1.0) must not appear as a w= assignment anymore.
        # A safe check: the mn1 device line should not contain w=1 after change.
        # Find the XMmn1 line
        mn1_line = next(
            (ln for ln in content_second.splitlines() if 'mn1' in ln.lower() and ln.strip().startswith('XM')),
            None
        )
        if mn1_line is not None:
            assert 'w=1.0' not in mn1_line and 'w=1 ' not in mn1_line, (
                f"Old w value still present in mn1 line: {mn1_line}"
            )

    def test_shared_variable_controls_two_elements(self, basic_netlist):
        """One Variable linked to both mn1 and mp1 widths controls both."""
        ckt = basic_netlist
        shared_w = Variable('w', value=5.0, bounds=(0.5, 20.0))

        ckt.update_element('mn1', ['w'], [shared_w])
        ckt.update_element('mp1', ['w'], [shared_w])

        deck_path = ckt.generate_netlist(corner='tt')
        content = _read_netlist(deck_path)

        # Count how many times w=5.0 (or w=5) appears — should be at least 2
        import re
        matches = re.findall(r'w=5(?:\.0)?(?:\s|$)', content)
        assert len(matches) >= 2, (
            f"Expected w=5.0 to appear at least twice (NMOS + PMOS), found {len(matches)}:\n{content}"
        )


# ---------------------------------------------------------------------------
# Class 2: VariableSpace → netlist pipeline
# ---------------------------------------------------------------------------

class TestVariableSpaceToNetlist:
    """Tests the full VariableSpace → generate_netlist() pipeline."""

    def test_apply_then_generate_reflects_values(self, basic_netlist):
        """space.apply({'w': 4.0, 'l': 0.5}) writes both values into the netlist."""
        ckt = basic_netlist
        w_var = Variable('w', value=1.0, bounds=(0.5, 20.0))
        l_var = Variable('l', value=0.15, bounds=(0.15, 2.0))

        ckt.update_element('mn1', ['w', 'l'], [w_var, l_var])

        space = VariableSpace([w_var, l_var])
        space.apply({'w': 4.0, 'l': 0.5})

        deck_path = ckt.generate_netlist(corner='tt')
        content = _read_netlist(deck_path)

        assert 'w=4.0' in content or 'w=4' in content, (
            f"Expected w=4.0 in netlist:\n{content}"
        )
        assert 'l=0.5' in content, (
            f"Expected l=0.5 in netlist:\n{content}"
        )

    def test_sweep_generates_different_netlists(self, basic_netlist):
        """Sampling 5 grid points and generating a netlist for each produces distinct content."""
        ckt = basic_netlist
        ckt.set_output_options(mode='overwrite')
        w_var = Variable('w', value=1.0, bounds=(1.0, 5.0))
        ckt.update_element('mn1', ['w'], [w_var])

        space = VariableSpace([w_var])
        points = space.sample(5, method='grid')

        contents = set()
        for pt in points:
            space.apply(pt)
            deck_path = ckt.generate_netlist(corner='tt')
            contents.add(_read_netlist(deck_path))

        assert len(contents) > 1, (
            "Expected at least 2 distinct netlists from 5 grid points, got only 1 unique netlist"
        )

    def test_apply_integer_var_writes_int(self, basic_netlist):
        """INTEGER Variable writes rounded integer into netlist."""
        ckt = basic_netlist

        # 'm' is the multiplicity parameter — dimensionless integer in sky130
        m_var = Variable('m', value=1, bounds=(1, 16), var_type='integer')
        ckt.update_element('mn1', ['m'], [m_var])

        space = VariableSpace([m_var])
        # Apply a float that rounds to 4
        space.apply({'m': 4})

        deck_path = ckt.generate_netlist(corner='tt')
        content = _read_netlist(deck_path)

        assert 'm=4' in content, (
            f"Expected 'm=4' in netlist for integer variable:\n{content}"
        )

    def test_lhs_sweep_all_distinct_netlists(self, basic_netlist):
        """5 LHS points all generate distinct netlists."""
        ckt = basic_netlist
        ckt.set_output_options(mode='overwrite')
        w_var = Variable('w', value=1.0, bounds=(0.5, 20.0))
        ckt.update_element('mn1', ['w'], [w_var])

        space = VariableSpace([w_var])
        points = space.sample(5, method='lhs')

        # LHS guarantees stratification — all 5 w values should be distinct
        assert len({pt['w'] for pt in points}) == 5, (
            "LHS should produce 5 distinct w values"
        )

        contents = set()
        for pt in points:
            space.apply(pt)
            deck_path = ckt.generate_netlist(corner='tt')
            contents.add(_read_netlist(deck_path))

        assert len(contents) == 5, (
            f"Expected 5 distinct netlists from 5 LHS points, got {len(contents)}"
        )


# ---------------------------------------------------------------------------
# Class 3: apply() + read-back round-trip
# ---------------------------------------------------------------------------

class TestVariableSpaceApplyRoundtrip:
    """Tests that apply() + read-back gives correct variable values."""

    def test_apply_updates_variable_value(self, basic_netlist):
        """After space.apply({'w': 7.0}), w_var.value == 7.0."""
        w_var = Variable('w', value=1.0, bounds=(0.5, 20.0))
        space = VariableSpace([w_var])

        space.apply({'w': 7.0})

        assert w_var.value == 7.0, (
            f"Expected w_var.value == 7.0, got {w_var.value}"
        )

    def test_apply_point_from_sample(self, basic_netlist):
        """space.sample(1, 'random')[0] applied → each variable's value matches the point."""
        w_var = Variable('w', value=1.0, bounds=(0.5, 20.0))
        l_var = Variable('l', value=0.15, bounds=(0.15, 2.0))
        space = VariableSpace([w_var, l_var])

        pts = space.sample(1, method='random')
        pt = pts[0]
        space.apply(pt)

        assert w_var.value == pt['w'], (
            f"w_var.value={w_var.value} does not match point w={pt['w']}"
        )
        assert l_var.value == pt['l'], (
            f"l_var.value={l_var.value} does not match point l={pt['l']}"
        )

    def test_multiple_apply_calls_update_correctly(self, basic_netlist):
        """Applying point A then point B leaves variable with point B's value."""
        w_var = Variable('w', value=1.0, bounds=(0.5, 20.0))
        space = VariableSpace([w_var])

        point_a = {'w': 2.5}
        point_b = {'w': 9.9}

        space.apply(point_a)
        assert w_var.value == 2.5, f"After apply(A): expected 2.5, got {w_var.value}"

        space.apply(point_b)
        assert w_var.value == 9.9, f"After apply(B): expected 9.9, got {w_var.value}"
