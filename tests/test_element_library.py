"""Tests for pynetlist.elements.ElementLibrary and the elements singleton."""

import pytest
from silk.elements import elements, ElementLibrary
from silk.elements.base import AnalogElement


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestElementsSingleton:

    def test_elements_is_element_library(self):
        """elements is an ElementLibrary instance."""
        assert isinstance(elements, ElementLibrary)

    def test_has_standard_devices(self):
        """elements has resistor, capacitor, inductor."""
        for name in ('resistor', 'capacitor', 'inductor'):
            assert hasattr(elements, name), f"elements missing '{name}'"

    def test_device_names_list(self):
        """device_names contains expected entries."""
        for name in ('resistor', 'capacitor'):
            assert name in elements.device_names


# ---------------------------------------------------------------------------
# Classes returned by ElementLibrary
# ---------------------------------------------------------------------------

class TestElementLibraryClasses:

    def test_resistor_is_class(self):
        """elements.resistor is a class."""
        assert isinstance(elements.resistor, type)

    def test_resistor_is_analog_element_subclass(self):
        """elements.resistor is an AnalogElement subclass."""
        assert issubclass(elements.resistor, AnalogElement)

    def test_resistor_spice_prefix(self):
        """Resistor class has SPICE prefix R."""
        assert elements.resistor._spice_prefix == 'R'


    def test_resistor_pin_names(self):
        """Resistor has pins p and n."""
        assert elements.resistor._pin_names == ('p', 'n')


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestElementInstantiation:

    def test_resistor_instantiates(self):
        """elements.resistor('r1', p=..., n=...) creates an instance."""
        r1 = elements.resistor('r1', p='vout', n='gnd')
        assert isinstance(r1, AnalogElement)
        assert r1.name == 'r1'

    def test_resistor_default_value(self):
        """Resistor uses default value=1000 when not supplied."""
        r1 = elements.resistor('r1', p='vout', n='gnd')
        assert r1.params['value'] == 1000

    def test_resistor_custom_value(self):
        """Resistor stores custom value."""
        r1 = elements.resistor('r1', p='vout', n='gnd', value=4700)
        assert r1.params['value'] == 4700


    def test_capacitor_default_value(self):
        """Capacitor uses default value=1e-12."""
        c1 = elements.capacitor('c1', p='vout', n='gnd')
        assert c1.params['value'] == 1e-12


# ---------------------------------------------------------------------------
# SPICE output
# ---------------------------------------------------------------------------

class TestElementSpice:

    def test_resistor_spice_format(self):
        """Resistor: Rr1 vout gnd <value>."""
        r1 = elements.resistor('r1', p='vout', n='gnd', value=1e3)
        spice = r1.to_spice()
        assert spice.startswith('Rr1')
        assert 'vout' in spice
        assert 'gnd' in spice


    def test_capacitor_spice_format(self):
        """Capacitor: Cc1 vout gnd <value>."""
        c1 = elements.capacitor('c1', p='vout', n='gnd', value=1e-12)
        spice = c1.to_spice()
        assert spice.startswith('Cc1')
        assert '1p' in spice

    def test_spice_ends_with_newline(self):
        """to_spice() output ends with newline."""
        r1 = elements.resistor('r1', p='a', n='b')
        assert r1.to_spice().endswith('\n')


# ---------------------------------------------------------------------------
# Custom YAML path
# ---------------------------------------------------------------------------

class TestElementLibraryCustomYaml:

    def test_custom_yaml_loads(self, tmp_path):
        """ElementLibrary can load a custom YAML file."""
        yaml_content = (
            'myres:\n'
            '  spice_prefix: R\n'
            '  pinList: [a, b]\n'
            '  paramConstraints:\n'
            '    - name: value\n'
            '      default: 100\n'
        )
        p = tmp_path / 'custom.yaml'
        p.write_text(yaml_content)
        lib = ElementLibrary(str(p))
        assert hasattr(lib, 'myres')
        r = lib.myres('r1', a='n1', b='n2')
        assert r.params['value'] == 100


# ---------------------------------------------------------------------------
# Integration: elements + ckt.add()
# ---------------------------------------------------------------------------

class TestElementsWithCktAdd:

    def test_add_resistor_to_netlist(self, pdk, empty_netlist):
        """Ideal resistor from ElementLibrary can be added via ckt.add()."""
        ckt = empty_netlist
        r1 = elements.resistor('r1', p='vout', n='gnd', value=1e3)
        ckt.add(r1)
        assert 'r1' in ckt.elem_dict

    def test_netlisting_includes_resistor(self, pdk, empty_netlist):
        """Netlist output includes the resistor SPICE line."""
        from silk.circuit.spice_netlister import PySpiceNetlister
        ckt = empty_netlist
        r1 = elements.resistor('r1', p='vout', n='gnd', value=4700)
        ckt.add(r1)
        netlister = PySpiceNetlister(pdk)
        result = netlister.generate_device_instances(ckt.elem_dict)
        assert 'Rr1' in result
        assert 'vout' in result


@pytest.mark.parametrize('name', ['vsource', 'isource'])
def test_source_factories_are_absent(name):
    assert name not in elements.device_names
    with pytest.raises(AttributeError, match=name):
        getattr(elements, name)


def test_passive_elements_do_not_warn():
    """Passive constructors stay silent."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        elements.resistor('r1', p='vout', n='gnd', value=1e3)
        elements.capacitor('c1', p='vout', n='gnd', value=1e-12)
