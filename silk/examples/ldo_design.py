"""
Parameterized LDO Design

Devices are instantiated directly via the factory API and added with
ldo.add() — pins and parameters are named keyword arguments, parameters
can be updated by direct attribute assignment (e.g. nmos_output.w = 12.0).
"""

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk import load_pdk
from silk import sources
from silk.elements import elements
from silk.simulation import analysis
# Let matplotlib use MPLBACKEND or matplotlibrc.
import matplotlib.pyplot as plt
from silk.examples import example_output_dir


class LDODesign:
    """Parameterized NMOS-output LDO regulator design."""

    # All W/L values in µm (PDK-native units for sky130)
    DEFAULT_PARAMS = {
        'm_input_diff_pair': 5,
        'm_pmos_load': 5,
        'm_input_bias': 2,
        'm_output_bias_ratio': 10,
        'm_nmos_output_ratio': 200,
        'input_width': 2.0,
        'input_length': 0.5,
        'load_width': 2.0,
        'load_length': 0.5,
        'output_width': 10.0,
        'output_length': 0.15,
        'bias_width': 2.0,
        'bias_length': 1.0,
    }

    def __init__(self, pdk_instance, dir_folder, fname, params=None):
        self.pdk = pdk_instance
        self.dir_folder = dir_folder
        self.fname = fname
        self.params = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)

    def build(self):
        """Build the LDO subcircuit module and return it."""
        p = self.params

        m_output_bias = p['m_input_bias'] * p['m_output_bias_ratio']
        m_nmos_output = p['m_nmos_output_ratio']

        ldo = Module('PDK_skywater130', self.dir_folder, self.fname,
                     pdk_details=self.pdk, module_name='ldo_nmos')
        ldo.define_module_ports({'vdd': PortDir.POWER, 'vss': PortDir.GROUND,
                                 'vin': PortDir.IN, 'vout': PortDir.OUT, 'vnbias': PortDir.IN})

        # Diff pair
        nmos_input_p = self.pdk.nmos1p8('nmos_input_p',
                                         d='vpmos_mirror', g='vin', s='vdiffdrain', b='vss',
                                         w=p['input_width'], l=p['input_length'],
                                         m=p['m_input_diff_pair'])
        nmos_input_n = self.pdk.nmos1p8('nmos_input_n',
                                         d='vst1_out', g='vout', s='vdiffdrain', b='vss',
                                         w=p['input_width'], l=p['input_length'],
                                         m=p['m_input_diff_pair'])

        # Active load
        pmos_load_p = self.pdk.pmos1p8('pmos_load_p',
                                        d='vpmos_mirror', g='vpmos_mirror', s='vdd', b='vdd',
                                        w=p['load_width'], l=p['load_length'],
                                        m=p['m_pmos_load'])
        pmos_load_n = self.pdk.pmos1p8('pmos_load_n',
                                        d='vst1_out', g='vpmos_mirror', s='vdd', b='vdd',
                                        w=p['load_width'], l=p['load_length'],
                                        m=p['m_pmos_load'])

        # Bias current source
        nmos_bias_diffpair = self.pdk.nmos1p8('nmos_bias_diffpair',
                                               d='vdiffdrain', g='vnbias', s='vss', b='vss',
                                               w=p['bias_width'], l=p['bias_length'],
                                               m=p['m_input_bias'])

        # Output NMOS
        nmos_output = self.pdk.nmos1p8('nmos_output',
                                        d='vdd', g='vst1_out', s='vout', b='vss',
                                        w=p['output_width'], l=p['output_length'],
                                        m=m_nmos_output)

        # Output bias
        nmos_bias_output = self.pdk.nmos1p8('nmos_bias_output',
                                             d='vout', g='vnbias', s='vss', b='vss',
                                             w=p['bias_width'], l=p['bias_length'],
                                             m=m_output_bias)

        # Bias diode
        nmos_bias_diode = self.pdk.nmos1p8('nmos_bias_diode',
                                            d='vnbias', g='vnbias', s='vss', b='vss',
                                            w=p['bias_width'], l=p['bias_length'], m=1)

        ldo.add(nmos_input_p, nmos_input_n,
                pmos_load_p, pmos_load_n,
                nmos_bias_diffpair,
                nmos_output,
                nmos_bias_output,
                nmos_bias_diode)

        return ldo


if __name__ == '__main__':
    import os
    pdk130 = load_pdk('sky130')

    dir_folder = example_output_dir()
    fname = 'ldo_example.cir'

    db_path = os.path.join(dir_folder, 'ldo_design.db')

    # Build the LDO subcircuit
    ldo_design = LDODesign(pdk130, dir_folder, fname)
    subckt_ldo = ldo_design.build()

    # Top-level netlist with database tracking
    main_netlist = Module('PDK_skywater130', dir_folder, fname,
                          pdk_details=pdk130,
                          db_path=db_path)

    # Add LDO module instance
    main_netlist.add_module(
        subckt_ldo, 'ldo_top',
        nets={
            'vdd':    'vdd3',
            'vss':    '0',
            'vin':    'vin_ldo',
            'vout':   'vldo_out',
            'vnbias': 'vnbias',
        }
    )

    # Sources
    main_netlist.add_source(sources.CurrentSource(
        'biascurrent', positive='vdd3', negative='vnbias', dc=100e-6))

    main_netlist.add_source(sources.VoltageSource(
        'supply', positive='vdd3', negative='0', dc=3.3, ac=1))

    main_netlist.add_source(sources.VoltageSource(
        'ldo_input', positive='vin_ldo', negative='0', dc=1.8,
        pwl=[0, 0, 99e-9, 0, 100e-9, 0.1, 199e-9, 0.1, 200e-9, 0]))

    main_netlist.add_source(sources.CurrentSource(
        'ldo_load_current', positive='0', negative='vldo_out'))

    main_netlist.add(elements.capacitor('load', p='vldo_out', n='0', value=100e-12))
    main_netlist.add_sim_type(analysis.Transient(step=10e-9, stop=50e-6))

    main_netlist.create_circuit_version("Initial LDO design")

    main_netlist.print_elements()
    main_netlist.print_nets()

    # simulate() generates the deck, runs SPICE, and loads results.
    main_netlist.simulate(corner='tt')
    main_netlist.get_signal_names()

    fig = main_netlist.plot(
        'vdd3', 'vsupply:I', 'vnbias', 'xldo_top.vst1_out',
        'vldo_out', 'vin_ldo', show=False)
    for ax, limits in zip(fig.axes[2:5], ([0, 2], [0, 4], [1, 2])):
        ax.set_ylim(limits)
    plt.show()
