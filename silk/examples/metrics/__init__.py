"""Examples of circuit measurements with :mod:`silk.metrics`.

Run a recipe with ``python -m silk.examples.metrics.<name>``. Each module
describes its circuit, measurements and printed results. The examples show
metric inputs, measurement windows and comparisons with circuit models.
Some comparisons use parameters fitted from the same simulation.

The examples require ngspice. ``rlc_step`` and ``rlc_filter_ac`` use ideal
devices and need no installed PDK models. Transistor examples need sky130
models. Use ``MPLBACKEND=Agg`` without a display. Results go to
``silk-output/`` or ``SILK_EXAMPLES_OUT``. The ring-oscillator example uses
temporary simulator directories that are removed after each run.
"""
