# Choosing which simulator signals to save

For ngspice, silk saves every node voltage by default (`save all`), including
internal subcircuit nets and simulator-created device nodes. This lets you
inspect another node after a run without repeating the simulation.

Use `Module.set_save_list` when you need a smaller rawfile. It restricts the
node voltages written by the simulator. In contrast, `emit_view(signals=...)`
filters the waveform page built from an existing rawfile. You can also drop
device-internal nodes with `omit_device_internal_nodes`, described below.

## Selectors

Call `set_save_list` on the top-level `Module` before simulating. For a
registered testbench, configure `tb.netlist` before `tb.run()`.

This example configures a save list without launching a simulator or requiring
installed PDK models:

```python
from tempfile import TemporaryDirectory
from silk import Module, load_pdk
from silk.elements import elements

with TemporaryDirectory() as output:
    circuit = Module('sky130', output, 'save-list.cir',
                     pdk_details=load_pdk('sky130'))
    with circuit:
        elements.resistor('r1', p='vin', n='vout', value=1e3)
        elements.capacitor('c1', p='vout', n='0', value=1e-12)

    circuit.set_save_list(['vout'])
    circuit.set_save_list(['*'], exclude=['vin'], max_depth=0)
    circuit.set_save_list(None)  # restore the backend's default selection
```

The signature is `set_save_list(signals=None, *, exclude=None, max_depth=None)`.
`signals` and `exclude` accept a string or a sequence of strings. With only
`exclude` or `max_depth`, selection starts from all known nodes. `exclude`
subtracts matches; `max_depth=0` keeps only top-level nodes. Calling
`set_save_list(None)` with no other arguments clears all selection filters.

Selectors are case-insensitive exact names, globs (`*` for any number of
characters, `?` for one), or consecutive hierarchy components. For example,
`out` can match nested nodes named `out`; `xdut.*` selects nodes under the
emitted instance `xdut`.

Use **bare node names**, without `v(...)` wrappers. Both emitted paths
(`xdut.vmid`) and designer paths (`dut.vmid` for a placement named `dut`)
are accepted, starting at the module being configured. A connected subcircuit
port path selects its parent net. If a selector matches different sets through
emitted and designer spelling, the emitted match wins and silk warns.

Square brackets are literal brackets, never glob character classes. In an
instance component, `[k]` and `<k>` are equivalent: `xx1[0].xxy[1].g[0]` and
`xx1<0>.xxy<1>.g[0]` select the same node. `xx1[0]` selects that block's nodes,
`xx1[*]` selects every copy, and `xx1[?]` selects single-character indices.
At a node leaf, brackets remain literal: `g[0]` and `g<0>` are different nets;
`g[*]` matches names starting with `g[` and ending with `]`.

Selectors expand during netlist generation against silk's node inventory.
It contains top-level nets and internal subcircuit nets, but not ground `0`
or separate vectors for connected subcircuit ports (those alias parent nets).
It also includes physical intermediate nodes from compound sources.
Simulator-created device nodes such as `…#body` are absent, so `['*']` selects
design nodes without those parasitic nodes.

An unmatched include or exclude selector raises `CircuitError` with
suggestions when available. A selection that leaves no nodes also raises
`CircuitError`.
This catches typos before launch rather than relying on the simulator to
reject unknown nodes.

## Limits and backend differences

- **Node voltages only.** Current selectors such as `i(vb)` or `@m1[id]`
  raise `NotSupportedError`. Non-string selector entries raise `TypeError`.
- **Partial results.** The waveform explorer, `annotate_currents`, device-stress
  lint and metrics can only use saved vectors. Reading an unsaved signal
  raises a signal-read error naming it. Netlist generation prints a notice;
  ngspice also records `*** save list active` in the deck.
  To inspect an omitted node, expand the save list and rerun; changing viewer
  filters cannot recover unsaved data. See the [waveform explorer](waveform-explorer.md).
- **Device-current capture.** On ngspice, combining a save list with
  `save_device_currents=True` raises `SimulationError`: that mode requires
  a deck-level `.save all`.
- **Noise.** On ngspice, a deck containing a Noise analysis ignores the save
  list and prints a note, because noise output uses its own spectral vectors.
- **Xyce.** Without an explicit list, Xyce captures top-level bench nets plus
  node probes declared before `tb.run()`. Declarative metric node bindings
  (`add_metric(type=..., signal=...)` or `signals={slot: path, ...}`) also
  request capture, including bindings emitted by codegen. Unsupported metric
  node requests refuse before launch with the metric name and path. An
  explicit save list replaces automatic node selection; probes and metrics
  do not override exclusions. Clear the list to restore automatic capture.
  Silk renders nested paths with Xyce's `:` separator and accepts dot paths
  such as `tb.load_signal('xdut.vmid')` at readback.
- **Time axis.** A save list reduces the number of vectors, not time points.
  Storage savings depend on the circuit, analysis and vector types.

Rawfile-only viewer selection uses the same glob and array-index rules on
raw vector names. Circuit-backed module views supply checked node context
and also enable designer selectors. For viewer branch-current names, the final component is an
instance too: `i(vbias[0])` also matches `i(vbias<0>)`.
A wrapped viewer selector such as `v(out)` matches only the
full vector name; bare `out` also matches nested nodes. For a wrapped Xyce
viewer selector, use its raw spelling, for example `v(xx1[0]:xxy[1]:g[0])`.
Bare dotted paths and block selectors work in Xyce views too. The viewer treats
`.` and `:` outside square brackets as hierarchy separators. Silk-authored
net, port and instance names already reject `:` and `.`; these separators
matter when selecting externally produced raw vectors.

## Dropping device-internal nodes

To retain circuit nodes while suppressing simulator-created device nodes, set
`dut.omit_device_internal_nodes = True` before running registered tests. Each
testbench inherits the setting during preparation. For a standalone top-level
`Module`, set it on that module instead.

The default is `False`. Enabling it emits `save nointernals` in ngspice control
mode or `.save nointernals` in batch current mode. Circuit nets inside
subcircuits and additive device-parameter/current saves remain available,
subject to their usual capture restrictions.

The ngspice compatibility table must report `save_nointernals` (version 45 or
newer). Versions 42–44 and unknown versions raise `NotSupportedError` before
launch. Xyce and behavioral simulation also refuse `True`. Filtering cannot
combine with any explicit save-list selection; call `set_save_list(None)`
first, or keep device internals and use a selective save list.

Ngspice filters names containing `#`, which could remove real connections
inside extracted models. Silk therefore permits built-in ideal circuits without
opaque content, or model content matching the finite sky130 certificate in
`silk/pdk/data/sky130-internal-node-certificate.yaml`. It verifies dependency
paths and SHA-256 hashes at emission and again before launch. The certificate
covers the default primitive library's tt/ff/ss/sf/fs sections, its manifest
capacitor include and all 33 catalog factories.

Changed or missing model files, redirected dependencies, extra includes,
alternate libraries, unsupported factories and Monte Carlo model entry points
refuse filtering. Custom cards/includes, emitters and replacement netlisters
also refuse. Extracted HVL/RF cells are outside the certificate. Registering a
PDK alone does not qualify its model content.

The filtered deck records the certificate identity and producing ngspice
version. After refusal, keep `omit_device_internal_nodes=False`, use selective
saves, or use a compatible binary and the verified model bundle.
