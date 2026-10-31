"""Draw the supply-regulated VCO's LDO and ring-oscillator hierarchy.

Build the VCO without testbenches and extract its composition graph from the
live Module. No simulation runs, so edges have no voltages and specs are
unmeasured. Print the block inventory and Mermaid flowchart, and save
``silk-output/composition_vco.png`` (or under ``SILK_EXAMPLES_OUT``).

Run: ``python -m silk.examples.composition_graph_demo``.
"""
import os


from silk.examples.supply_regulated_vco_example import build_supply_regulated_vco  # noqa: E402

from silk.circuit.composition import (  # noqa: E402
    module_graph,
    graph_to_mermaid,
    save_graph_png,
)
from silk.examples import example_output_dir


def main():
    # The composed device only — no testbench rig around it.
    vco = build_supply_regulated_vco(with_tests=False)

    graph = module_graph(vco, depth=1)

    # A one-line inventory of what the walker found.
    kinds = {}
    for node in graph['nodes']:
        kinds[node['kind']] = kinds.get(node['kind'], 0) + 1
    print(f"Composition of '{graph['name']}':")
    print(f"  children : " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    print(f"  nets     : {len(graph['edges'])} shared (>=2 pins), "
          f"{len(graph['ports'])} top-level ports")
    for node in graph['nodes']:
        if node['kind'] == 'submodule':
            print(f"    - {node['name']} ({node['module_name']}): "
                  f"{node['device_count']} devices, "
                  f"specs {node['spec']['status']}")

    out_dir = example_output_dir()
    png_path = os.path.join(out_dir, 'composition_vco.png')
    save_graph_png(graph, png_path)
    print(f"\nPNG written: {png_path}")

    print("\n--- Mermaid (flowchart LR) ---")
    print(graph_to_mermaid(graph))


if __name__ == '__main__':
    main()
