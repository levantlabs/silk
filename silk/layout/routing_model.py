"""silk.layout.routing_model — metal-stack abstraction for layout estimation.

A ``RoutingModel`` carries the constants for ONE metal layer, either
constructed directly (unchanged since Phase 1) or via ``RoutingModel.from_pdk``
/ ``RoutingModel.stack_from_pdk``, which read a PDK's manifest-declared
``routing:`` section. No PDK-specific number lives in this module —
every value is caller-supplied or sourced from the manifest at call time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(frozen=True)
class RoutingModel:
    """Single-layer metal-stack constants for wire sizing and parasitic estimation.

    Parameters
    ----------
    sheet_resistance : float
        Wire sheet resistance, ohms/square.
    cap_per_um2 : float
        Capacitance per unit area, fF/μm².
    j_max : float
        Electromigration current density limit, A/μm.
    min_width : float
        Minimum manufacturable wire width, μm.
    max_width : float
        Maximum wire width (routing channel limit), μm.
    thickness : float
        Metal thickness, μm. Reserved for future fringe-capacitance work;
        unused in Phase 1. Defaults to 0.36 μm.
    """
    sheet_resistance: float   # ohms/square
    cap_per_um2:      float   # fF/um²
    j_max:            float   # A/um
    min_width:        float   # um
    max_width:        float   # um
    thickness:        float = 0.36   # um — reserved, unused in Phase 1

    @property
    def id(self) -> str:
        """Stable string identifier for snapshot metadata.

        Deterministic given the same field values; different RoutingModel
        instances with identical fields produce the same id.
        """
        return (
            f"{self.sheet_resistance:.4g}"
            f"|{self.cap_per_um2:.4g}"
            f"|{self.j_max:.4g}"
            f"|{self.min_width:.4g}"
            f"|{self.max_width:.4g}"
            f"|{self.thickness:.4g}"
        )

    # -- manifest-sourced construction (TASK-25) ---------------------------

    @staticmethod
    def _manifest_routing(pdk) -> tuple:
        """Return ``(pdk_name, routing_dict)`` for *pdk* (a loaded PDK object
        or a bare PDK name string), resolved through ``silk.pdk.registry`` —
        never by opening a manifest file directly. Lazy import: this keeps
        ``silk.layout`` a lazy consumer of ``silk.pdk``, matching every other
        PDK access in this package (see docs/architecture.md's layout->pdk
        edge)."""
        from silk.pdk import registry as _registry

        name = pdk if isinstance(pdk, str) else getattr(pdk, 'pdk_name', None)
        if not name:
            raise ValueError(
                "RoutingModel: pdk has no usable name — pass a PDK name "
                "string, or a PDK object loaded via silk.load_pdk() (whose "
                ".pdk_name is set)."
            )
        m = _registry.manifest(name)
        routing = m.get('routing')
        if not routing:
            raise ValueError(
                f"RoutingModel: PDK {name!r} declares no 'routing' section in "
                f"its manifest ({m['_manifest_dir']}/{name}.pdk.yaml) — no "
                f"metal-stack data has been sourced for this PDK yet. "
                f"Construct RoutingModel(...) directly with your own values, "
                f"or add a sourced 'routing:' section to its .pdk.yaml."
            )
        return name, routing

    @classmethod
    def from_pdk(cls, pdk, layer: Optional[str] = None) -> 'RoutingModel':
        """Build a ``RoutingModel`` for one layer of *pdk*'s manifest-declared
        metal stack.

        Parameters
        ----------
        pdk : BasePDK or str
            A PDK object (as returned by ``silk.load_pdk``) or a bare PDK
            name — either resolves through ``silk.pdk.registry``.
        layer : str, optional
            Which manifest-declared layer to use. Defaults to the manifest's
            ``routing.default_layer``.

        Raises
        ------
        ValueError
            The PDK's manifest has no ``routing`` section, or *layer* (or the
            manifest's ``default_layer``) is not one of its declared layers.
        """
        name, routing = cls._manifest_routing(pdk)
        layers = routing.get('layers') or {}
        layer = layer or routing.get('default_layer')
        if layer not in layers:
            raise ValueError(
                f"RoutingModel.from_pdk: PDK {name!r} manifest has no routing "
                f"layer {layer!r}; declared layers: {sorted(layers)}."
            )
        entry = layers[layer]
        return cls(
            sheet_resistance=entry['sheet_resistance'],
            cap_per_um2=entry['cap_per_um2'],
            j_max=entry['j_max'],
            min_width=entry['min_width'],
            max_width=entry['max_width'],
            thickness=entry.get('thickness', 0.36),
        )

    @classmethod
    def stack_from_pdk(cls, pdk) -> Dict[str, 'RoutingModel']:
        """All of *pdk*'s manifest-declared layers, as an ordered mapping of
        layer name to ``RoutingModel``.

        Order matches the manifest's ``routing.layers`` declaration order —
        lowest layer first — which is the escalation order
        ``silk.layout.estimator``'s ``escalation_rule`` expects.
        """
        name, routing = cls._manifest_routing(pdk)
        layers = routing.get('layers') or {}
        if not layers:
            raise ValueError(
                f"RoutingModel.stack_from_pdk: PDK {name!r} manifest 'routing' "
                f"section declares no layers."
            )
        return {
            layer_name: cls.from_pdk(pdk, layer=layer_name)
            for layer_name in layers
        }


#: Type accepted wherever a caller may pass either one layer or an ordered
#: stack of layers (dict of layer-name -> RoutingModel, lowest first, or a
#: plain sequence in the same order). See ``silk.layout.estimator``.
RoutingModelOrStack = Union[RoutingModel, Dict[str, RoutingModel], tuple, list]
