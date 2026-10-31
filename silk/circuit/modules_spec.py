"""Spec declaration/reconciliation and behavioral-model methods for :class:`~silk.circuit.modules.Module`.

Split out of ``modules.py`` as a per-subsystem **part**. ``_SpecMethods`` is a pure mixin: no state, no separate identity. silk.specs / behavioral-engine / numpy are imported lazily inside the methods that need them.
"""

from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:  # pragma: no cover
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object


_EXECUTIONS = ('vector', 'sample')


class _SpecMethods(_Base):
    """Spec / reconciliation / behavioral-model methods mixed into ``Module``. Holds no state."""

    def set_behavioral_model(self, fn, params=None, *, execution='vector'):
        """Attach a Python callable as the behavioral model for this subcircuit.

        Parameters
        ----------
        fn : callable
            Use
            ``fn(ports, t, params)``. Read ``ports.VIN.v`` and assign
            ``ports.VOUT.v`` or ``ports.VOUT.i``; return ``None``. String-key
            access to ports and returned values raise ``SimulationError``
            with the supported spelling. The parameter view supports both
            ``params.gain`` and ``params['gain']`` over the live attachment
            dictionary. Statistical declarations resolve once per run.
        params : dict, optional
            Initial parameter values. Stored by reference so that
            calibration writes are visible on subsequent simulation
            runs without requiring the user to re-attach the model.
        execution : {'vector', 'sample'}, optional
            Selects the calling contract; defaults to ``'vector'``. ``'vector'``
            calls the model once per run (once per DC point) with the whole
            time array; ``'sample'`` calls it once per time point. The
            contract is never inferred or switched during a run, and it is
            part of the model's behavioural identity.

        Raises
        ------
        TypeError
            If ``fn`` is not callable.
        ValueError
            If an explicit ``execution`` is neither ``'vector'`` nor ``'sample'``.
        silk.exceptions.NotSupportedError
            If ``params`` holds a statistical declaration nested inside a
            container instead of as a top-level entry.

        """
        if not callable(fn):
            raise TypeError(
                f"set_behavioral_model() requires a callable, got {type(fn).__name__!r}"
            )
        if not isinstance(execution, str) or execution not in _EXECUTIONS:
            raise ValueError(
                f"set_behavioral_model() execution must be 'vector' or 'sample', "
                f"got {execution!r}")
        from silk.simulation._behavioral_variation import validate_params
        validate_params(params, self.module_name)
        self._behavioral_model = fn
        self._behavioral_params = params
        self._behavioral_form = 'ports'
        self._behavioral_execution = execution

    def declare_spec(
        self,
        name,
        metric,
        op,
        target,
        units='',
        description='',
        parent=None,
        calibration=None,
        *,
        key=None,
    ):
        """Create a first-class Spec owned by this Module.

        Parameters
        ----------
        name : str
            Short label for the spec (e.g. 'system_NF', 'lna_NF').
        metric : str
            Either the name of a metric declared on one of this module's
            ``ModuleTestbench`` instances, or a dotted path of the form
            ``'<object>.<attribute>'`` pointing into Pillar 2 state
            (e.g. ``'net.vout.peak_current'``). Metric resolution
            happens at reconciliation time.
        op : str
            One of ``'<'``, ``'<='``, ``'>'``, ``'>='``, ``'=='``.
        target : float
            The declared target value.
        units : str, optional
        description : str, optional
        parent : Spec, optional
            Explicit parent reference. When provided, the new spec is
            appended to ``parent.children``.
        calibration : CalibrationBinding, optional
            When set, measurements from allowed sources write onto
            ``calibration.target_module._behavioral_params[calibration.param_name]``
            during reconcile_specs().
        key : str, optional
            Stable local identity. When omitted, the exact ``name`` is used.

        Returns
        -------
        Spec
            The newly-created Spec, added to ``self._specs``.

        Raises
        ------
        silk.exceptions.DuplicateNameError
            If a spec with the same key (``key``, else ``name``) is already
            declared on this module.
        TypeError
            If ``target`` is not a real number (booleans are refused) or
            ``key`` is not a string.
        ValueError
            If ``key`` is empty or whitespace-only.
        """
        import inspect
        from silk.exceptions import DuplicateNameError
        from silk.specs import Spec
        candidate_key = name if key is None else key
        if any(existing.key == candidate_key for existing in self._specs):
            raise DuplicateNameError(
                f"Spec key {candidate_key!r} is already declared on module "
                f"{self.module_name!r}"
            )
        frame = inspect.stack()[1]
        declared_at = f'{os.path.basename(frame.filename)}:{frame.lineno}'
        spec = Spec(
            name=name,
            module=self,
            metric=metric,
            op=op,
            target=target,
            units=units,
            description=description,
            parent=parent,
            calibration=calibration,
            declared_at=declared_at,
            key=key,
        )
        self._specs.append(spec)
        return spec

    def reconcile_specs(self, prefer_source=None):
        """Walk this module and its submodules, reconcile every Spec.

        Traversal order is pre-order depth-first: this module's specs
        first, then each submodule's specs. Returns a single
        ``SpecReconciliationReport`` covering the entire tree.

        Parameters
        ----------
        prefer_source : str, optional
            If provided, the reconciler uses the measurement from that
            source when available, falling back to the latest otherwise.

        Returns
        -------
        SpecReconciliationReport
        """
        from silk.specs.reconciler import reconcile
        all_specs = [(instance.owner_path, spec)
                     for instance in self.iter_instances(include_self=True)
                     for spec in instance.module._specs]
        return reconcile(module=self, specs=all_specs, prefer_source=prefer_source)

    def calibrate_behavioral(self, param_name, value):
        """Directly overwrite a behavioral-model parameter.

        This is the manual path — normally calibration fires as a
        side effect of ``reconcile_specs()`` via a
        ``CalibrationBinding``. Use this method when you want to push
        a measured value (e.g. from a lab measurement file) onto the
        behavioral model without going through the reconciler.

        Raises
        ------
        RuntimeError
            If the behavioral model is not parameterized
            (``_behavioral_params`` is ``None``).
        KeyError
            If ``param_name`` is not already a key in the params dict.
            Missing keys are declaration errors — use
            ``set_behavioral_model(fn, params={...})`` to declare.
        """
        if self._behavioral_params is None:
            raise RuntimeError(
                f"Module '{self.module_name}' is not parameterized; "
                f"call set_behavioral_model(fn, params={{...}}) first."
            )
        if param_name not in self._behavioral_params:
            raise KeyError(
                f"Param '{param_name}' is not declared on module "
                f"'{self.module_name}'. Known params: "
                f"{sorted(self._behavioral_params.keys())}"
            )
        from silk.simulation._behavioral_variation import refuse_calibration
        refuse_calibration(self._behavioral_params, param_name, self.module_name)
        self._behavioral_params[param_name] = value

    def _iter_all_specs(self):
        """Yield all Spec objects on self and all nested submodules.

        Pre-order: this module's specs first, then walk each
        submodule INSTANCE's specs recursively. Uses
        ``self._element_mgr.elem_dict`` which is keyed by instance name
        (so ``'lna1'`` and ``'lna2'`` are distinct) rather than
        ``self.module_dict`` which is keyed by module type.
        """
        for spec in self._specs:
            yield spec
        elem_dict = getattr(self._element_mgr, 'elem_dict', {}) or {}
        for entry in elem_dict.values():
            sub = entry.get('device') if isinstance(entry, dict) else None
            if sub is None:
                continue
            if hasattr(sub, '_iter_all_specs'):
                yield from sub._iter_all_specs()

    def simulate_behavioral(self, t_step, t_stop):
        """Run the behavioral backend with an implicit transient analysis.

        Runs on the ``'behavioral'`` backend through the shared simulation
        and rawfile reader path. The grid includes ``t_stop``; ``results_file``
        names the written rawfile. ``Net.voltage`` and ``_behavioral_t`` stay
        available for direct readers and lint. The caller's analysis
        declarations and simulator selection are restored after this call,
        including when the run fails, so a later ``simulate()`` or testbench
        run uses the backend that was selected before.

        Parameters
        ----------
        t_step : float
            Transient time step in seconds.
        t_stop : float
            Stop time in seconds; the last grid point is ``t_stop``.

        Returns
        -------
        Module
            This module, as returned by :meth:`simulate`; read results with
            ``load_signal`` or ``results_file``.

        Raises
        ------
        silk.exceptions.SimulationError
            If the behavioral engine cannot run the circuit (for example
            feedback across modelled blocks).
        silk.exceptions.NotSupportedError
            If an element or topology is outside the behavioral engine's
            subset.
        NotImplementedError
            If an independent source uses a waveform with no numpy model
            (for example ``exponential``).
        """
        from silk.simulation import Transient
        analysis = Transient(step=t_step, stop=t_stop)
        selection = (self._simulator, getattr(self, '_simulator_explicit', False))
        self.simulator = 'behavioral'
        declared = self.sim_analysis_dict
        self.sim_analysis_dict = {'transient': analysis}
        try:
            return self.simulate()
        finally:
            self.sim_analysis_dict = declared
            self._simulator, self._simulator_explicit = selection
