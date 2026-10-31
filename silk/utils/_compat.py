"""Version-portability shims. One copy of each decision.

``np.trapz`` became ``np.trapezoid`` in NumPy 2.0. Twelve call sites across
silk/metrics and silk/signals used the new name bare, so on NumPy 1.x the
import succeeded and the METRIC died later with AttributeError — while
``silk/circuit/modules_results.py`` had quietly carried its own correct shim
since the rename. Thirteen independent copies of one decision is how that
happens; this module is the one copy.
"""
import numpy as np

#: Trapezoidal integration under either NumPy major version.
trapezoid = getattr(np, 'trapezoid', None) or np.trapz
