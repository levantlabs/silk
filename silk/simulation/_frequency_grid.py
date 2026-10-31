"""Shared rounding for requested and completed discrete sweep grids."""
import math


def _whole_steps(quotient, uncertainty):
    nearest = round(quotient)
    if abs(quotient - nearest) <= uncertainty:
        return nearest
    return math.floor(quotient)


def logarithmic_span(start, stop, base):
    ratio = stop / start
    if math.isfinite(ratio):
        return math.log(ratio, base)
    return math.log(stop, base) - math.log(start, base)


def logarithmic_steps(points, start, stop, base):
    quotient = points * logarithmic_span(start, stop, base)
    # Never round a genuinely sub-step request into one step: ngspice's
    # floor remains zero there, which is the memory-runaway boundary.
    if quotient < 1:
        return 0
    return _whole_steps(quotient, 8 * math.ulp(quotient))
