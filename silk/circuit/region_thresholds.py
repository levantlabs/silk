"""Validated per-PDK region policy data, selected through silk settings.

This module only loads policy; op_table applies it to captured run evidence.
"""
import math
from pathlib import Path

import yaml

from silk import settings

DEFAULT_THRESHOLDS_PATH = Path(__file__).with_name('data') / 'region-thresholds.default.yaml'


class RegionThresholdsError(ValueError):
    """An unreadable or malformed region policy, with its file and key."""


def load_thresholds_file(path):
    """Load a version-1 YAML policy into a mapping of PDK names to sets.

    Each supplied set must be complete. Unknown/custom PDK names are allowed;
    a report with no PDK identity uses the explicit ``default`` entry.
    """
    path = Path(path).expanduser()

    def fail(key, detail):
        raise RegionThresholdsError(f'{path}: {key}: {detail}')

    # SafeLoader defers collection construction until after construct_object
    # returns. Retain each node's first key path for that later validation.
    node_paths = {}

    class UniqueLoader(yaml.SafeLoader):
        def construct_object(self, node, deep=False):
            try:
                return super().construct_object(node, deep=deep)
            except (ValueError, KeyError, IndexError, AttributeError, OverflowError) as exc:
                # SafeLoader's explicit scalar tags can leak builtin errors
                # (invalid !!float/!!bool/!!timestamp, or an empty !!int).
                # Normalize at this boundary so doctor gets a typed diagnostic.
                if isinstance(exc, RegionThresholdsError):
                    raise
                raise yaml.constructor.ConstructorError(
                    None, None, str(exc), node.start_mark) from exc

        def construct_mapping(self, node, deep=False):
            prefix = node_paths.get(node, '<root>')
            if not isinstance(node, yaml.MappingNode):
                fail(prefix, f'expected a mapping node, got {node.id}')
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    fail('<key>', 'mapping keys must be strings')
                key_path = key if prefix == '<root>' else f'{prefix}.{key}'
                if key in result:
                    fail(key_path, 'duplicate key')
                node_paths.setdefault(value_node, key_path)
                try:
                    result[key] = self.construct_object(value_node, deep=deep)
                except yaml.YAMLError as exc:
                    fail(key_path, f'invalid YAML value: {exc}')
            return result

    try:
        with path.open(encoding='utf-8') as stream:
            data = yaml.load(stream, Loader=UniqueLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        fail('<root>', f'cannot read policy: {exc}')
    if not isinstance(data, dict):
        fail('<root>', 'expected a mapping')
    for key in data:
        if key not in ('version', 'pdks'):
            fail(key, 'unknown key')
    if type(data.get('version')) is not int or data['version'] != 1:
        fail('version', 'expected integer 1')
    entries = data.get('pdks')
    if not isinstance(entries, dict):
        fail('pdks', 'expected a mapping of PDK names to threshold sets')
    common = ('min_current_a', 'weak_sat_margin_v')
    margins = ('subthreshold_margin_kt_q', 'subthreshold_margin_v')
    for name, entry in entries.items():
        prefix = f'pdks.{name}'
        if not name.strip():
            fail(prefix, 'PDK name must not be empty')
        if not isinstance(entry, dict):
            fail(prefix, 'expected a threshold mapping')
        for key in entry:
            if key not in (*common, *margins):
                fail(f'{prefix}.{key}', 'unknown threshold')
        for key in common:
            if key not in entry:
                fail(f'{prefix}.{key}', 'required threshold is missing')
        if sum(key in entry for key in margins) != 1:
            fail(f'{prefix}.subthreshold_margin_kt_q/subthreshold_margin_v',
                 'supply exactly one margin: a kT/q multiplier or fixed volts')
        for key, value in entry.items():
            try:
                valid = (type(value) in (int, float)
                         and math.isfinite(value) and value >= 0)
            except OverflowError:
                valid = False
            if not valid:
                fail(f'{prefix}.{key}', 'expected a finite, nonnegative number')
    return entries


def shipped_default():
    """The explicit nominal policy for standalone threshold objects."""
    entries = load_thresholds_file(DEFAULT_THRESHOLDS_PATH)
    if 'default' not in entries:
        raise RegionThresholdsError(f'{DEFAULT_THRESHOLDS_PATH}: pdks.default: missing set')
    return entries['default']


def resolve_policy(pdk_name=None):
    """User set for this PDK, then its shipped set; never guess a known PDK."""
    name = pdk_name or 'default'
    if name != 'default':
        from silk.pdk import registry
        try:
            name = registry.manifest(name)['name']
        except registry.PDKRegistryError:
            # Custom PDKs can be supplied in a customer policy file.
            pass
    user_path = settings.get('SILK_REGION_THRESHOLDS')
    if user_path:
        entries = load_thresholds_file(user_path)
        if name in entries:
            return entries[name]
    entries = load_thresholds_file(DEFAULT_THRESHOLDS_PATH)
    if name not in entries:
        raise RegionThresholdsError(
            f'{DEFAULT_THRESHOLDS_PATH}: pdks.{name}: no threshold set; '
            'supply this PDK in SILK_REGION_THRESHOLDS or pass region_thresholds=')
    return entries[name]
