"""Reject the nonportable node name gnd in silk-generated Xyce decks."""
from silk.circuit.emitted_names import emitted_submodule_name
from silk.spice_names import designer_component
import re

from silk.circuit.ports import ElementType
from silk.exceptions import CircuitError


def _check(names, prefix=''):
    for name in names:
        if str(name).casefold() == 'gnd':
            raise CircuitError(
                f"Xyce net {designer_component(prefix) + str(name)!r} is named gnd: Xyce does not "
                "alias it to ground and can silently produce wrong results. "
                "Use '0' for global ground, or rename it to a 'vss' port "
                "and wire that port explicitly in the parent.")


def check_inventory(output_vars, elem_dict, module_dict):
    """Check even unsaved/unused nets; return definition-to-instance paths."""
    paths = {}
    _check(output_vars or ())

    def walk(elements, definitions, prefix, ancestors):
        for name, entry in (elements or {}).items():
            if entry.get('type') != ElementType.MODULE:
                continue
            instance = entry['device']
            # Port bindings belong to the parent scope.
            _check((pin.net for pin in (instance.pin_list or {}).values()), prefix)
            path = prefix + emitted_submodule_name(name) + '.'
            definition = (definitions or {}).get(instance.module_name, {}).get(
                'device', instance)
            paths.setdefault(instance.module_name.casefold(), path)
            for block in (instance, definition):
                _check(block.net_dict, path)
                if id(block) not in ancestors:
                    walk(block.elem_dict, block.module_dict, path,
                         ancestors | {id(block)})

    walk(elem_dict, module_dict, '', set())
    return paths


def check_emitted_nodes(deck, paths):
    """Check terminal fields after fault realization, before disk writes.

    The inventory cannot see fault-added elements or inline custom cards.
    Inspect the element families silk emits, preserving comments, parameter
    values and model names. External include files are not parsed here.
    """
    # Keep braced expressions and quoted values intact, including whitespace.
    token_pattern = r'\{[^}]*\}|"[^"]*"|\S+'
    counts = dict.fromkeys('RCLVIDBFH', 2)
    counts.update(dict.fromkeys('MEGTS', 4))
    counts.update({'J': 3, 'Z': 3})
    scope = ''
    lines = []
    for raw in deck.splitlines()[1:]:  # The first line is the SPICE title.
        line = raw.strip()
        if not line or line.startswith('*'):
            continue
        if line.startswith('+') and lines:
            lines[-1] += ' ' + line[1:]
        else:
            lines.append(line)
    for line in lines:
        line = line.split(';', 1)[0]
        fields = re.findall(token_pattern, line)
        if not fields:
            continue
        command = fields[0].lower()
        if command == '.subckt':
            scope = fields[1].casefold()
            _check(fields[2:], paths.get(scope, scope + '.'))
            continue
        if command == '.ends':
            scope = ''
            continue
        prefix = paths.get(scope, scope + '.') if scope else ''
        kind = fields[0][0].upper()
        if command == '.global':
            _check(fields[1:])
        elif kind == 'X':
            # Last positional field is the subcircuit name; later fields
            # are named parameters (or the optional PARAMS: separator).
            end = next((i for i, field in enumerate(fields[1:], 1)
                        if '=' in field or field.lower() == 'params:'), len(fields))
            _check(fields[1:end - 1], prefix)
        elif kind == 'Q':
            # Q: c b e [substrate] model [area] [OFF] [named params].
            end = next((i for i, field in enumerate(fields[1:], 1)
                        if '=' in field), len(fields))
            positional = fields[1:end]
            _check(positional[:3], prefix)
            while len(positional) > 4 and (positional[-1].lower() == 'off'
                                  or positional[-1].startswith('{')
                                  or re.fullmatch(
                                      r'[+\-]?(?:\d+(?:\.\d*)?|\.\d+)'
                                      r'(?:[eE][+\-]?\d+)?[a-zA-Z]*', positional[-1])):
                positional.pop()
            _check(positional[:-1], prefix)
        elif kind in counts:
            poly = re.search(r'\bpoly\s*\(\s*(\d+)\s*\)', line, re.I)
            if kind in 'EG' and poly:
                _check(fields[1:3], prefix)
                controls = re.findall(token_pattern, line[poly.end():])
                _check(controls[:2 * int(poly.group(1))], prefix)
            else:
                _check(fields[1:1 + counts[kind]], prefix)
        if not command.startswith('.'):
            # Voltage references in behavioral source expressions also name
            # electrical nodes, even without an ordinary terminal binding.
            for expression in re.findall(r'\{[^}]*\}', line):
                for reference in re.findall(r'\bv\s*\(([^()]*)\)', expression, re.I):
                    _check((name.strip() for name in reference.split(',')), prefix)
