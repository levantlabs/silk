"""Validation of Python names in file-backed device catalogs."""

from silk.utils import require_python_identifier


def _device_python_names(name, spec, path):
    """Yield names and diagnostic contexts in both device-catalog formats."""
    context = f"{path}: device {name!r}"
    yield name, f'{context}, device key'
    for field in ('pinList', 'pinAliases', 'paramList', 'paramAliases'):
        for key in spec.get(field) or ():
            yield key, f'{context}, {field} key'
    if spec.get('source_class') is not None:
        yield spec['source_class'], f'{context}, source_class'
    for param in spec.get('paramConstraints') or ():
        yield param['name'], f'{context}, paramConstraints name'


def _device_name_errors(name, spec, path):
    """Yield identifier errors; callers choose fail-fast or full-file reporting."""
    for value, context in _device_python_names(name, spec, path):
        try:
            require_python_identifier(value, context)
        except ValueError as exc:
            yield exc


def _validate_device_names(name, spec, path, error_type):
    """Validate both PDK and ideal-element spellings before creating classes."""
    error = next(_device_name_errors(name, spec, path), None)
    if error is not None:
        raise error_type(str(error)) from error
