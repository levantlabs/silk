from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ViolationLocation(ABC):
    """Base class for the tagged-union location a violation points at.

    A subclass names one kind of design element: a net
    (:class:`NetLocation`), a port (:class:`PortLocation`), an instance
    (:class:`InstanceLocation`), a device terminal (:class:`TerminalLocation`)
    or a whole module (:class:`ModuleLocation`). Waiver matchers, report
    serialisation and report diffs all key on the concrete subclass.
    """
    @abstractmethod
    def describe(self) -> str:
        """Return a short ``kind:identifier`` label for the location.

        Returns
        -------
        str
            The label, for example ``'net:vout'``. Each subclass documents
            its own format; reports and diffs use it as the display and
            identity string.
        """


@dataclass
class NetLocation(ViolationLocation):
    """A violation on a net, optionally naming the ports tied to it.

    Attributes
    ----------
    net_name : str
        Name of the net.
    port_names : list of str
        Names of the ports the finding involves. Empty by default.
    """
    net_name: str
    port_names: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Return ``'net:<net_name>'``.

        Returns
        -------
        str
            The label; ``port_names`` is not part of it.
        """
        return f"net:{self.net_name}"


@dataclass
class PortLocation(ViolationLocation):
    """A violation on a module port.

    Attributes
    ----------
    port_name : str
        Name of the port.
    module_name : str
        Name of the module that owns the port. Empty by default, meaning the
        module is not recorded.
    """
    port_name: str
    module_name: str = ""

    def describe(self) -> str:
        """Return ``'port:<module_name>.<port_name>'``.

        Returns
        -------
        str
            The label. When ``module_name`` is empty the module prefix and
            its dot are omitted: ``'port:<port_name>'``.
        """
        prefix = f"{self.module_name}." if self.module_name else ""
        return f"port:{prefix}{self.port_name}"


@dataclass
class InstanceLocation(ViolationLocation):
    """A violation on one instance (device or submodule).

    Attributes
    ----------
    instance_path : str
        Dot-separated hierarchical instance path, for example
        ``'ldo.sram_macro'``. Scope exclusions match against this string.
    """
    instance_path: str

    def describe(self) -> str:
        """Return ``'instance:<instance_path>'``.

        Returns
        -------
        str
            The label.
        """
        return f"instance:{self.instance_path}"


@dataclass
class TerminalLocation(ViolationLocation):
    """A violation on one terminal of an instance, or on a terminal pair.

    Attributes
    ----------
    instance_path : str
        Dot-separated hierarchical path of the instance.
    terminal : str
        Name the finding is reported against. The junction-stress rule
        (STR-001) puts the rated pair name here, for example ``'Vgs'``.
    terminal_pair : tuple of str
        The two terminals of the rated pair, lower-case (``('g', 's')`` for
        ``'Vgs'`` as STR-001 reports it). Empty by default.
    """
    instance_path: str
    terminal: str
    terminal_pair: tuple[str, ...] = ()

    def describe(self) -> str:
        """Return ``'terminal:<instance_path>.<terminal>'``.

        Returns
        -------
        str
            The label; ``terminal_pair`` is not part of it.
        """
        return f"terminal:{self.instance_path}.{self.terminal}"


@dataclass
class ModuleLocation(ViolationLocation):
    """A violation on a module as a whole.

    Attributes
    ----------
    module_name : str
        Name of the module.
    """
    module_name: str

    def describe(self) -> str:
        """Return ``'module:<module_name>'``.

        Returns
        -------
        str
            The label.
        """
        return f"module:{self.module_name}"
