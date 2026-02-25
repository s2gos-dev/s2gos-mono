from typing import Dict, List, Optional, Type

from .base import SimulationBackend


class BackendRegistry:
    """Registry for managing available simulation backends."""

    def __init__(self):
        self._backends: Dict[str, Type[SimulationBackend]] = {}

    def register(self, name: str, backend_class: Type[SimulationBackend]) -> None:
        """Register a backend class.

        Args:
            name: Backend identifier
            backend_class: Backend class implementing SimulationBackend
        """
        if not issubclass(backend_class, SimulationBackend):
            raise TypeError(
                f"Backend {backend_class} must inherit from SimulationBackend"
            )

        self._backends[name] = backend_class

    def get_backend(self, name: str, render_config) -> Optional[SimulationBackend]:
        """Get a backend instance by name.

        Args:
            name: Backend identifier
            render_config: Configuration for backend initialization

        Returns:
            Backend instance if available, None otherwise
        """
        if name not in self._backends:
            return None

        backend_class = self._backends[name]
        try:
            backend = backend_class(render_config)
            if backend.is_available():
                return backend
        except Exception:
            pass

        return None

    def list_available(self) -> List[str]:
        """List names of available backends.

        Returns:
            List of backend names that can be instantiated
        """
        available = []
        for name, backend_class in self._backends.items():
            try:
                # Try to instantiate with minimal config to test availability
                backend = backend_class(None)
                if backend.is_available():
                    available.append(name)
            except Exception:
                pass

        return available

    def list_all(self) -> List[str]:
        """List all registered backend names.

        Returns:
            List of all registered backend names (available and unavailable)
        """
        return list(self._backends.keys())

    def get_default_backend(self, render_config) -> Optional[SimulationBackend]:
        """Get the first available backend.

        Args:
            render_config: Configuration for backend initialization

        Returns:
            First available backend instance, None if none available
        """
        for name in self.list_available():
            backend = self.get_backend(name, render_config)
            if backend:
                return backend

        return None


# Global registry instance
_registry = BackendRegistry()


def register_backend(name: str, backend_class: Type[SimulationBackend]) -> None:
    """Register a backend class globally.

    Args:
        name: Backend identifier
        backend_class: Backend class implementing SimulationBackend
    """
    _registry.register(name, backend_class)


def get_backend(name: str, render_config) -> Optional[SimulationBackend]:
    """Get a backend instance by name.

    Args:
        name: Backend identifier
        render_config: Configuration for backend initialization

    Returns:
        Backend instance if available, None otherwise
    """
    return _registry.get_backend(name, render_config)


def list_available_backends() -> List[str]:
    """List names of available backends.

    Returns:
        List of backend names that can be instantiated
    """
    return _registry.list_available()


def list_all_backends() -> List[str]:
    """List all registered backend names.

    Returns:
        List of all registered backend names (available and unavailable)
    """
    return _registry.list_all()


def get_default_backend(render_config) -> Optional[SimulationBackend]:
    """Get the first available backend.

    Args:
        render_config: Configuration for backend initialization

    Returns:
        First available backend instance, None if none available
    """
    return _registry.get_default_backend(render_config)
