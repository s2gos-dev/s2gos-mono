from abc import ABC, abstractmethod
from typing import List, Optional

import xarray as xr
from s2gos_utils.scene import SceneDescription
from s2gos_utils.typing import PathLike


class SimulationBackend(ABC):
    """Abstract base class for radiative transfer simulation backends.

    This class provides a generic interface for different radiative transfer models.
    """

    def __init__(self, simulation_config):
        """Initialize backend with simulation configuration.

        Args:
            simulation_config: Complete simulation configuration (sensors, illumination, etc.)
        """
        self.simulation_config = simulation_config

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are available.

        Returns:
            True if backend can be used, False otherwise
        """
        pass

    @abstractmethod
    def run_simulation(
        self,
        scene_description: SceneDescription,
        scene_dir: PathLike,
        output_dir: Optional[PathLike] = None,
        **kwargs,
    ) -> xr.Dataset:
        """Run complete simulation pipeline.

        This is the main entry point for running simulations. Implementations
        should handle all aspects of the simulation internally:
        - Translation of scene description to backend-specific format
        - Scene setup and surface creation
        - Backend-specific configuration
        - Simulation execution
        - Result processing and standardization

        Args:
            scene_description: Scene description from s2gos_generator
            scene_dir: Directory containing scene assets (meshes, textures)
            output_dir: Directory for simulation outputs (optional)
            **kwargs: Additional backend-specific args

        Returns:
            xarray.Dataset containing simulation results following S2GOS standards
        """
        pass

    def validate_configuration(self) -> List[str]:
        """Validate simulation configuration for this backend.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Generic validation - can be overridden by specific backends
        if not self.simulation_config.sensors and not getattr(
            self.simulation_config, "measurements", []
        ):
            errors.append("No sensors or measurements defined in configuration")

        # Check if backend supports all requested measurement types
        unsupported = [
            m.type
            for m in getattr(self.simulation_config, "measurements", [])
            if m.type not in self.supported_measurements
        ]
        if unsupported:
            errors.append(f"Unsupported measurement types: {unsupported}")

        return errors

    def validate_scene(
        self, scene_description: SceneDescription, scene_dir: PathLike
    ) -> List[str]:
        """Validate scene description and assets.

        Optional method for backends to check scene validity before simulation.
        Default implementation returns no errors.

        Args:
            scene_description: Scene description to validate
            scene_dir: Directory containing scene assets

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    @property
    def name(self) -> str:
        """Backend name for identification."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def supported_platforms(self) -> List[str]:
        """List of supported observation platforms.

        Returns:
            List of platform names this backend supports
        """
        ...

    @property
    @abstractmethod
    def supported_measurements(self) -> List[str]:
        """List of supported measurement types.

        Returns:
            List of measurement type names this backend supports
        """
        ...

    def __str__(self) -> str:
        return f"{self.name}(available={self.is_available()})"
