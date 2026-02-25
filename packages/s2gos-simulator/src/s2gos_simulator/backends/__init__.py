try:
    from .eradiate.backend import EradiateBackend as EradiateSimulator

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False
    EradiateSimulator = None

__all__ = ["EradiateSimulator", "ERADIATE_AVAILABLE"]
