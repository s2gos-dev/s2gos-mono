import numpy as np
import pytest

from s2gos_simulator.backends.eradiate.surface_builder import uv_window_transform


@pytest.fixture(scope="module")
def mi():
    """Mitsuba with a variant set, which its scalar types need before construction."""
    import mitsuba

    if mitsuba.variant() is None:
        mitsuba.set_variant("scalar_rgb")
    return mitsuba


@pytest.mark.parametrize(
    "area_config",
    [
        {},
        {"texture_uv_window": None},
        {"texture_uv_window": [0.0, 0.0, 1.0, 1.0]},
    ],
    ids=["no-window", "null-window", "identity-window"],
)
def test_no_transform_when_the_texture_is_the_aoi(area_config):
    """The bitmap then gets no ``to_uv`` at all, which is the usual case."""
    assert uv_window_transform(area_config) is None


def test_a_real_window_maps_the_unit_square_onto_it(mi):
    """The transform must be 4x4. Mitsuba accepts a 3x3 without error, then silently
    drops its translation, sliding every windowed texture off its geometry."""
    u0, v0, u1, v1 = 0.25, 0.5, 0.75, 1.0

    transform = uv_window_transform({"texture_uv_window": [u0, v0, u1, v1]})

    assert np.array(transform.matrix).shape == (4, 4)
    origin = np.array(transform @ mi.ScalarPoint3f(0.0, 0.0, 0.0))
    far = np.array(transform @ mi.ScalarPoint3f(1.0, 1.0, 0.0))
    assert origin[:2] == pytest.approx([u0, v0])
    assert far[:2] == pytest.approx([u1, v1])
