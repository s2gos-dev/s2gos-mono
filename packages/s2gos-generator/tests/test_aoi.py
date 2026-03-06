"""Tests for lazy AOI polygon properties on SceneResourceContext."""

from unittest.mock import MagicMock


def _make_context(
    center_lat: float = 45.0,
    center_lon: float = 15.0,
    aoi_size_km: float = 10.0,
    buffer_size_km: float = None,
    background_size_km: float = None,
):
    """Build a SceneResourceContext with lazy AOI properties, bypassing the full constructor."""
    from s2gos_generator.core.context import SceneResourceContext

    ctx = object.__new__(SceneResourceContext)
    ctx.center_lat = center_lat
    ctx.center_lon = center_lon
    ctx.aoi_size_km = aoi_size_km
    ctx._target_aoi_polygon = None
    ctx._buffer_aoi_polygon = None
    ctx._background_aoi_polygon = None
    ctx._coord_system = None

    config = MagicMock()
    if buffer_size_km is not None:
        config.buffer.size_km = buffer_size_km
    else:
        config.buffer = None
    if background_size_km is not None:
        config.background.size_km = background_size_km
    else:
        config.background = None
    ctx.config = config

    return ctx


class TestTargetAOIPolygon:
    def test_returns_valid_polygon(self):
        ctx = _make_context()
        polygon = ctx.target_aoi_polygon
        assert polygon is not None
        assert polygon.is_valid

    def test_returns_same_object_on_second_access(self):
        ctx = _make_context()
        p1 = ctx.target_aoi_polygon
        p2 = ctx.target_aoi_polygon
        assert p1 is p2


class TestBufferAOIPolygon:
    def test_returns_none_when_no_buffer_configured(self):
        ctx = _make_context(buffer_size_km=None)
        assert ctx.buffer_aoi_polygon is None

    def test_returns_polygon_when_buffer_configured(self):
        ctx = _make_context(buffer_size_km=20.0)
        polygon = ctx.buffer_aoi_polygon
        assert polygon is not None
        assert polygon.is_valid

    def test_buffer_larger_than_target(self):
        ctx = _make_context(aoi_size_km=10.0, buffer_size_km=20.0)
        assert ctx.buffer_aoi_polygon.area > ctx.target_aoi_polygon.area

    def test_returns_same_object_on_second_access(self):
        ctx = _make_context(buffer_size_km=20.0)
        p1 = ctx.buffer_aoi_polygon
        p2 = ctx.buffer_aoi_polygon
        assert p1 is p2


class TestBackgroundAOIPolygon:
    def test_returns_none_when_no_background_configured(self):
        ctx = _make_context(background_size_km=None)
        assert ctx.background_aoi_polygon is None

    def test_returns_polygon_when_background_configured(self):
        ctx = _make_context(background_size_km=50.0)
        polygon = ctx.background_aoi_polygon
        assert polygon is not None
        assert polygon.is_valid

    def test_background_larger_than_buffer(self):
        ctx = _make_context(buffer_size_km=20.0, background_size_km=50.0)
        assert ctx.background_aoi_polygon.area > ctx.buffer_aoi_polygon.area

    def test_returns_same_object_on_second_access(self):
        ctx = _make_context(background_size_km=50.0)
        p1 = ctx.background_aoi_polygon
        p2 = ctx.background_aoi_polygon
        assert p1 is p2
