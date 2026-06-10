"""Tests for the straight-skeleton-based roof construction."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from s2gos_generator.assets.building_roof import (
    _skeleton_faces,
    build_hip_roof,
    compute_pitched_geometry,
)


def _rotate(coords, angle_deg):
    rad = math.radians(angle_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    return [(x * cos - y * sin, x * sin + y * cos) for x, y in coords]


def _faces_have_no_crossings(sk) -> bool:
    """Every pair of distinct faces must have only their shared edge in common
    -- never crossing geometry. Tests intersection of face boundaries pairwise."""
    from shapely.geometry import Polygon as _P

    polys = [_P(sk.nodes[f, :2]).buffer(0) for f in sk.faces]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            inter = polys[i].intersection(polys[j])
            # Faces share edges only -- intersection should have ~zero area.
            if inter.area > 1e-6 * polys[i].area:
                return False
    return True


def _xy_face_area_sum(skeleton) -> float:
    total = 0.0
    for f in skeleton.faces:
        coords = skeleton.nodes[f, :2]
        n = len(coords)
        a = 0.0
        for i in range(n):
            x0, y0 = coords[i]
            x1, y1 = coords[(i + 1) % n]
            a += x0 * y1 - x1 * y0
        total += abs(a) * 0.5
    return total


@pytest.mark.parametrize(
    "name, coords, expected_faces",
    [
        ("square", [(0, 0), (4, 0), (4, 4), (0, 4)], 4),
        ("rect_long", [(0, 0), (6, 0), (6, 2), (0, 2)], 4),
        ("triangle", [(0, 0), (3, 0), (1.5, 2.5)], 3),
        (
            "L",
            [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)],
            6,
        ),
        (
            "T",
            [(0, 0), (6, 0), (6, 2), (4, 2), (4, 6), (2, 6), (2, 2), (0, 2)],
            8,
        ),
        ("pentagon", [(0, 0), (5, 0), (6, 3), (3, 5), (-1, 3)], 5),
        (
            "hexagon",
            [(1, 0), (3, 0), (4, 1.732), (3, 3.464), (1, 3.464), (0, 1.732)],
            6,
        ),
    ],
)
def test_skeleton_coverage(name, coords, expected_faces):
    poly = Polygon(coords)
    sk = _skeleton_faces(poly)
    assert sk is not None, f"{name}: skeleton failed"
    assert len(sk.faces) == expected_faces, f"{name}: face count"
    area = _xy_face_area_sum(sk)
    assert abs(area - poly.area) / poly.area < 0.005, (
        f"{name}: area {area} vs {poly.area}"
    )


@pytest.mark.parametrize(
    "name, coords",
    [
        ("rect_long", [(0, 0), (6, 0), (6, 2), (0, 2)]),
        ("L", [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]),
        ("T", [(0, 0), (6, 0), (6, 2), (4, 2), (4, 6), (2, 6), (2, 2), (0, 2)]),
    ],
)
@pytest.mark.parametrize("angle_deg", [0.0, 17.0, 30.0, 45.0, 73.0])
def test_skeleton_robust_under_rotation(name, coords, angle_deg):
    """Skeleton must succeed on all rotations of common OSM-style footprints."""
    poly = Polygon(_rotate(coords, angle_deg))
    sk = _skeleton_faces(poly)
    assert sk is not None, f"{name} @ {angle_deg}°: skeleton failed"
    err = abs(_xy_face_area_sum(sk) - poly.area) / poly.area
    assert err < 0.005, f"{name} @ {angle_deg}°: area err {err:.4%}"

    crossings_ok = _faces_have_no_crossings(sk)
    if not crossings_ok and name in {"U", "plus"} and angle_deg == 30.0:
        pytest.xfail(
            f"{name} @ {angle_deg}°: known skeleton crossing; flat-roof fallback applies"
        )
    assert crossings_ok, f"{name} @ {angle_deg}°: crossing faces"


def test_skeleton_pathological_returns_none_or_valid():
    # Degenerate inputs should never raise. Either return a valid skeleton
    # or None — both are acceptable for the caller's flat fallback.
    sliver = Polygon([(0, 0), (10, 0), (10, 0.001), (0, 0.001)])
    sk = _skeleton_faces(sliver)
    if sk is not None:
        assert _xy_face_area_sum(sk) > 0

    near_collinear = Polygon([(0, 0), (4, 0), (4.001, 2), (4, 4), (0, 4)])
    sk = _skeleton_faces(near_collinear)
    assert sk is None or _xy_face_area_sum(sk) > 0


def _projected_area(mesh) -> float:
    v = mesh.vertices
    total = 0.0
    for f in mesh.faces:
        x0, y0 = v[f[0], 0], v[f[0], 1]
        x1, y1 = v[f[1], 0], v[f[1], 1]
        x2, y2 = v[f[2], 0], v[f[2], 1]
        total += abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * 0.5
    return total


@pytest.mark.parametrize(
    "name, coords",
    [
        ("square", [(0, 0), (4, 0), (4, 4), (0, 4)]),
        ("rect", [(0, 0), (6, 0), (6, 2), (0, 2)]),
        ("L", [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]),
        (
            "T",
            [(0, 0), (6, 0), (6, 2), (4, 2), (4, 6), (2, 6), (2, 2), (0, 2)],
        ),
    ],
)
def test_hip_roof_covers_footprint(name, coords):
    poly = Polygon(coords)
    eaves_z, apex_z = 10.0, 12.0
    mesh = build_hip_roof(poly, eaves_z=eaves_z, apex_z=apex_z, pitch_deg=45.0)
    assert mesh is not None, f"{name}: build_hip_roof returned None"
    err = abs(_projected_area(mesh) - poly.area) / poly.area
    assert err < 0.01, f"{name}: roof coverage error {err:.3%}"
    # Z range is bounded.
    assert mesh.vertices[:, 2].min() >= eaves_z - 1e-6
    assert mesh.vertices[:, 2].max() <= apex_z + 1e-6


def test_hip_roof_apex_clamped_to_apex_z():
    # Wide square: 45° pitch would naturally rise above apex_z; clamp must hold.
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    eaves_z, apex_z = 0.0, 1.0  # very low apex relative to inradius 5
    mesh = build_hip_roof(poly, eaves_z=eaves_z, apex_z=apex_z, pitch_deg=45.0)
    assert mesh is not None
    assert mesh.vertices[:, 2].max() <= apex_z + 1e-6


def test_hip_roof_invalid_inputs():
    assert build_hip_roof(None, 0.0, 1.0, 30.0) is None
    assert build_hip_roof(Polygon(), 0.0, 1.0, 30.0) is None
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert build_hip_roof(poly, 0.0, 1.0, 0.0) is None  # zero pitch


def test_compute_pitched_geometry_returns_hip():
    info = compute_pitched_geometry(
        total_height=10.0,
        pitch_deg=30.0,
        target_roof_height=2.0,
    )
    assert info is not None
    assert math.isclose(info["pitch_deg"], 30.0)
