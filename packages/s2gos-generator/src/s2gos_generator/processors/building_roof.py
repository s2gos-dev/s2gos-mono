"""Hip-roof construction on top of a 2D straight skeleton.

Pitched roofs reuse a single straight skeleton of the wall footprint
(see `straight_skeleton.py`). Every face is lifted
to 3D as

    z = clamp(eaves_z + t * tan(pitch), apex_z)

and triangulated with earcut, so every wall slopes up to the skeleton ridge.

All meshes are returned in world coordinates (z is absolute).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import mapbox_earcut as earcut
import numpy as np
import trimesh
from shapely.geometry import Polygon

from .straight_skeleton import Skeleton


@dataclass
class _SkelFaces:
    """Per-edge straight-skeleton faces in the (nodes, faces) form the roof
    builder consumes: ``nodes`` is an (N, 3) array of (x, y, t) where ``t`` is the
    node's height above the eaves (its sweep distance from the footprint
    boundary), and ``faces`` lists CCW node-index loops, one per footprint edge."""

    nodes: np.ndarray
    faces: list[list[int]]


def _skeleton_faces(poly: Polygon) -> Optional[_SkelFaces]:
    """Build the straight skeleton and return its native per-edge roof faces.

    The skeleton emits ``nodes`` (x, y, t) and one CCW face per polygon edge
    directly (see ``straight_skeleton.Skeleton``); the node height ``t`` is the
    sweep distance, exactly what the roof lift needs.
    """
    if poly is None or poly.is_empty:
        return None

    try:
        sk = Skeleton(poly)
        sk.compute()
    except Exception as exc:
        logging.debug("Skeleton computation failed: %s", exc)
        return None

    if not sk.faces:
        return None

    return _SkelFaces(nodes=sk.nodes, faces=sk.faces)


def _orient_normals_up(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    flip = normals[:, 2] < 0
    out = faces.copy()
    out[flip] = faces[flip][:, [0, 2, 1]]
    return out


def _validate_and_clean(mesh: Optional[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if mesh is None or len(mesh.faces) == 0:
        return None
    try:
        mesh.merge_vertices()
        mask = mesh.area_faces > 1e-9
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
    except Exception:
        return None
    if len(mesh.faces) == 0 or not np.all(np.isfinite(mesh.vertices)):
        return None
    return mesh


def build_hip_roof(
    poly: Polygon,
    eaves_z: float,
    apex_z: float,
    pitch_deg: float,
) -> Optional[trimesh.Trimesh]:
    """Hipped roof: every wall slopes up to the skeleton ridge."""
    if poly is None or poly.is_empty:
        return None
    tan_pitch = math.tan(math.radians(pitch_deg))
    if tan_pitch <= 0:
        return None

    sk = _skeleton_faces(poly)
    if sk is None or not sk.faces:
        return None

    z = np.minimum(apex_z, eaves_z + sk.nodes[:, 2] * tan_pitch)
    verts3 = np.column_stack([sk.nodes[:, 0], sk.nodes[:, 1], z])

    tri_faces: list[list[int]] = []
    for face in sk.faces:
        if len(face) < 3:
            continue
        coords_2d = np.ascontiguousarray(sk.nodes[face, :2], dtype=np.float64)
        rings = np.array([len(face)], dtype=np.uint32)
        indices = earcut.triangulate_float64(coords_2d, rings)
        if len(indices) == 0:
            continue
        face_arr = np.asarray(face, dtype=np.int64)
        tri_faces.extend(face_arr[np.asarray(indices).reshape(-1, 3)].tolist())

    if not tri_faces:
        return None

    faces_arr = _orient_normals_up(verts3, np.array(tri_faces))
    mesh = trimesh.Trimesh(vertices=verts3, faces=faces_arr, process=False)
    return _validate_and_clean(mesh)


def compute_pitched_geometry(
    total_height: float,
    pitch_deg: float,
    target_roof_height: float,
) -> Optional[dict]:
    tan_pitch = math.tan(math.radians(pitch_deg))
    if tan_pitch <= 0:
        return None

    roof_h = min(target_roof_height, total_height * 0.5)
    if roof_h <= 0:
        return None

    return {
        "eaves_z_offset": total_height - roof_h,
        "apex_z_offset": total_height,
        "pitch_deg": pitch_deg,
    }
