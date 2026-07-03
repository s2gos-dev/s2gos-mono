"""
Straight skeleton algorithm.

Implements Felkel & Obdrzalek's algorithm with some deviations.
Edges move inward at unit speed and the priority queue resolves edge events (an edge
shrinks to zero) and split events (a reflex vertex runs into an opposite edge, possibly
merging two boundary loops).
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Optional, Union

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

# A 2D point in the working plane.
Point = tuple[float, float]
# A skeleton arc as an ordered pair of endpoints.
Arc = tuple[Point, Point]


class Edge:
    """An oriented polygon edge e_i."""

    def __init__(self, p1: Point, p2: Point) -> None:
        self.p1: Point = p1
        self.p2: Point = p2
        self.ground_start_node: Optional[int] = None
        self.ground_end_node: Optional[int] = None

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length == 0:
            self.dir_x, self.dir_y = 0.0, 0.0
            self.n_x, self.n_y = 0.0, 0.0
        else:
            self.dir_x, self.dir_y = dx / length, dy / length
            self.n_x, self.n_y = -self.dir_y, self.dir_x  # Inward normal


class VertexNode:
    """One node V_i in a LAV (see paper)."""

    # Monotonic id giving every node a stable, address-independent ordering so that
    # iteration over the active set is deterministic (the algorithm is otherwise
    # sensitive to tie-break order among coincident candidates).
    _seq = itertools.count()

    def __init__(
        self,
        edge_in: Edge,
        edge_out: Edge,
        point: Point,
        tol_angle: float,
        t0: float = 0.0,
    ) -> None:
        self.seq: int = next(VertexNode._seq)
        self.edge_in: Edge = edge_in
        self.edge_out: Edge = edge_out
        self.v0: Point = point
        self.t0: float = t0
        self.prev_node: Optional[VertexNode] = None
        self.next_node: Optional[VertexNode] = None
        self.processed: bool = False

        cross_dir = edge_in.dir_x * edge_out.dir_y - edge_in.dir_y * edge_out.dir_x
        self.is_reflex: bool = cross_dir < -tol_angle

        # Vertex velocity v ensures the node stays on both incident edges as they sweep
        # inward at unit speed (v . n == 1). Solved as: v = (n_in + n_out) / (1 + n_in . n_out).
        # For anti-parallel edges (a collapsed "needle"), the denominator drops to zero;
        # these lack a finite velocity and are resolved structurally by the event queue later.
        n_in = (edge_in.n_x, edge_in.n_y)
        n_out = (edge_out.n_x, edge_out.n_y)
        denom = 1.0 + (n_in[0] * n_out[0] + n_in[1] * n_out[1])

        if denom < tol_angle:
            # Needle vertex: keep a unit placeholder (the shared edge tangent) so
            # _in_wedge / plotting never divide by zero; it never times an event.
            self.v: Point = (edge_out.dir_x, edge_out.dir_y)
        else:
            self.v = ((n_in[0] + n_out[0]) / denom, (n_in[1] + n_out[1]) / denom)

    def pos(self, t: float) -> Point:
        dt = t - self.t0
        return (self.v0[0] + dt * self.v[0], self.v0[1] + dt * self.v[1])


class IntersectionEvent:
    """One entry in the priority queue (paper §2.1 step 1c)."""

    _id_counter = itertools.count()

    def __init__(
        self,
        t: float,
        event_type: str,
        point: Point,
        v_a: VertexNode,
        v_b: Optional[VertexNode] = None,
        opp_edge: Optional[Edge] = None,
    ) -> None:
        self.t: float = t
        self.event_type: str = event_type
        self.I: Point = point
        self.v_a: VertexNode = v_a
        self.v_b: Optional[VertexNode] = v_b
        self.opp_edge: Optional[Edge] = opp_edge
        self.opp_Y: Optional[VertexNode] = None
        self.opp_X: Optional[VertexNode] = None
        self.two_node: bool = False
        self._auto_id: int = next(IntersectionEvent._id_counter)

    def __lt__(self, other: IntersectionEvent) -> bool:
        return (self.t, self._auto_id) < (other.t, other._auto_id)


def _dedupe_consecutive(seq: list[int]) -> list[int]:
    """Drop consecutive duplicate node indices and any closing repeat."""
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _remove_collinear(
    face: list[int], nodes: np.ndarray, tol: float = 1e-6
) -> list[int]:
    """Drop vertices nearly collinear with their neighbours (keeps >= 3)."""
    if len(face) <= 3:
        return face
    out = list(face)
    while len(out) > 3:
        n = len(out)
        deleted = False
        for i in range(n):
            p_prev = nodes[out[(i - 1) % n]]
            p_cur = nodes[out[i]]
            p_next = nodes[out[(i + 1) % n]]
            ax, ay = p_cur[0] - p_prev[0], p_cur[1] - p_prev[1]
            bx, by = p_next[0] - p_cur[0], p_next[1] - p_cur[1]
            mag_a = math.hypot(ax, ay)
            mag_b = math.hypot(bx, by)
            if mag_a < 1e-9 or mag_b < 1e-9:
                continue
            cross = (ax * by - ay * bx) / (mag_a * mag_b)
            dot = (ax * bx + ay * by) / (mag_a * mag_b)
            if abs(cross) < tol and dot > 0.999:
                del out[i]
                deleted = True
                break
        if not deleted:
            break
    return out


class Skeleton:
    def __init__(
        self,
        poly: Union[Polygon, MultiPolygon],
        tol_dist: float = 1e-8,
        tol_angle: float = 1e-19,
        tol_time: float = 0,
    ) -> None:
        self.poly: Union[Polygon, MultiPolygon] = poly
        self.tol_dist: float = tol_dist
        self.tol_angle: float = tol_angle
        self.tol_time: float = tol_time

        self.current_time: float = 0.0  # sweep distance / roof height
        self.rings: list[list[tuple[float, ...]]] = []
        self.active_nodes: set[VertexNode] = set()
        self.edges: list[Edge] = []
        self.pq: list[IntersectionEvent] = []
        self.arcs: list[Arc] = []

        self.nodes: np.ndarray = np.empty((0, 3))
        self.faces: list[list[int]] = []
        self._nodes: list[tuple[float, float, float]] = []
        self._node_index: dict[tuple[int, int], int] = {}
        self._node_eps: float = 1e-9
        self._arc_records: list[tuple[int, int, Edge, Edge]] = []
        self._edge_arcs: dict[Edge, list[int]] = {}

        # Coordinates are solved in a unit-span normalized frame so the absolute
        # tolerances behave the same regardless of input units;
        # results are mapped back at the end.
        self._scale: float = 1.0
        self._origin: Point = (0.0, 0.0)
        # Hard safety net: if the event loop fails to terminate (degenerate input),
        # bail and leave faces empty so the caller falls back to a flat roof.
        self.timed_out: bool = False
        # Per-run cap on event-loop iterations, set in _run_once.
        self._event_cap: int = 0

    def _node(self, pos: Point, t: float) -> int:
        """Return the index of the node at ``pos`` (deduped), creating it with
        height ``t`` (its birth time / sweep distance) on first sight."""
        eps = self._node_eps
        key = (round(pos[0] / eps), round(pos[1] / eps))
        idx = self._node_index.get(key)
        if idx is None:
            idx = len(self._nodes)
            self._node_index[key] = idx
            self._nodes.append((float(pos[0]), float(pos[1]), float(t)))
        return idx

    def _record_arc(self, u_idx: int, w_idx: int, edge_a: Edge, edge_b: Edge) -> None:
        """Record a skeleton arc bounded by ``edge_a`` and ``edge_b`` (the
        incident edges of the moving vertex whose trace this arc is)."""
        ai = len(self._arc_records)
        self._arc_records.append((u_idx, w_idx, edge_a, edge_b))
        self._edge_arcs.setdefault(edge_a, []).append(ai)
        if edge_b is not edge_a:
            self._edge_arcs.setdefault(edge_b, []).append(ai)

    def _push_event(self, event: Optional[IntersectionEvent]) -> None:
        if event is not None and event.t >= self.current_time - self.tol_time:
            heapq.heappush(self.pq, event)

    def _edge_event_valid(self, event: IntersectionEvent) -> bool:
        a, b = event.v_a, event.v_b
        return not a.processed and not b.processed and a.next_node is b

    def _split_event_valid(self, event: IntersectionEvent) -> bool:
        # Paper §2.2 step 2e: the opposite edge is searched in the SLAV *when the
        # split event is processed*, not when it is created. Between creation and
        # now the edge may have been split further (Fig. 5), so the live opposite
        # pair must be re-resolved against the current active vertices.
        v = event.v_a
        if v.processed:
            return False
        pair = self._find_split_pair(event.I, event.opp_edge)
        if pair is None:
            return False
        event.opp_Y, event.opp_X = pair
        return True

    def _find_split_pair(
        self, point: Point, ej: Edge
    ) -> Optional[tuple[VertexNode, VertexNode]]:
        """Paper §2.2 step 2e / Fig. 5: locate the currently-active pair
        (Y, X = Y.next_node) bounding the live sub-segment of the original edge
        ``ej`` whose wedge contains the candidate point ``point``.

        The reference to a split edge is stored in *every* LAV that shares it
        (the paper's "multiple hits" during the SLAV traversal), including across
        different loops — which is what lets an outer-boundary reflex vertex split
        onto a hole edge (Fig. 7). The wedge test makes the match unambiguous: for
        a fixed ``point`` and edge line, only one active sub-segment can contain it.
        """
        for Y in sorted(self.active_nodes, key=lambda n: n.seq):
            if Y.processed or Y.edge_out is not ej:
                continue
            X = Y.next_node
            if X.processed:
                continue
            if self._in_wedge(point, Y, X, ej):
                return Y, X
        return None

    def _clean_ring(
        self, raw_coords: list[tuple[float, ...]]
    ) -> list[tuple[float, ...]]:
        """Removes adjacent points that are virtually identical to avoid 0-length edge crashes."""
        clean = []
        for pt in raw_coords:
            if (
                not clean
                or math.hypot(pt[0] - clean[-1][0], pt[1] - clean[-1][1])
                > self.tol_dist
            ):
                clean.append(pt)
        # Check closing point
        if (
            len(clean) > 1
            and math.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1])
            <= self.tol_dist
        ):
            clean.pop()
        return clean

    def compute(self) -> list[Arc]:
        polys = self.poly.geoms if hasattr(self.poly, "geoms") else [self.poly]
        for p in polys:
            p_ccw = orient(p, sign=1.0)
            self.rings.append(self._clean_ring(list(p_ccw.exterior.coords)[:-1]))
            for hole in p_ccw.interiors:
                self.rings.append(self._clean_ring(list(hole.coords)[:-1]))

        minx, miny, maxx, maxy = self.poly.bounds
        span = max(maxx - minx, maxy - miny, 1.0)
        self._scale = 1.0 / span
        self._origin = (float(self.poly.centroid.x), float(self.poly.centroid.y))

        # Run once; on a non-terminating degenerate input retry with a tiny jitter
        # that breaks exact symmetry (sub-mm for real footprints). If every attempt
        # times out, leave faces empty so the caller falls back to a flat roof.
        for jitter in (0.0, 1e-6, 5e-6, 2e-5):
            seed = 0 if jitter == 0.0 else int(jitter * 1e9)
            self._run_once(jitter, seed)
            if not self.timed_out and self.faces:
                break
        return self.arcs

    def _reset_run_state(self) -> None:
        self.current_time = 0.0
        self.active_nodes = set()
        self.edges = []
        self.pq = []
        self.arcs = []
        self.nodes = np.empty((0, 3))
        self.faces = []
        self._nodes = []
        self._node_index = {}
        self._arc_records = []
        self._edge_arcs = {}
        self.timed_out = False

    def _run_once(self, jitter: float, seed: int) -> None:
        """One full skeleton pass in normalized coordinates (optionally jittered).
        Populates self.nodes, self.faces, self.arcs (de-normalized)."""
        self._reset_run_state()
        VertexNode._seq = itertools.count()
        self._node_eps = 1e-7
        cx, cy = self._origin
        rng = np.random.default_rng(seed) if jitter else None

        def _nrm(pt: tuple[float, ...]) -> Point:
            x = (pt[0] - cx) * self._scale
            y = (pt[1] - cy) * self._scale
            if rng is not None:
                x += float(rng.uniform(-jitter, jitter))
                y += float(rng.uniform(-jitter, jitter))
            return (x, y)

        n_vertices = 0
        # step 1a (all geometry built in normalized coordinates)
        for ring in self.rings:
            n = len(ring)
            if n < 3:
                continue
            n_vertices += n
            nring = [_nrm(pt) for pt in ring]

            ring_edges = []
            for i in range(n):
                e = Edge(nring[i], nring[(i + 1) % n])
                ring_edges.append(e)
                self.edges.append(e)

            ground = [self._node(nring[i], 0.0) for i in range(n)]
            for i in range(n):
                ring_edges[i].ground_start_node = ground[i]
                ring_edges[i].ground_end_node = ground[(i + 1) % n]

            # step 1b
            ring_nodes = []
            for i in range(n):
                node = VertexNode(
                    ring_edges[(i - 1) % n], ring_edges[i], nring[i], self.tol_angle
                )
                ring_nodes.append(node)

            for i in range(n):
                ring_nodes[i].prev_node = ring_nodes[(i - 1) % n]
                ring_nodes[i].next_node = ring_nodes[(i + 1) % n]
                self.active_nodes.add(ring_nodes[i])

        self._event_cap = 2000 + 300 * n_vertices

        # step 1c
        for node in sorted(self.active_nodes, key=lambda n: n.seq):
            self._compute_events_for_node(node)

        # step 2
        self.process_events()
        if self.timed_out:
            return

        self._resolve_remnant_loops()
        self._assemble()

        # Coverage self-check (in normalized space): a correct skeleton tiles the
        # footprint exactly. A shortfall/overshoot means a degenerate event corrupted
        # the assembly; treat as failure so compute() retries with jitter.
        expected = self.poly.area * self._scale * self._scale
        covered = 0.0
        for f in self.faces:
            c = self.nodes[f, :2]
            k = len(c)
            a = 0.0
            for i in range(k):
                x0, y0 = c[i]
                x1, y1 = c[(i + 1) % k]
                a += x0 * y1 - x1 * y0
            covered += abs(a) * 0.5
        if expected <= 0 or abs(covered - expected) / expected > 0.005:
            self.faces = []
            return

        # Reject crossing/overlapping faces.
        if not self._faces_disjoint():
            self.faces = []
            return

        # Map node coordinates and heights back to the input frame.
        if self.nodes.size:
            self.nodes[:, 0] = self.nodes[:, 0] / self._scale + cx
            self.nodes[:, 1] = self.nodes[:, 1] / self._scale + cy
            self.nodes[:, 2] = self.nodes[:, 2] / self._scale

        # Derive the point-pair arc list (for .plot() / the arc API) from the
        # de-normalized nodes, dropping zero-length degenerate arcs.
        min_len = self.tol_dist / self._scale
        self.arcs = [
            (
                (float(self.nodes[u, 0]), float(self.nodes[u, 1])),
                (float(self.nodes[w, 0]), float(self.nodes[w, 1])),
            )
            for (u, w, _a, _b) in self._arc_records
            if u != w
            and math.hypot(
                self.nodes[u, 0] - self.nodes[w, 0],
                self.nodes[u, 1] - self.nodes[w, 1],
            )
            > min_len
        ]

    def _faces_disjoint(self) -> bool:
        """True if the assembled faces form a valid partition (no overlaps)."""
        seen = set()
        for f in self.faces:
            k = len(f)
            for i in range(k):
                e = (f[i], f[(i + 1) % k])
                if e in seen:
                    return False
                seen.add(e)
        return True

    def _assemble(self) -> None:
        """Group skeleton arcs (+ each edge's ground segment) into one CCW face
        per original polygon edge, populating ``self.nodes`` and ``self.faces``."""
        nodes = np.array(self._nodes, dtype=float) if self._nodes else np.empty((0, 3))
        self.nodes = nodes
        faces = []

        for e in self.edges:
            face_edges = []
            if e.ground_start_node != e.ground_end_node:
                face_edges.append((e.ground_start_node, e.ground_end_node))
            for ai in self._edge_arcs.get(e, ()):
                u, w, _a, _b = self._arc_records[ai]
                if u != w:
                    face_edges.append((u, w))
            if len(face_edges) < 3:
                continue

            adj = {}
            for idx, (u, w) in enumerate(face_edges):
                adj.setdefault(u, []).append((w, idx))
                adj.setdefault(w, []).append((u, idx))

            start = face_edges[0][0]
            visited = set()
            chain = [start]
            curr = start
            while True:
                opts = [(nxt, idx) for (nxt, idx) in adj[curr] if idx not in visited]
                if not opts:
                    break
                nxt_node, edge_idx = opts[0]
                visited.add(edge_idx)
                chain.append(nxt_node)
                curr = nxt_node
                if curr == start:
                    break

            loop = _dedupe_consecutive(chain)
            loop = _remove_collinear(loop, nodes)
            if len(loop) < 3:
                continue

            # Orient CCW via the shoelace sign.
            area2 = 0.0
            m = len(loop)
            for i in range(m):
                x0, y0 = nodes[loop[i], 0], nodes[loop[i], 1]
                x1, y1 = nodes[loop[(i + 1) % m], 0], nodes[loop[(i + 1) % m], 1]
                area2 += x0 * y1 - x1 * y0
            if area2 < 0:
                loop = loop[::-1]
            faces.append(loop)

        self.faces = faces

    def _resolve_remnant_loops(self) -> None:
        """Close any degenerate loops still active after the queue is drained."""
        changed = True
        while changed:
            changed = False
            for n in sorted(
                (x for x in self.active_nodes if not x.processed), key=lambda x: x.seq
            ):
                if n.processed:
                    continue
                m = n.next_node
                if m is n:  # 1-node loop: degenerate point, nothing to emit
                    n.processed = True
                    self.active_nodes.discard(n)
                    changed = True
                elif m.next_node is n and not m.processed:  # 2-node loop: ridge segment
                    if math.hypot(n.v0[0] - m.v0[0], n.v0[1] - m.v0[1]) > self.tol_dist:
                        self._record_arc(
                            self._node(n.v0, n.t0),
                            self._node(m.v0, m.t0),
                            n.edge_in,
                            n.edge_out,
                        )
                    n.processed = m.processed = True
                    self.active_nodes.difference_update([n, m])
                    changed = True

    def _compute_edge_event(
        self, A: VertexNode, B: VertexNode
    ) -> Optional[IntersectionEvent]:
        """Step 1c: Computes the time and location where the edge shared
        by adjacent nodes A and B collapses to zero length.

        Instead of intersecting two angle-bisector rays as in the paper (near parallel
        problems), we solve this kinematically. The shared edge collapses exactly when
        three consecutive moving wavefronts (A.edge_in, the shared edge, and B.edge_out)
        crash into a single point simultaneously.

        A stationary edge line is defined by `(x - p) . n = 0`. As it sweeps inward
        at 1 unit of distance per 1 unit of time `t`, its equation becomes
        `(x - p) . n = t`, which elegantly rearranges to `n_x*x + n_y*y - t = n . p`.

        Stacking this equation for all three edges creates a 3x3 linear system:

            [ n1_x  n1_y  -1 ]   [ x ]   [ n1 . p1 ]
            [ n2_x  n2_y  -1 ] * [ y ] = [ n2 . p2 ]
            [ n3_x  n3_y  -1 ]   [ t ]   [ n3 . p3 ]

        The solution (x, y, t) yields the exact meeting point and time. This is
        more robust against degeneracies: if the three walls form a parallel or
        diverging channel that will never collapse, the matrix naturally becomes
        singular, allowing us to safely ignore it without crashing.
        """
        # A 2-node loop is always a fully-degenerate collinear pair (its two edges are
        # the same segment traversed in opposite directions), so it has already
        # collapsed onto a single ridge arc -- resolve it immediately.
        if B.next_node is A:
            ev = IntersectionEvent(
                max(self.current_time, A.t0, B.t0), "edge", A.v0, A, v_b=B
            )
            ev.two_node = True
            return ev

        e1, e2, e3 = A.edge_in, A.edge_out, B.edge_out
        sol = self._concurrency(e1, e2, e3)
        if sol is None:
            return None
        x, y, t = sol
        if t < self.current_time - self.tol_time:
            return None

        return IntersectionEvent(t, "edge", (x, y), A, v_b=B)

    @staticmethod
    def _line_coeffs(e: Edge) -> tuple[float, float, float, float]:
        return (e.n_x, e.n_y, -1.0, e.n_x * e.p1[0] + e.n_y * e.p1[1])

    def _concurrency(
        self, e1: Edge, e2: Edge, e3: Edge
    ) -> Optional[tuple[float, float, float]]:
        """Solve the 3x3 line-concurrency system for (x, y, t); None if singular."""
        a1, b1, c1, d1 = self._line_coeffs(e1)
        a2, b2, c2, d2 = self._line_coeffs(e2)
        a3, b3, c3, d3 = self._line_coeffs(e3)

        det = (
            a1 * (b2 * c3 - b3 * c2)
            - b1 * (a2 * c3 - a3 * c2)
            + c1 * (a2 * b3 - a3 * b2)
        )
        if abs(det) < self.tol_angle:
            return None

        # Cramer's rule
        dx = (
            d1 * (b2 * c3 - b3 * c2)
            - b1 * (d2 * c3 - d3 * c2)
            + c1 * (d2 * b3 - d3 * b2)
        )
        dy = (
            a1 * (d2 * c3 - d3 * c2)
            - d1 * (a2 * c3 - a3 * c2)
            + c1 * (a2 * d3 - a3 * d2)
        )
        dt = (
            a1 * (b2 * d3 - b3 * d2)
            - b1 * (a2 * d3 - a3 * d2)
            + d1 * (a2 * b3 - a3 * b2)
        )
        return (dx / det, dy / det, dt / det)

    def _in_wedge(self, point: Point, Y: VertexNode, X: VertexNode, ej: Edge) -> bool:
        """Test whether the candidate point lays in the area limited by the edge and bisectors."""

        # 1. Point must be inside the polygon (behind the shrinking edge)
        dist_to_ej = (point[0] - ej.p1[0]) * ej.n_x + (point[1] - ej.p1[1]) * ej.n_y
        if dist_to_ej < -self.tol_dist:
            return False

        # 2. Check Y bisector side using NORMALIZED cross products
        Yv_len = math.hypot(*Y.v)
        if Yv_len > 0:
            dy, dx = point[1] - Y.v0[1], point[0] - Y.v0[0]
            dist_I = math.hypot(dx, dy)
            if dist_I > self.tol_dist:
                cb_norm = (Y.v[0] * dy - Y.v[1] * dx) / (Yv_len * dist_I)
                ce_norm = (Y.v[0] * ej.dir_y - Y.v[1] * ej.dir_x) / Yv_len
                if cb_norm * ce_norm < -self.tol_angle:
                    return False

        # 3. Check X bisector side using NORMALIZED cross products
        Xv_len = math.hypot(*X.v)
        if Xv_len > 0:
            dy, dx = point[1] - X.v0[1], point[0] - X.v0[0]
            dist_I = math.hypot(dx, dy)
            if dist_I > self.tol_dist:
                cb_norm = (X.v[0] * dy - X.v[1] * dx) / (Xv_len * dist_I)
                ce_norm = (X.v[0] * (-ej.dir_y) - X.v[1] * (-ej.dir_x)) / Xv_len
                if cb_norm * ce_norm < -self.tol_angle:
                    return False

        return True

    def _compute_events_for_node(self, V: VertexNode) -> None:
        if V.processed:
            return

        if not V.prev_node.processed:
            e_prev = self._compute_edge_event(V.prev_node, V)
            if e_prev:
                self._push_event(e_prev)

        if not V.next_node.processed:
            e_next = self._compute_edge_event(V, V.next_node)
            if e_next:
                self._push_event(e_next)

        if not V.is_reflex:
            return

        best_split = None

        for ej in self.edges:
            if ej == V.edge_in or ej == V.edge_out:
                continue

            a = V.v[0] * ej.n_x + V.v[1] * ej.n_y
            approach_rate = 1.0 - a
            if approach_rate <= self.tol_angle:
                continue

            dist_v0_to_ej = (V.v0[0] - ej.p1[0]) * ej.n_x + (
                V.v0[1] - ej.p1[1]
            ) * ej.n_y
            if dist_v0_to_ej < -self.tol_dist:
                continue

            # V moves as pos(t) = v0 + (t - t0) * v; it reaches ej's offset line when
            # (pos(t) - ej.p1) . n_ej == t. Solving keeps the t0 (birth-time) term so
            # split timing is correct for reflex vertices created mid-sweep, not just
            # the original ones (t0 == 0).
            t = (dist_v0_to_ej - V.t0 * a) / approach_rate
            if t < self.current_time - self.tol_time:
                continue

            point = V.pos(t)

            # The candidate point and its time t depend only on V's bisector and
            # ej's supporting line, so they are fixed for the life of the event.
            # We only confirm here that some live sub-segment of ej currently
            # admits the split; the *actual* opposite pair (Y, X) is re-resolved
            # at processing time in _split_event_valid (paper §2.2 step 2e), since
            # ej may be split again before this event fires.
            if self._find_split_pair(point, ej) is None:
                continue

            if best_split is None or t < best_split.t:
                best_split = IntersectionEvent(t, "split", point, V, opp_edge=ej)

        if best_split is not None:
            self._push_event(best_split)

    def process_events(self) -> None:
        iters = 0
        while self.pq:
            # Safety net: degenerate inputs can re-inject near-simultaneous events
            # without ever draining the queue. Bail instead of hanging.
            iters += 1
            if iters > self._event_cap or len(self.pq) > 300_000:
                self.timed_out = True
                return
            event = heapq.heappop(self.pq)

            # Never process an event behind the current sweep front
            if event.t < self.current_time - self.tol_time:
                continue
            self.current_time = max(self.current_time, event.t)

            if event.event_type == "edge":
                if not self._edge_event_valid(event):
                    continue

                v_a, v_b = event.v_a, event.v_b

                if v_a.processed or v_b.processed:
                    continue

                # 2-node loop: a fully-collapsed collinear slab. Its two nodes are the
                # endpoints of a single ridge arc; connect them directly.
                if event.two_node or v_b.next_node is v_a:
                    if (
                        math.hypot(v_a.v0[0] - v_b.v0[0], v_a.v0[1] - v_b.v0[1])
                        > self.tol_dist
                    ):
                        self._record_arc(
                            self._node(v_a.v0, v_a.t0),
                            self._node(v_b.v0, v_b.t0),
                            v_a.edge_in,
                            v_a.edge_out,
                        )
                    v_a.processed = v_b.processed = True
                    self.active_nodes.difference_update([v_a, v_b])
                    continue

                if v_a.prev_node.prev_node is v_b:
                    v_c = v_a.prev_node
                    i_node = self._node(event.I, event.t)
                    for V in (v_a, v_b, v_c):
                        self._record_arc(
                            self._node(V.v0, V.t0), i_node, V.edge_in, V.edge_out
                        )
                    v_a.processed = v_b.processed = v_c.processed = True
                    self.active_nodes.difference_update([v_a, v_b, v_c])
                    continue

                i_node = self._node(event.I, event.t)
                for V in (v_a, v_b):
                    self._record_arc(
                        self._node(V.v0, V.t0), i_node, V.edge_in, V.edge_out
                    )

                pred_Va, succ_Vb = v_a.prev_node, v_b.next_node
                v_a.processed = v_b.processed = True
                self.active_nodes.difference_update([v_a, v_b])

                v_new = VertexNode(
                    v_a.edge_in, v_b.edge_out, event.I, self.tol_angle, t0=event.t
                )
                v_new.prev_node, v_new.next_node = pred_Va, succ_Vb
                pred_Va.next_node, succ_Vb.prev_node = v_new, v_new
                self.active_nodes.add(v_new)

                self._compute_events_for_node(v_new)
                self._compute_events_for_node(pred_Va)
                self._compute_events_for_node(succ_Vb)

            elif event.event_type == "split":
                if not self._split_event_valid(event):
                    continue

                v = event.v_a
                if v.processed:
                    continue

                Y, X = event.opp_Y, event.opp_X
                if Y.processed or X.processed:
                    self._compute_events_for_node(v)
                    continue

                self._record_arc(
                    self._node(v.v0, v.t0),
                    self._node(event.I, event.t),
                    v.edge_in,
                    v.edge_out,
                )

                v.processed = True
                self.active_nodes.discard(v)

                pred_V, succ_V = v.prev_node, v.next_node
                ej = event.opp_edge

                v1 = VertexNode(v.edge_in, ej, event.I, self.tol_angle, t0=event.t)
                v2 = VertexNode(ej, v.edge_out, event.I, self.tol_angle, t0=event.t)

                v1.prev_node, v1.next_node = pred_V, X
                pred_V.next_node, X.prev_node = v1, v1

                v2.prev_node, v2.next_node = Y, succ_V
                Y.next_node, succ_V.prev_node = v2, v2

                self.active_nodes.update([v1, v2])

                self._compute_events_for_node(v1)
                self._compute_events_for_node(v2)
                self._compute_events_for_node(pred_V)
                self._compute_events_for_node(succ_V)
                self._compute_events_for_node(X)
                self._compute_events_for_node(Y)

    def plot(
        self, save_path: Optional[str] = None, dpi: int = 200, show: bool = True
    ) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(40, 40))

        polys = self.poly.geoms if hasattr(self.poly, "geoms") else [self.poly]

        for p in polys:
            x, y = p.exterior.xy
            ax.plot(x, y, color="black", linewidth=2, label="Polygon")

            for hole in p.interiors:
                hx, hy = hole.xy
                ax.plot(hx, hy, color="black", linewidth=1, linestyle="--")

        for i, arc in enumerate(self.arcs):
            p1, p2 = arc
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="red",
                linewidth=1.5,
                alpha=0.7,
                label="Straight Skeleton" if i == 0 else "",
            )

        minx, miny, maxx, maxy = self.poly.bounds

        width = maxx - minx
        height = maxy - miny
        pad = 0.05 * max(width, height)

        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)

        ax.set_aspect("equal", adjustable="box")

        ax.set_title("Straight Skeleton")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        if save_path:
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.1,
            )

        if show:
            plt.show()

        plt.close(fig)
