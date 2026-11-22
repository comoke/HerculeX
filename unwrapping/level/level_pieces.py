"""Pieces from whole levels solved exactly."""

import ctypes
import itertools
import os
import time

import numpy as np
from scipy import ndimage as ni

from sampling.piece_images import axis_points
from unwrapping.detection.kernel_bindings import CELL
from unwrapping.level import kernel_bindings as kb
from unwrapping.level import piece_flattening as pf
from unwrapping.level import piece_growth as pg
from unwrapping.level import scroll_axis
from unwrapping.level.orientation import turned
from unwrapping.level.piece_flattening import (
    FLATTEN_STEPS,
    FLATTEN_TOLERANCE,
    MIN_PIECE_NODES,
)
from unwrapping.level.piece_kernel_bindings import library as piece_kernels
from unwrapping.level.sheet_graph import (
    file_backed,
    filtered,
    load_field,
    merged_edges,
    radial_pairs,
)
from unwrapping.level.winding_order import best_curl

HIDDEN = 0.5
DRAWS = 8
THREADS = 8
DRAW_BYTES = float(os.environ.get("HERCULEX_DRAW_BYTES", "3e9"))
UNITS = 64
COARSE_STEP = 32
SEED = 1
BOOT = 1
JOIN_SHAPE = 0
SPREAD = 0.30
FLOOR_MM2 = 100.0
GUARD_CUT = 2
GUARD_ROUNDS = 2
CURVE_VOXELS = 8
HELD_TO_AXIS = 0.01
FLATTEN_THREADS = 8
FOLD_VOXELS = 3.0
CONTACT_CELLS = 5
CONTACT_LAYERS = 2.0
CONTACT_DEPTH = 4.0
SEGMENT_PLANES = 256
SEGMENT_EXTENT = (2, 98)
STACK_TURNED_SHARE = 0.10
AXIS_TURNED_SHARE = 0.05
FAR_VOXELS = 1.0e6


def agreeing(nodes, runs):
    """Runs of edges without those between opposed normals."""
    kept = []
    for run in runs:
        steps = np.empty_like(run)
        piece_kernels().hx_turned_steps(run.ctypes.data, len(run), nodes.ctypes.data, steps.ctypes.data)
        kept.append(file_backed(np.ascontiguousarray(run[steps["step"] == 0])))
    return kept


def axis_of(nodes, record):
    """The published umbilicus where the scan has one, else where the normals converge."""
    try:
        points = axis_points(record["sample"], record["volume"])
    except Exception:
        points = None
    if points is not None and len(points) > 1:
        return np.ascontiguousarray(points / 2.0 ** record["level"])
    return scroll_axis.axis(nodes, float(nodes["z"].min()), float(nodes["z"].max()) + 1)


def dominant_normal(nodes):
    """The xy direction most normals lie along, sign free."""
    n = np.stack([nodes["nx"], nodes["ny"]], axis=1).astype(np.float64) / 127
    n = n[np.hypot(n[:, 0], n[:, 1]) > 0.7]
    return np.linalg.eigh(n.T @ n)[1][:, -1]


def core_segments(nodes):
    """Per window of at least SEGMENT_PLANES cell planes, the scan split evenly: the core as a segment through the nodes' centroid along their long axis, its half length half the long extent less the short one."""
    plane = (nodes["z"] // CELL).astype(np.int64)
    low, high = int(plane.min()), int(plane.max()) + 1
    edges = np.linspace(low, high, max(1, (high - low) // SEGMENT_PLANES) + 1)
    rows, last = [], None
    for a, b in itertools.pairwise(edges):
        here = (plane >= a) & (plane < b)
        if here.sum() < 1000:
            continue
        xy = np.stack([nodes["x"][here], nodes["y"][here]], axis=1).astype(np.float64)
        middle = xy.mean(axis=0)
        vectors = np.linalg.eigh(np.cov((xy - middle).T))[1]
        along, across = vectors[:, -1], vectors[:, 0]
        if last is not None and along @ last < 0:
            along = -along
        last = along
        extent = [np.subtract(*np.percentile((xy - middle) @ v, SEGMENT_EXTENT[::-1])) for v in (along, across)]
        z = float(nodes["z"][here].mean())
        rows.append((z, *middle, *along, max(0.0, 0.5 * (extent[0] - extent[1]))))
    return np.array(rows, dtype=np.float64)


def segment_centres(nodes, segments):
    """Each node's nearest point of the core segment of its height, segments interpolated between windows."""
    z = nodes["z"].astype(np.float64)
    cx, cy, dx, dy, half = (np.interp(z, segments[:, 0], segments[:, k]) for k in range(1, 6))
    length = np.maximum(np.hypot(dx, dy), 1e-9)
    dx, dy = dx / length, dy / length
    t = np.clip((nodes["x"] - cx) * dx + (nodes["y"] - cy) * dy, -half, half)
    return cx + t * dx, cy + t * dy


def oriented(nodes, centre):
    """Angles about the centres and normals turned away from them."""
    ox, oy = nodes["x"] - centre[0], nodes["y"] - centre[1]
    flip = (nodes["nx"].astype(np.float32) * ox + nodes["ny"].astype(np.float32) * oy) < 0
    for axis in ("nx", "ny", "nz"):
        nodes[axis] = np.where(flip, -nodes[axis], nodes[axis])
    return np.arctan2(oy, ox).astype(np.float32)


def turned_share(nodes, first, grid, centre):
    """Share of nodes turned to agree with their sheets when angles and normals are taken about a centre, normals left as they were."""
    saved = {axis: nodes[axis].copy() for axis in ("nx", "ny", "nz")}
    theta = oriented(nodes, centre)
    share = pg.turn_to_sheets(nodes, pg.sheet_edges(nodes, first, grid, theta, pg.node_quality(nodes)))
    for axis, values in saved.items():
        nodes[axis] = values
    return share


def centres_of(nodes, first, grid, record):
    """Each node's centre: far along the sheets for a flat stack, the scan's axis where its sheets agree with it, else the nearest point of the core segment."""
    start = time.time()
    direction = dominant_normal(nodes)
    far = (
        float(np.median(nodes["x"])) + FAR_VOXELS * direction[0],
        float(np.median(nodes["y"])) + FAR_VOXELS * direction[1],
    )
    z0, z1 = float(nodes["z"].min()), float(nodes["z"].max()) + 1
    as_stack = turned_share(nodes, first, grid, far)
    found = {"turned_as_stack": round(as_stack, 4)}
    if as_stack < STACK_TURNED_SHARE:
        record["centres"] = {"rule": "stack", **found, "seconds": round(time.time() - start, 1)}
        return far, np.array([[z0, *far], [z1, *far]], dtype=np.float64)
    axis = axis_of(nodes, record)
    about_axis = turned_share(nodes, first, grid, scroll_axis.centre_at(axis, nodes["z"]))
    found["turned_about_axis"] = round(about_axis, 4)
    if about_axis < AXIS_TURNED_SHARE:
        record["centres"] = {"rule": "axis", **found, "seconds": round(time.time() - start, 1)}
        return scroll_axis.centre_at(axis, nodes["z"]), axis
    segments = core_segments(nodes)
    core_mm = 2 * segments[:, 5] * record["scan_voxel_um"] * 2 ** record["level"] / 1000
    record["centres"] = {
        "rule": "segment",
        **found,
        "core_mm": core_mm.round(1).tolist(),
        "segments": segments.round(4).tolist(),
        "seconds": round(time.time() - start, 1),
    }
    return segment_centres(nodes, segments), np.ascontiguousarray(segments[:, :3])


def prepare(chunks, planes=None):
    """Nodes with normals turned to agree along the sheets and repaired before rays are cast, their angle about each
    node's centre, the sheet edges, each node's next sheets along its normal, the centres' knots."""
    raw, cells, record, grid, ids = load_field(chunks, planes)
    nodes, first, cell_of, ids = filtered(raw, cells, grid, ids)
    record["ids"] = ids
    del raw, cells, cell_of
    centre, knots = centres_of(nodes, first, grid, record)
    theta = oriented(nodes, centre)
    del centre
    record["edges"] = pg.sheet_edges(nodes, first, grid, theta, pg.node_quality(nodes))
    record["turned"] = pg.turn_to_sheets(nodes, record["edges"])
    repair, report = turned(nodes, record["edges"])
    for axis in ("nx", "ny", "nz"):
        nodes[axis] = np.where(repair, -nodes[axis], nodes[axis])
    record["orientation"] = report
    record["edges"] = agreeing(nodes, record["edges"])
    pairs = radial_pairs(nodes, first, grid, record["cell_planes"][0], np.ones(len(nodes), dtype=bool))
    return file_backed(nodes), theta, first, grid, file_backed(pairs), record, knots


def flatten_pieces(nodes, root, a, b):
    """Normals' signs and (u, v) of every piece, each laid flat in frames carried along its sheet."""
    count = len(nodes)
    sizes = np.bincount(root, minlength=count)
    big = np.flatnonzero(sizes >= MIN_PIECE_NODES)
    big = big[np.argsort(-sizes[big], kind="stable")]
    rank = np.full(count, -1, dtype=np.int32)
    rank[big] = np.arange(len(big), dtype=np.int32)
    label = rank[root]
    members = np.flatnonzero(label >= 0).astype(np.int32)
    order = np.ascontiguousarray(members[np.argsort(label[members], kind="stable")])
    piece_first = np.zeros(len(big) + 1, dtype=np.int64)
    np.cumsum(sizes[big], out=piece_first[1:])
    inside = int((label[a] >= 0).sum())
    ea, eb = np.zeros(inside, dtype=np.int32), np.zeros(inside, dtype=np.int32)
    edge_first = np.zeros(len(big) + 1, dtype=np.int64)
    piece_kernels().hx_group_edges(
        pf.pointer(a),
        pf.pointer(b),
        ctypes.c_int64(len(a)),
        pf.pointer(label),
        len(big),
        pf.pointer(edge_first),
        pf.pointer(ea),
        pf.pointer(eb),
    )
    return (
        order,
        piece_first,
        *kb.flatten(
            nodes,
            order,
            piece_first,
            ea,
            eb,
            edge_first,
            FLATTEN_STEPS,
            FLATTEN_TOLERANCE,
            HELD_TO_AXIS,
            FLATTEN_THREADS,
        ),
    )


def contact_cells(nodes, normal, u, v, flat, grid, locked):
    """Cells holding nodes of more than one sheet kept with the nodes nearest the depth their locked neighbours give,
    their mean moved to the cell's centre as every cell's is (piece_flattening.centred)."""
    occupied = grid[0] != 0
    weight = ni.uniform_filter(locked.astype(np.float64), CONTACT_CELLS, mode="constant")
    near = np.stack([ni.uniform_filter(np.where(locked, g, 0.0), CONTACT_CELLS, mode="constant") for g in grid])
    near /= np.maximum(weight, 1e-9)
    open_cells = (occupied & ~locked & (weight > 1.5 / CONTACT_CELLS**2)).ravel()
    pick = open_cells[flat]
    if not pick.any():
        return grid, locked
    xyz = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1).astype(np.float64)
    depth = np.full(len(nodes), np.inf)
    depth[pick] = ((xyz[pick] - near.reshape(3, -1)[:, flat[pick]].T) * normal[pick]).sum(axis=1)
    nearest = np.full(locked.size, np.inf)
    np.minimum.at(nearest, flat[pick], np.abs(depth[pick]))
    take = pick & (np.abs(depth) <= nearest[flat] + CONTACT_LAYERS)
    count = np.maximum(np.bincount(flat[take], minlength=locked.size), 1).astype(np.float64)
    mean = np.stack([np.bincount(flat[take], weights=xyz[take, k], minlength=locked.size) for k in range(3)]) / count
    offset = np.bincount(flat[take], weights=depth[take], minlength=locked.size) / count
    square = np.bincount(flat[take], weights=depth[take] ** 2, minlength=locked.size) / count
    spread = np.sqrt(np.maximum(square - offset**2, 0))
    good = open_cells & (np.bincount(flat[take], minlength=locked.size) > 0)
    good &= (spread <= FOLD_VOXELS) & (np.abs(offset) <= CONTACT_DEPTH)
    mean = pf.centred(xyz, u, v, locked.shape, mean, take, locked.ravel()[flat])
    out = grid.reshape(3, -1).copy()
    out[:, good] = mean[:, good]
    return out.reshape(grid.shape), locked | good.reshape(locked.shape)


def rasterise(nodes, normal, u, v):
    """Cells centred on the sheet (piece_flattening.rasterise), with both layers of a sheet in one cell taken as sheet, and contact cells kept with the sheet their neighbours continue."""
    grid, locked, crowding = pf.rasterise(nodes, normal, u, v)
    iu, iv = ((u - u.min()) / pf.CELL).astype(np.int64), ((v - v.min()) / pf.CELL).astype(np.int64)
    flat = iv * locked.shape[1] + iu
    xyz = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1).astype(np.float64)
    count = np.maximum(np.bincount(flat, minlength=locked.size), 1)
    mean = np.stack([np.bincount(flat, weights=xyz[:, k], minlength=locked.size) for k in range(3)]) / count
    depth = ((xyz - mean[:, flat].T) * normal).sum(axis=1)  # from the cell's mean, before it is centred
    spread = np.sqrt(np.bincount(flat, weights=depth**2, minlength=locked.size) / count)
    locked = (grid[0] != 0) & (spread.reshape(locked.shape) <= FOLD_VOXELS)
    grid, locked = contact_cells(nodes, normal, u, v, flat, grid, locked)
    return grid, locked, crowding


def along_curve(nodes, patch, count):
    """Patches numbered along a Morton curve, so that neighbours share cache lines in the solve."""
    first = np.zeros(count, dtype=np.int32)
    first[patch[::-1]] = np.arange(len(patch) - 1, -1, -1, dtype=np.int32)
    bits = np.arange(1 << 12, dtype=np.uint64)
    table = sum(((bits >> np.uint64(i)) & np.uint64(1)) << np.uint64(3 * i) for i in range(12))
    key = sum(
        table[(nodes[axis][first] // CURVE_VOXELS).astype(np.int32) & 4095] << np.uint64(shift)
        for shift, axis in enumerate("xyz")
    )
    rank = np.empty(count, dtype=np.int32)
    rank[np.argsort(key, kind="stable")] = np.arange(count, dtype=np.int32)
    return rank[patch]


def levels_of_pieces(root, potential, attract, pairs, rays, theta, chirality, rays_up=None):
    """Each node's level from the exact solve over grown pieces."""
    ids, piece = np.unique(root, return_inverse=True)
    piece = piece.astype(np.int32)
    p, q, k, term = kb.level_terms(piece, potential, len(ids), attract, pairs, rays, theta, chirality)
    x, _ = kb.minimize(
        len(ids),
        p,
        q,
        k,
        *kb.term_weights(len(p), term, attract, rays, HIDDEN, UNITS, rays_up),
        np.zeros(len(ids), dtype=np.int32),
        COARSE_STEP,
    )
    return x[piece].astype(np.int64) + chirality * potential.astype(np.int64)


def firm_edges(
    nodes, theta, attract, pairs, quality, chirality, grown_root, grown_potential, patch, count, inner, clock
):
    """Levels of every node by one exact solve, a mask of the edges every reweighted solve keeps, of those the solve keeps, and the rays' weights and skip costs."""
    rays, rays_up = pg.field_rays(nodes, pairs, quality, attract, pg.FIELD_RAY_RIM)
    patch = along_curve(nodes, patch, count)
    start = np.zeros(count, dtype=np.int32)
    start[patch] = (
        levels_of_pieces(grown_root, grown_potential, attract, pairs, rays, theta, chirality, rays_up)
        - chirality * inner
    )
    clock("start")
    p, q, k, term = kb.level_terms(patch, inner, count, attract, pairs, rays, theta, chirality)
    clock("terms")
    x, log = kb.minimize(count, p, q, k, *kb.term_weights(len(p), term, attract, rays, HIDDEN, UNITS, rays_up), start)
    del start
    clock("solve")
    threads = max(1, min(THREADS, int(DRAW_BYTES / (20 * len(p) + 56 * count))))
    votes = kb.votes(
        count, p, q, k, term, attract, rays, HIDDEN, x, DRAWS, threads, UNITS, SEED, BOOT, JOIN_SHAPE, rays_up
    )
    clock("draws")
    kept = (x[q] - x[p]) == k
    firm_term = kept & (votes >= DRAWS)
    y = x[patch].astype(np.int64) + chirality * inner
    step = chirality * attract["step"].astype(np.int64)
    edge_kept = (y[attract["b"]] - y[attract["a"]]) == step
    edge_term = term[: len(attract)]
    firm = edge_kept & np.where(edge_term == -1, True, firm_term[np.where(edge_term >= 0, edge_term, -2 - edge_term)])
    return (
        np.ascontiguousarray(firm),
        np.ascontiguousarray(edge_kept),
        y.astype(np.int32),
        rays,
        rays_up,
        {
            "patches": count,
            "terms": len(p),
            "energy": int(log[2]),
            "cuts": int(log[0]),
            "kept": float(edge_kept.mean()),
            "firm": float(firm.mean()),
        },
    )


def node_area(nodes, scan_voxel_um, level):
    norm = np.sqrt(sum(nodes[c].astype(np.float32) ** 2 for c in ("nx", "ny", "nz")))
    spread = sum(np.abs(nodes[c].astype(np.float32)) for c in ("nx", "ny", "nz")) / np.maximum(norm, 1e-6)
    return ((CELL * 2**level * scan_voxel_um / 1000) ** 2 / np.maximum(spread, 1.0)).astype(np.float32)


def refused_pieces(nodes, root, potential, attract, area_node, cells=False):
    """Nodes of pieces that fail the guard; with cells, also each node of such a piece that lies in a cell of two
    layers."""
    inside = pg.same_label(attract, root) & pg.potential_agrees(attract, potential)
    a, b = np.ascontiguousarray(attract["a"][inside]), np.ascontiguousarray(attract["b"][inside])
    members, first, flip, u, v = flatten_pieces(nodes, root, a, b)
    ids, piece = np.unique(root, return_inverse=True)
    area = np.bincount(piece, weights=area_node)
    refused = np.zeros(len(ids), dtype=bool)
    doubled = np.zeros(len(nodes), dtype=bool)
    report = []
    for r in np.argsort(-np.diff(first)):
        group = members[first[r] : first[r + 1]]
        pid = piece[group[0]]
        if area[pid] < FLOOR_MM2:
            break
        normal = np.stack([nodes[key][group].astype(np.float32) for key in ("nx", "ny", "nz")], axis=1) * flip[
            group, None
        ].astype(np.float32)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        grid, locked, _ = pf.rasterise(nodes[group], normal, u[group], v[group])
        spread = 1 - float(locked.sum() / max(int((grid[0] != 0).sum()), 1))
        refused[pid] = spread > SPREAD
        if cells and refused[pid]:
            doubled[group] = second_layer(nodes[group], normal, u[group], v[group], grid, locked)
        report.append((round(float(area[pid]) / 100, 1), round(spread, 3)))
    return (refused[piece], doubled, report) if cells else (refused[piece], report)


def second_layer(nodes, normal, u, v, grid, locked):
    """Nodes of a piece's cells of two layers that the cut drops: all of them, or with GUARD_CUT 2 all but the layer
    nearest the depth the cell's locked neighbours give (as contact_cells keeps it when writing)."""
    iu = ((u - u.min()) / CELL).astype(np.int64)
    iv = ((v - v.min()) / CELL).astype(np.int64)
    open_node = ~locked[iv, iu]
    if GUARD_CUT < 2 or not open_node.any():
        return open_node
    weight = ni.uniform_filter(locked.astype(np.float64), CONTACT_CELLS, mode="constant")
    near = np.stack([ni.uniform_filter(np.where(locked, g, 0.0), CONTACT_CELLS, mode="constant") for g in grid])
    near /= np.maximum(weight, 1e-9)
    xyz = np.stack([nodes["x"], nodes["y"], nodes["z"]], axis=1).astype(np.float64)
    depth = ((xyz - near[:, iv, iu].T) * normal).sum(axis=1)
    flat = iv * locked.shape[1] + iu
    nearest = np.full(locked.size, np.inf)
    np.minimum.at(nearest, flat[open_node], np.abs(depth[open_node]))
    guided = weight[iv, iu] > 1.5 / CONTACT_CELLS**2
    keep = open_node & guided & (np.abs(depth) <= nearest[flat] + CONTACT_LAYERS)
    return open_node & ~keep


def guard_cut(nodes, root, potential, attract, area_node):
    """The guard, cutting a refused piece out of its cells of two layers: the rest splits along the piece's own
    edges, and each part passes or is cut again, up to GUARD_ROUNDS times; what still fails is refused."""
    refused, doubled, report = refused_pieces(nodes, root, potential, attract, area_node, cells=True)
    out = refused & doubled
    trial = refused & ~doubled
    for _ in range(GUARD_ROUNDS):
        if not trial.any():
            break
        use = pg.same_label(attract, root) & pg.potential_agrees(attract, potential)
        use &= trial[attract["a"]] & trial[attract["b"]]
        group = np.where(trial, root + 1, 0).astype(np.int32)
        cut_root, cut_potential = pg.settled(nodes, attract, use, group).labels()
        root = np.where(trial, cut_root, root).astype(np.int32)
        potential = np.where(trial, cut_potential, potential).astype(np.int32)
        probe = np.where(trial, root, np.arange(len(nodes), dtype=np.int32)).astype(np.int32)
        again, doubled, more = refused_pieces(nodes, probe, potential, attract, area_node, cells=True)
        out |= trial & again & doubled
        trial &= again & ~doubled
        report += more
    return root, potential, out | trial, report


def fallen_back(nodes, attract, kept, root, potential, refused, coarse_root):
    """Refused nodes into the quick pieces they lie in, joined only along the edges the solve keeps."""
    inside = kept & pg.same_label(attract, coarse_root) & refused[attract["a"]] & refused[attract["b"]]
    back_root, back_potential = pg.settled(nodes, attract, inside, np.zeros(len(nodes), dtype=np.int32)).labels()
    return np.where(refused, back_root, root).astype(np.int32), np.where(refused, back_potential, potential).astype(
        np.int32
    )


def two_tiers(nodes, attract, firm, kept, area_node):
    """Pieces of firm edges where they reach a usable size."""
    root, potential = pg.settled(nodes, attract, firm, np.zeros(len(nodes), dtype=np.int32)).labels()
    _ids, piece = np.unique(root, return_inverse=True)
    large = (np.bincount(piece, weights=area_node) >= FLOOR_MM2)[piece]
    rest_root, rest_potential = pg.settled(nodes, attract, kept, large.astype(np.int32) * (root + 1)).labels()
    return np.where(large, root, rest_root).astype(np.int32), np.where(large, potential, rest_potential).astype(
        np.int32
    )


def quick_pieces(nodes, theta, loose, pairs, quality, record, curl=None):
    """Quick pieces the level solve starts from: patches merged on the evidence between them, and the curl."""
    attract = merged_edges(loose)
    loose.clear()
    repel = np.zeros(len(pairs), dtype=attract.dtype)
    repel["a"], repel["b"], repel["weight"] = (
        pairs["a"],
        pairs["b"],
        pg.field_rays(nodes, pairs, quality, attract, pg.RAY_RIM)[0],
    )
    repel = file_backed(np.ascontiguousarray(repel[np.argsort(-repel["weight"], kind="stable")]))
    kept = np.ones(len(attract), dtype=bool)
    patch, count, inner = pg.patches_of(nodes, attract, kept, repel)
    forest, stats = pg.agglomerated(nodes, patch, count, attract, kept, repel)
    root, potential = forest.labels()
    del forest, repel
    apart = attract[~pg.same_label(attract, root)]
    chirality, costs, _ = best_curl(nodes, root, potential, apart, pairs, np.ones(len(nodes), dtype=np.int8))
    record["frame_costs"] = {str(c): round(v, 4) for c, v in costs.items()}
    return attract, root, potential, patch, count, inner, curl or chirality, stats


def piece_terms(root, potential, level, attract, pairs, rays, theta, chirality, rays_up=None):
    """What the scan's one frame needs of a unit."""
    sizes = np.bincount(root, minlength=len(root))
    big = np.flatnonzero(sizes >= MIN_PIECE_NODES)
    big = big[np.argsort(-sizes[big], kind="stable")]
    rank = np.full(len(root), len(big), dtype=np.int32)
    rank[big] = np.arange(len(big), dtype=np.int32)
    label = rank[root]
    out = {"level": level[big] - chirality * potential[big], "chirality": chirality}
    out["turns"] = (
        np.bincount(label, weights=potential + theta / (2 * np.pi), minlength=len(big) + 1)[: len(big)] / sizes[big]
    )
    for curl, name in ((1, "plus"), (-1, "minus")):
        p, q, k, term = kb.level_terms(label, potential, len(big) + 1, attract, pairs, rays, theta, curl)
        up, dn = kb.term_weights(len(p), term, attract, rays, HIDDEN, UNITS, rays_up)
        real = (p < len(big)) & (q < len(big))
        out |= {
            f"p_{name}": p[real],
            f"q_{name}": q[real],
            f"k_{name}": k[real],
            f"up_{name}": up[real],
            f"dn_{name}": dn[real],
        }
    return out


def grow_pieces(nodes, theta, pairs, record, curl=None):
    """(root, winding potential, along-sheet edges inside pieces as a and b, edges between pieces, stats)."""
    times, last = {}, [time.time()]

    def clock(name):
        now = time.time()
        times[name] = round(now - last[0], 1)
        last[0] = now

    quality = pg.node_quality(nodes)
    attract, coarse_root, coarse_potential, patch, count, inner, chirality, stats = quick_pieces(
        nodes, theta, record.pop("edges"), pairs, quality, record, curl
    )
    clock("quick pieces")
    firm, kept, level, rays, rays_up, report = firm_edges(
        nodes, theta, attract, pairs, quality, chirality, coarse_root, coarse_potential, patch, count, inner, clock
    )
    area_node = node_area(nodes, record["scan_voxel_um"], record["level"])
    root, potential = two_tiers(nodes, attract, firm, kept, area_node)
    del firm
    clock("firm pieces")
    if GUARD_CUT:
        root, potential, refused, pieces = guard_cut(nodes, root, potential, attract, area_node)
    else:
        refused, pieces = refused_pieces(nodes, root, potential, attract, area_node)
    clock("guard")
    if refused.any():
        root, potential = fallen_back(nodes, attract, kept, root, potential, refused, coarse_root)
    del kept, coarse_root, coarse_potential
    same = pg.same_label(attract, root)
    inside = same & pg.potential_agrees(attract, potential)
    record["piece_terms"] = piece_terms(root, potential, level, attract, pairs, rays, theta, chirality, rays_up)
    record["node_level"] = level
    record["level_route"] = report | {"refused_share": float(refused.mean()), "pieces": pieces[:40], "seconds": times}
    print("level route", {k: v for k, v in record["level_route"].items() if k != "pieces"}, flush=True)
    return root, potential, attract["a"][inside], attract["b"][inside], attract[~same], stats
