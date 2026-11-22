"""A surface from the unwrapping stage sampled along its normals into a layer stack of the raw scan."""

import json
import mmap
import os
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from sampling.kernel_bindings import library
from scans.catalog.scan_catalog import load, volumes
from scans.transfer.parallel_fetch import fetch_all
from storage.brick_store import BrickStore

LAYERS = 31
ROOM_BYTES = 25 * 2**30
FLOOR_BYTES = 22 * 2**30
ROOM_WAIT_SECONDS = 30
RETRY_PAUSE_SECONDS = 60
reader_waiting = threading.Event()
TILE = 128
NORMAL_UM = 225.0
MESH_SMOOTHING_CELLS = 1.5
BRIDGE_CELLS = 8
RELAXATIONS = 40
FOLD_CELLS = 3.0
SAMPLING = f"mesh-bridged-{BRIDGE_CELLS}-smoothed-{MESH_SMOOTHING_CELLS}"
# 1 asks the system to read the planes a surface will sample before the sampling threads reach them
READ_AHEAD = os.environ.get("HERCULEX_READ_AHEAD", "1") == "1"


def load_surface(folder):
    """(3, rows, columns) float32 x, y, z in voxels, with the surface's record."""
    import tifffile

    folder = Path(folder)
    return np.stack([tifffile.imread(folder / f"{a}.tif") for a in "xyz"]), json.loads(
        (folder / "meta.json").read_text()
    )


def smoothed_mesh(xyz, cells=MESH_SMOOTHING_CELLS):
    """A mesh smoothed along itself, holes left where they are."""
    if cells == 0:
        return xyz
    from scipy import ndimage as ni

    good = xyz[0] > 0
    weight = np.maximum(ni.gaussian_filter(good.astype(np.float32), cells), 1e-4)
    return np.where(good, np.stack([ni.gaussian_filter(np.where(good, p, 0), cells) / weight for p in xyz]), -1).astype(
        np.float32
    )


def bridged_mesh(xyz, stride, reach=BRIDGE_CELLS):
    """A mesh carried across its narrow gaps and holes by a relaxed fill."""
    good = xyz[0] > 0
    if reach == 0 or good.all():
        return xyz
    from scipy import ndimage as ni

    pad = reach + 2
    gap = (
        ni.binary_fill_holes(
            ni.binary_closing(np.pad(good, pad), ni.generate_binary_structure(2, 2), iterations=reach)
        )[pad:-pad, pad:-pad]
        & ~good
    )
    if not gap.any():
        return xyz
    weight = np.maximum(ni.gaussian_filter(good.astype(np.float32), reach / 2), 1e-6)
    grid = np.stack(
        [np.where(good, p, ni.gaussian_filter(np.where(good, p, 0), reach / 2) / weight) for p in xyz]
    ).astype(np.float32)
    for _ in range(RELAXATIONS):
        edged = np.pad(grid, ((0, 0), (1, 1), (1, 1)), mode="edge")
        around = 0.25 * (edged[:, :-2, 1:-1] + edged[:, 2:, 1:-1] + edged[:, 1:-1, :-2] + edged[:, 1:-1, 2:])
        grid[:, gap] = around[:, gap]
    known = good | gap
    fold = np.zeros_like(known)
    for axis in (1, 2):
        step = np.sqrt((np.diff(grid, axis=axis) ** 2).sum(axis=0)) > FOLD_CELLS * stride
        here = tuple(slice(None, -1) if a == axis - 1 else slice(None) for a in (0, 1))
        there = tuple(slice(1, None) if a == axis - 1 else slice(None) for a in (0, 1))
        long = step & known[here] & known[there]
        fold[here] |= long
        fold[there] |= long
    return np.where(known & ~(fold & gap), grid, -1).astype(np.float32)


def readable_mesh(xyz, stride, cells=MESH_SMOOTHING_CELLS, reach=BRIDGE_CELLS):
    """The mesh a stack is sampled on."""
    return smoothed_mesh(bridged_mesh(xyz, stride, reach), cells)


def grid_normals(xyz, centre, node_um):
    """Unit normals on the surface's grid, turned towards the scroll's centre, where the writing faces."""
    from scipy import ndimage as ni

    good = xyz[0] > 0
    fill = ni.distance_transform_edt(~good, return_distances=False, return_indices=True)
    smooth = np.stack([ni.gaussian_filter(p[fill[0], fill[1]], NORMAL_UM / node_um) for p in xyz])
    du, dv = np.gradient(smooth, axis=(1, 2))
    normal = np.cross(du, dv, axis=0)
    normal /= np.maximum(np.linalg.norm(normal, axis=0), 1e-9)
    inward = centre[:, None, None] - smooth
    inward[2] = 0
    if (normal * inward)[:, good].sum() < 0:
        normal = -normal
    return normal.astype(np.float32)


def columns(plane, stride):
    """A grid plane at every voxel of the surface, bilinear between grid points."""
    h, w = plane.shape
    y = np.minimum(np.arange((h - 1) * stride + 1) / stride, h - 1.000001)
    x = np.minimum(np.arange((w - 1) * stride + 1) / stride, w - 1.000001)
    a, b = y.astype(np.int64), x.astype(np.int64)
    fy, fx = (y - a).astype(np.float32)[:, None], (x - b).astype(np.float32)[None, :]
    top = plane[np.ix_(a, b)] * (1 - fx) + plane[np.ix_(a, b + 1)] * fx
    low = plane[np.ix_(a + 1, b)] * (1 - fx) + plane[np.ix_(a + 1, b + 1)] * fx
    return top * (1 - fy) + low * fy


def scan_store(sample, volume_id, out, level=0):
    """The partial store of one level of a scan, and the base URL of its bricks."""
    row = next(v for v in volumes(load()) if v["sample"] == sample and v["volume"] == volume_id)
    root, path = row["roots"]["ome-zarr"], row["products"]["ome-zarr"]
    base = f"{root}/{path.strip('/')}/{level}"
    array = json.load(urllib.request.urlopen(f"{base}/.zarray", timeout=60))
    return BrickStore(out, tuple(array["shape"]), tuple(array["chunks"]), source=f"{path}/{level}"), base


def keep_room(store):
    """Offsets saved, so a stopped fetch resumes, then a wait while the disc is full."""
    store.save()
    start = time.time()
    while (free := shutil.disk_usage(store.folder).free) < (FLOOR_BYTES if reader_waiting.is_set() else ROOM_BYTES):
        print(
            f"fetch of {store.folder.name} waits for disc room: {free / 2**30:.1f} GiB free, {'the read waits on it' if reader_waiting.is_set() else 'ahead of the read'}, {(time.time() - start) / 60:.0f} min so far",
            flush=True,
        )
        time.sleep(ROOM_WAIT_SECONDS)


def fetched_all(base, items, deliver, report=None, fetch=fetch_all):
    """fetch_all, with what failed tried once more after a pause."""
    failures = fetch(base, items, deliver, report=report)
    if failures:
        print(
            len(failures),
            "bricks failed, first",
            failures[0],
            "; trying them again in",
            RETRY_PAUSE_SECONDS,
            "s",
            flush=True,
        )
        time.sleep(RETRY_PAUSE_SECONDS)
        again = {key for key, _ in failures}
        failures = fetch(base, [item for item in items if item[0] in again], deliver, report=report)
    return failures


def column_cells(cells, stride):
    """Each voxel column's grid cell and fraction along one axis of a grid of so many cells."""
    place = np.minimum(np.arange((cells - 1) * stride + 1) / stride, cells - 1.000001)
    cell = place.astype(np.int32)
    return cell, (place - cell).astype(np.float32)


class LayerStack:
    """A surface's layers in memory, read as a brick store is."""

    def __init__(self, cube):
        self.cube, self.shape = cube, cube.shape

    def read(self, z, y, x):
        return self.cube[z[0] : z[1], y[0] : y[1], x[0] : x[1]]


def read_ahead(data, store, offsets, xyz, normal, stride):
    """Advise the system of every brick plane a surface's stack reads, so they arrive in large reads ahead of the faults."""
    from sampling.touched_planes import mark_ranges

    ranges = np.empty((*store.grid, 2), dtype=np.int32)
    first, last = ranges[..., 0], ranges[..., 1]
    first.fill(store.brick[0])
    last.fill(-1)
    mark_ranges(ranges, store, xyz, normal, stride)
    chosen = (last >= 0) & (offsets >= 0)
    plane, page, size = store.brick[1] * store.brick[2], mmap.PAGESIZE, len(data)
    begins = offsets[chosen] + first[chosen].astype(np.int64) * plane
    ends = np.minimum(offsets[chosen] + (last[chosen].astype(np.int64) + 1) * plane, size)
    for begin, end in zip((begins // page * page).tolist(), ends.tolist()):
        if end > begin:
            data._mmap.madvise(mmap.MADV_WILLNEED, begin, end - begin)


def flatten(xyz, stride, normal, store, *, layers=LAYERS):
    """The centred layer stack of a surface, tile by tile."""
    _, h, w = xyz.shape
    grid = np.ascontiguousarray(np.concatenate([(xyz[:1] > 0).astype(np.float32), xyz, normal]), dtype=np.float32)
    (row, row_part), (col, col_part) = column_cells(h, stride), column_cells(w, stride)
    H, W = len(row), len(col)
    stack = np.zeros((layers, H, W), dtype=np.uint8)
    lib = library()
    offsets = store.offsets.copy()
    data = np.memmap(store.path, dtype=np.uint8, mode="r")
    geometry = np.array([*store.grid, *store.brick], dtype=np.int32)
    if READ_AHEAD and layers == LAYERS and len(data):
        read_ahead(data, store, offsets, xyz, normal, stride)

    def run(origin):
        ty, tx = origin
        block, aim = np.empty((2, 3, TILE, TILE), dtype=np.float32)
        cube = np.empty((layers, TILE, TILE), dtype=np.uint8)
        if lib.hx_sample_grid(
            data.ctypes.data,
            offsets.ctypes.data,
            geometry.ctypes.data,
            grid.ctypes.data,
            h,
            w,
            row.ctypes.data,
            col.ctypes.data,
            row_part.ctypes.data,
            col_part.ctypes.data,
            H,
            W,
            ty,
            tx,
            TILE,
            layers,
            -(layers // 2),
            block.ctypes.data,
            aim.ctypes.data,
            cube.ctypes.data,
        ):
            part = stack[:, ty : ty + TILE, tx : tx + TILE]
            part[:] = cube[:, : part.shape[1], : part.shape[2]]

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(run, [(y, x) for y in range(0, H, TILE) for x in range(0, W, TILE)]))
    return LayerStack(stack)
