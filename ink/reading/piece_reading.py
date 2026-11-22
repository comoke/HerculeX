"""Ink stage: every piece sampled once, its written face chosen, its ink map read."""

import argparse
import collections
import json
import multiprocessing
import os
import shutil
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import zstandard
from PIL import Image

from ink.model import input_stack, models, tiles
from ink.model.input_stack import DEPTH, PAST_FACE, face_layer_of, scrambled
from ink.model.reader import SOURCES, lane_reader
from ink.model.stock_model import SURE, checkpoint_hash
from runner import events
from sampling import column_sampling as sampler
from sampling.fibre_faces import recto_face
from sampling.piece_images import (
    axis_of_pieces,
    axis_points,
    fetch_bricks,
    shown,
    store_lock,
    surface_of,
)
from sampling.sheet_images import across_sheet
from storage.brick_store import UNKNOWN, BrickStore
from storage.data_roots import (
    FACES,
    INK_LAYER_STACKS,
    PRODUCTION_FACES,
    ROUTE_OUTPUTS,
    ct_bricks,
    named_folder,
    route_id,
    stage_folder,
)

Image.MAX_IMAGE_PIXELS = None  # Faces are generated locally and may exceed Pillow's default limit.
MODEL = models.chosen()
WEIGHTS = MODEL.weights
STACK_ROUTE = route_id(ROUTE_OUTPUTS)
WINDOWS = Path(
    os.environ.get("HERCULEX_WINDOWS", named_folder(INK_LAYER_STACKS, STACK_ROUTE.removesuffix("_route"), STACK_ROUTE))
)
SHIFTS = tuple(sorted({0, *map(int, os.environ.get("HERCULEX_DEPTH_SHIFTS", "0").split(","))}))
UNITS = set(filter(None, os.environ.get("HERCULEX_UNITS", "").split(",")))
REACH = max(map(abs, SHIFTS))
CORE_LAYERS = sampler.LAYERS
if REACH:
    sampler.LAYERS = CORE_LAYERS + 2 * REACH  # Bricks are fetched deep enough for the farthest shift.
KEEP_FREE_BYTES = 40 * 2**30
LOOK_SIDE, LOOKS = 512, 2
LOOKED_PIXELS = 2**20
SAMPLERS = int(os.environ.get("HERCULEX_SAMPLERS", "4"))
# 1 keeps every unit's fetched CT store after its read, so the next read of the same units fetches only new bricks
KEEP_BRICKS = os.environ.get("HERCULEX_KEEP_BRICKS") == "1"
KEEP_STACKS = os.environ.get("HERCULEX_KEEP_STACKS") != "0"
PIECES_AHEAD = int(os.environ.get("HERCULEX_PIECES_AHEAD", "6" if len(SHIFTS) == 1 else "3"))
WRITERS = 2
SAME_OUTPUT = {
    current: set(earlier)
    for current, *earlier in (
        line.split()
        for line in (Path(__file__).parent / "same_output_models.txt").read_text().splitlines()
        if line.strip()
    )
}
READING = "reading.started"  # written when a read starts; the early fetch (unit_fetch) leaves the stores to it then
_store = None
_store_folder = None
_model = None
_model_ready = None


def digest():
    model = checkpoint_hash(WEIGHTS)
    for source in SOURCES:
        model.update(source.read_bytes())
    model.update(sampler.SAMPLING.encode())
    if (FACES, SHIFTS) != (PRODUCTION_FACES, (0,)):
        model.update(json.dumps([FACES, SHIFTS]).encode() + Path(input_stack.__file__).read_bytes())
    return model.hexdigest()[:12]


def start_sampler(store_folder, ready=None):
    global _store, _store_folder, _model_ready
    _store, _store_folder, _model_ready = None, store_folder, ready


def wait_for_model():
    """Sampling may start before model loading, with its writes held until the model opens."""
    global _model_ready
    if _model_ready is not None:
        _model_ready.wait()
        _model_ready = None


def core_of(cube):
    """The production layers of a stack sampled wider about the same middle."""
    extra = max((cube.shape[0] - CORE_LAYERS) // 2, 0)
    return cube[extra : cube.shape[0] - extra], extra


def window(cube, shift=0, *, layers=DEPTH, contiguous=True):
    """The production window, moved by whole layers or widened about its middle."""
    core, extra = core_of(cube)
    first = extra + min(face_layer_of(core) + PAST_FACE, core.shape[0] - DEPTH) + shift - (layers - DEPTH) // 2
    part = cube[first : first + layers]
    return np.ascontiguousarray(part) if contiguous else part


def depth_reads(kept, measure=None):
    """The written face at each shift of SHIFTS out of its window widened by REACH, the production window first with the statistics given."""
    others = [np.ascontiguousarray(kept[REACH + shift : REACH + shift + DEPTH]) for shift in SHIFTS if shift]
    return [
        (np.ascontiguousarray(kept[REACH : REACH + DEPTH]), measure),
        *((w, tiles.measured(w)) for w in others),
    ]


def fullest(sheet):
    """Corners of the windows of a piece most covered by sheet."""
    h, w = sheet.shape
    side = min(LOOK_SIDE, h, w)
    step = side // 2
    rows, cols = range(0, h - side + 1, step), range(0, w - side + 1, step)
    if sheet.dtype == np.bool_ and side == 2 * step:
        # Each window contains four disjoint cells. Count each cell only once.
        cells = np.array(
            [
                [np.count_nonzero(sheet[y : y + step, x : x + step]) for x in range(0, (len(cols) + 1) * step, step)]
                for y in range(0, (len(rows) + 1) * step, step)
            ],
            dtype=np.int64,
        )
        mass = cells[:-1, :-1] + cells[1:, :-1] + cells[:-1, 1:] + cells[1:, 1:]
    else:
        mass = np.array([[sheet[y : y + side, x : x + side].astype(np.int64).sum() for x in cols] for y in rows])
    chosen = []
    for spot in np.argsort(-mass.ravel(), kind="stable"):
        y, x = rows[spot // len(cols)], cols[spot % len(cols)]
        if all(abs(y - a) >= side or abs(x - b) >= side for a, b in chosen):
            chosen.append((y, x))
        if len(chosen) == LOOKS:
            break
    return chosen, side


def sure_of(read):
    return float((read[read > 0] >= SURE).mean()) if (read > 0).any() else 0.0


def faces_of(cube, sheet):
    """The written face widened by REACH, which face it is, what is known so far, and what the reader is to take in turn with the look each is for."""
    core, _ = core_of(cube)
    fibres = recto_face(core)
    written = "far" if fibres["face"] == "far" else "near"
    looks = {"fibres": fibres, "near": 0.0, "far": 0.0}
    wide = DEPTH + 2 * REACH
    if sheet.sum() < LOOKED_PIXELS:
        kept = window(cube[::-1] if written == "far" else cube, layers=wide)
        return kept, written, looks, depth_reads(kept, tiles.measured(kept[REACH : REACH + DEPTH])), []
    faces = {"near": window(cube, contiguous=written == "near"), "far": window(cube[::-1], contiguous=written == "far")}
    kept = window(cube[::-1] if written == "far" else cube, layers=wide) if REACH else faces[written]
    rng = np.random.default_rng(cube.shape[1] * 31 + cube.shape[2])
    spots, side = fullest(sheet)
    stacks, seen = [], []
    for name in sorted(faces, key=lambda face: face == written):
        windows = [np.ascontiguousarray(faces[name][:, y : y + side, x : x + side]) for y, x in spots]
        stacks += [(part, tiles.measured(faces[name]) if k == 0 else None) for k, part in enumerate(windows)]
        seen += [name] * len(windows)
    return kept, written, looks, [*stacks, (scrambled(windows[0], rng), None), *depth_reads(kept)], [*seen, "scrambled"]


def side_of(out):
    return out.with_name(f"{out.stem}-taught-side.json")


def layer0(out):
    return out.with_name(f"{out.stem}-layer0.png")


def taught_png(out):
    return out.with_name(f"{out.stem}-taught.png")


def read_side(record):
    try:
        side = json.loads(record.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return side if isinstance(side, dict) else {}


def read_by(record, model, turn=None):
    side = read_side(record)
    recorded = side.get("model")
    return (turn is None or side.get("turn", 1) == turn) and (
        recorded == model or recorded in SAME_OUTPUT.get(model, ())
    )


def current(path, folder):
    try:
        return path.stat().st_mtime >= (folder / "meta.json").stat().st_mtime
    except FileNotFoundError:
        return False


def kept_window(folder, *, writing=False):
    """Cache path keyed by the scan as well as the scroll and piece."""
    production = (FACES, REACH) == (PRODUCTION_FACES, 0)
    sampling = sampler.SAMPLING if production else f"{sampler.SAMPLING}-{FACES}-{DEPTH + 2 * REACH}-layers"
    scan = folder.parents[2]
    root = WINDOWS / scan.parent.name
    suffix = Path(sampling) / folder.parent.name / f"{folder.name}.zst"
    current = root / scan.name / suffix
    legacy = root / suffix
    if (
        not writing
        and not current.exists()
        and legacy.is_file()
        and list(scan.parent.glob("*/pieces")) == [scan / "pieces"]
    ):
        return legacy
    return current


def cached_window(folder, out, turn):
    kept = kept_window(folder)
    record = side_of(out)
    if not (current(kept, folder) and out.is_file() and layer0(out).is_file() and record.is_file()):
        return {}
    if kept.stat().st_mtime > record.stat().st_mtime:
        return {}
    side = read_side(record)
    return side if side.get("written") and side.get("turn", 1) == turn else {}


def keep(path, face):
    """A window kept while the disc has room."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free > KEEP_FREE_BYTES:
        part = path.with_suffix(".part")
        compressor = zstandard.ZstdCompressor(level=1, threads=2)
        with part.open("wb") as out:
            out.write(struct.pack("III", *face.shape))
            out.write(compressor.compress(np.ascontiguousarray(face)))
        part.replace(path)


def kept_face(path):
    raw = path.read_bytes()
    shape = struct.unpack_from("III", raw)
    return np.frombuffer(
        zstandard.ZstdDecompressor().decompress(memoryview(raw)[12:], max_output_size=int(np.prod(shape))),
        dtype=np.uint8,
    ).reshape(shape)


def complete_store(folder):
    """A fetched store; one with bricks still unknown would read as zeros there, so it is refused."""
    store = BrickStore(folder)
    unknown = int((store.offsets == UNKNOWN).sum())
    if unknown:
        raise RuntimeError(f"{folder} holds {unknown} bricks never fetched; rerun to fetch them")
    return store


def sampled(folder, axis, turn):
    """A piece's stack along its normals and its scan images."""
    global _store
    if _store is None:
        _store = complete_store(_store_folder)
    xyz, meta = sampler.load_surface(folder)
    margined, normal = surface_of(xyz, meta, axis)
    stack = sampler.flatten(margined, meta["stride"], normal * turn, _store, layers=CORE_LAYERS + 2 * REACH)
    sheet, layers = across_sheet(stack, meta["voxel_um"], all_layers=False)
    return sheet, layers[0], stack.cube


def write_faces(out, sheet, central, *, replace_ink=True):
    wait_for_model()
    out.parent.mkdir(parents=True, exist_ok=True)
    if replace_ink:
        side_of(out).unlink(missing_ok=True)
    write_png(layer0(out), central)
    write_png(out, sheet)


def write_png(path, pixels):
    part = path.with_suffix(".png.part")
    Image.fromarray(pixels).save(part, format="PNG", compress_level=3)
    part.replace(path)


def write_side(path, side):
    part = path.with_suffix(".part")
    part.write_text(json.dumps(side))
    part.replace(path)


def parcelled(arrays):
    """Arrays in one block of shared memory for the reader."""
    block = shared_memory.SharedMemory(create=True, size=max(sum(a.nbytes for a in arrays), 1))
    at = 0
    for part in arrays:
        np.ndarray(part.shape, part.dtype, block.buf, at)[...] = part
        at += part.nbytes
    block.close()
    return block.name, [(a.shape, a.dtype.str) for a in arrays]


def unparcelled(name, shapes):
    """The arrays of a block, copied out so the block can go at once."""
    block = shared_memory.SharedMemory(name=name)
    arrays, at = [], 0
    for shape, kind in shapes:
        arrays.append(np.ndarray(shape, kind, block.buf, at).copy())
        at += arrays[-1].nbytes
    block.close()
    block.unlink()
    return arrays


def sampled_piece(job):
    """One piece as far as the reader."""
    folder, out, axis, turn, _ = job
    start = time.time()
    record = side_of(out)
    try:
        looks = cached_window(folder, out, turn)
        cached = bool(looks)
        if cached:
            face, sheet = kept_face(kept_window(folder)), np.asarray(Image.open(out))
            written = looks.pop("written")
            stacks, seen = depth_reads(face, tiles.measured(face[REACH : REACH + DEPTH])), []
        else:
            sheet, central, cube = sampled(folder, axis, turn)
            if ink_is_current(job):
                write_faces(out, sheet, central, replace_ink=False)
                return {
                    "row": {
                        "piece": folder.name,
                        "unit": folder.parent.name,
                        "written": read_side(record).get("written", "near"),
                        "seconds": round(time.time() - start, 1),
                    }
                }
            face, written, looks, stacks, seen = faces_of(cube, sheet > 0)
        measures = [m for _, m in stacks if m is not None]
        prepared = [tiles.prepare_stack(stack) for stack, _ in stacks]
        if cached:
            wait_for_model()
        else:
            write_faces(out, sheet, central)
            if KEEP_STACKS:
                keep(kept_window(folder, writing=True), face)
        block = parcelled(
            [sheet, *(s.values for s in prepared), *(s.coverage for s in prepared), *(t for t, _ in measures)]
        )
        return {
            "job": job,
            "start": start,
            "looks": looks,
            "written": written,
            "seen": seen,
            "block": block,
            "scales": [m and m[1] for _, m in stacks],
        }
    except Exception as error:
        return {"row": {"piece": folder.name, "unit": folder.parent.name, "failed": repr(error)[:200]}}


def opened(piece):
    """The stacks of a sampled piece, each with what fixes its statistics, and its sheet."""
    sheet, *arrays = unparcelled(*piece.pop("block"))
    scales = piece.pop("scales")
    count = len(scales)
    stacks, masks, measures = arrays[:count], arrays[count : 2 * count], iter(arrays[2 * count :])
    return [
        (tiles.PreparedStack(stack, mask), scale and (next(measures), scale))
        for stack, mask, scale in zip(stacks, masks, scales)
    ], sheet


def written_piece(piece, sheet, maps):
    """A read piece's map and record."""
    folder, out, _, turn, model = piece["job"]
    try:
        failure = next((read for read in maps if isinstance(read, Exception)), None)
        if failure:
            raise failure
        looks = piece["looks"]
        for name in dict.fromkeys(piece["seen"]):
            looks[name] = float(np.mean([sure_of(read) for read, seen in zip(maps, piece["seen"]) if seen == name]))
        read = maps[-1] if len(SHIFTS) == 1 else np.mean(maps[-len(SHIFTS) :], axis=0)
        write_png(taught_png(out), shown(read, sheet))
        write_side(side_of(out), looks | {"written": piece["written"], "model": model, "turn": turn})
        return {
            "piece": folder.name,
            "unit": folder.parent.name,
            "written": piece["written"],
            "seconds": round(time.time() - piece["start"], 1),
        }
    except Exception as error:
        return {"piece": folder.name, "unit": folder.parent.name, "failed": repr(error)[:200]}


def jobs_of(base, axis, model, wanted):
    """The read jobs for some (area, unit, piece, turn), in the order given."""
    faces = stage_folder(base, "faces") / FACES
    jobs = []
    for _, unit, piece, turn in wanted:
        out = faces / unit / f"piece-{piece:04d}.png"
        jobs.append((base / "pieces" / unit / f"piece-{piece:04d}", out, axis, turn, model))
    return jobs


def read_all(jobs, store_folder, keep):
    """A store's pieces."""
    global _model
    start, rows, room, reading, stopped = (
        time.time(),
        [],
        threading.Semaphore(PIECES_AHEAD),
        collections.deque(),
        threading.Event(),
    )

    def ended(row):
        rows.append(row)
        room.release()
        if len(rows) % 10 == 0 or "failed" in row:
            print(store_folder.name, len(rows), "of", len(jobs), row, round(time.time() - start), "s", flush=True)

    def spaced():
        for job in jobs:
            room.acquire()
            if stopped.is_set():
                return
            yield job

    def stacks_of(pieces):
        for piece in pieces:
            if "row" in piece:
                ended(piece["row"])
            else:
                stacks, sheet = opened(piece)
                reading.append((piece, sheet, len(stacks)))
                yield from stacks

    context = multiprocessing.get_context("spawn")
    ready = context.Event() if _model is None else None
    with (
        context.Pool(SAMPLERS, start_sampler, (str(store_folder), ready)) as pool,
        ThreadPoolExecutor(WRITERS) as writers,
    ):
        pieces = pool.imap_unordered(sampled_piece, spaced(), chunksize=1)
        maps = []
        try:
            if ready is not None:
                _model = lane_reader(WEIGHTS)
                ready.set()
            for read in _model.maps(stacks_of(pieces)):
                maps.append(read)
                if len(maps) == reading[0][2]:
                    piece, sheet, _ = reading.popleft()
                    writers.submit(written_piece, piece, sheet, maps).add_done_callback(
                        lambda done: ended(done.result())
                    )
                    maps = []
        finally:
            stopped.set()
            room.release()
    failed = sum("failed" in row for row in rows)
    if failed:
        raise RuntimeError(f"{failed} of {len(jobs)} pieces in {store_folder.name} failed. Rerun to resume.")
    if not keep and store_folder.exists():
        shutil.rmtree(store_folder)


def queue_of(sample, volume, first=None):
    """What is left to read, store by store."""
    base = ROUTE_OUTPUTS / sample / volume
    placed = json.loads((stage_folder(base, "sheet") / "layout.json").read_text())["pieces"]
    if not placed:
        return []
    wanted = {
        (p["unit"], p["piece"]): (p["area"], p["unit"], p["piece"], p.get("turn", 1))
        for p in placed
        if not UNITS or p["unit"] in UNITS
    }
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"{MODEL.name}: no checkpoint at {WEIGHTS}; see ./run.sh ink.model.models")
    model = digest()
    queue = []
    for store, pieces in json.loads(Path(first).read_text()) if first else ():
        try:
            ordered = [wanted.pop(tuple(k)) for k in pieces]
        except KeyError as error:
            raise ValueError(f"{first}: unknown or repeated piece {error}") from error
        queue.append((ct_bricks(sample, store), jobs_of(base, None, model, ordered)))
    by_store = collections.defaultdict(list)
    for row in wanted.values():
        by_store[row[1]].append(row)
    by_store = {ct_bricks(sample, unit): rows for unit, rows in by_store.items()}
    for store in sorted(by_store, key=lambda path: (not path.exists(), path)):
        queue.append((store, jobs_of(base, None, model, sorted(by_store[store], reverse=True))))
    pending = [(store, jobs) for store, planned in queue if (jobs := [job for job in planned if not done(job)])]
    if not pending:
        return []
    axis = axis_points(sample, volume)
    if axis is None:
        units = sorted(p for p in (base / "pieces").glob("rows-*") if not p.name.endswith(".building"))
        axis = axis_of_pieces(units or [base / "pieces" / unit for unit in sorted({p["unit"] for p in placed})])
    return [
        (store, [(folder, out, axis, turn, model) for folder, out, _, turn, model in jobs]) for store, jobs in pending
    ]


def planned_stores(queue):
    """All each store will read, in the order stores are first wanted."""
    planned = collections.defaultdict(list)
    for folder, jobs in queue:
        planned[folder] += jobs
    return planned


def fetch_bricks_for_read(sample, volume, jobs, store_folder):
    """Fetch CT only for pieces that will sample from it."""
    uncached = []
    for job in jobs:
        folder, out, _, turn, _ = job
        if not cached_window(folder, out, turn):
            uncached.append(job)
    if uncached:
        fetch_bricks(sample, volume, uncached, store_folder)
    return bool(uncached)


def fetch_ahead(sample, volume, stores, first=None):
    """A read's first stores before it starts, while the link is free and its other slots are not."""
    for folder, jobs in planned_stores(queue_of(sample, volume, first)).items():
        if stores <= 0:
            break
        if fetch_bricks_for_read(sample, volume, jobs, folder):
            stores -= 1


def run(sample, volume, first=None):
    """Every piece, store by store, a store's bricks fetched while the one before is read."""
    queue = queue_of(sample, volume, first)
    if not queue:
        print(f"{sample} {volume}: no ink maps to update", flush=True)
        return
    print(f"{sample} {volume}: {sum(len(jobs) for _, jobs in queue)} pieces queued", flush=True)
    started = stage_folder(ROUTE_OUTPUTS / sample / volume, "sheet") / READING
    started.parent.mkdir(parents=True, exist_ok=True)
    started.write_text(f"{time.time()}\n")
    planned = planned_stores(queue)

    def fetched(folder):
        """A store's planes once, for all it will read."""
        if folder in planned:
            with events.span("fetch", store=folder.name):
                fetch_bricks_for_read(sample, volume, planned.pop(folder), folder)

    with ThreadPoolExecutor(1) as ahead:
        coming = ahead.submit(fetched, queue[0][0])
        for k, (folder, jobs) in enumerate(queue):
            sampler.reader_waiting.set()
            try:
                with events.span("wait-fetch", store=folder.name):
                    coming.result()
            finally:
                sampler.reader_waiting.clear()
            if k + 1 < len(queue):
                coming = ahead.submit(fetched, queue[k + 1][0])
            with events.span("read", store=folder.name, pieces=len(jobs)), store_lock(folder):
                read_all(jobs, folder, KEEP_BRICKS or any(folder == later[0] for later in queue[k + 1 :]))


def ink_is_current(job):
    folder, out, _, turn, model = job
    record = side_of(out)
    if not (current(record, folder) and read_by(record, model, turn)):
        return False
    return taught_png(out).is_file()


def done(job):
    out = job[1]
    return out.is_file() and layer0(out).is_file() and ink_is_current(job)


def main():
    ahead = sys.argv[1:2] == ["ahead"]
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="To prepare CT without reading ink, use ahead SCROLL SCAN STORES [ORDER]."
    )
    parser.add_argument("scroll")
    parser.add_argument("scan")
    if ahead:
        parser.add_argument("stores", type=int)
    parser.add_argument("order", nargs="?")
    args = parser.parse_args(sys.argv[2:] if ahead else None)
    try:
        if ahead:
            fetch_ahead(args.scroll, args.scan, args.stores, args.order)
        else:
            run(args.scroll, args.scan, args.order)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Ink: {error}\n")


if __name__ == "__main__":
    main()
