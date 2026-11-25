"""A stream of stacks reads as each stack alone, survives engine failures and shares memory unchanged."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tifffile
import torch

from ink.model import tiles
from ink.model.reader_apple import tile_reader
from ink.model.reader_apple.neural_engine import SIDE, EngineReader, package_of
from ink.model.tiles import DEPTH, STEM_CHANNELS
from ink.reading import piece_reading
from ink.reading.piece_reading import parcelled, unparcelled
from storage.brick_store import BrickStore

sample_piece = piece_reading.sampled_piece


def mark_before_sampling_writes(job):
    wait = piece_reading.wait_for_model

    def mark():
        job[1].with_suffix(".waiting").touch()
        wait()

    piece_reading.wait_for_model = mark
    try:
        return sample_piece(job)
    finally:
        piece_reading.wait_for_model = wait


class StandInStem:
    device = "cpu"

    def statistics_of(self, tiles, scale):
        return scale, float(tiles.astype(np.float64).mean())

    def __call__(self, image, statistics):
        return (
            image.float().mean(1, keepdim=True).repeat(1, 32, 1, 1)
            * (torch.arange(32).reshape(1, -1, 1, 1) / 32 + statistics[1] / 255)
        ).half()

    def tiles(self, block, statistics, gap):
        for offset in (0, gap) if gap else (0,):
            tile = torch.from_numpy(block[:, :, offset : offset + SIDE].copy())
            yield self(tiles.scaled(tile, statistics[0])[None], statistics)


def stand_in_reader(engine):
    reader = tile_reader.TileReader.__new__(tile_reader.TileReader)
    reader.torch, reader.device, reader.stem, reader.engine = torch, "cpu", StandInStem(), engine
    reader.weights, reader.gpu_engine = None, engine
    reader.body = lambda features: {"ink": features.float().mean(1, keepdim=True) * 0.2}
    return reader


def engine(features):
    return features[0].astype(np.float32).mean(0) * np.float32(0.3) - np.float32(0.1)


def stacks():
    rng = np.random.default_rng(3)
    return [
        rng.integers(1, 255, shape, dtype=np.uint8)
        for shape in ((17, 700, 640), (17, 96, 64), (17, 300, 900), (17, 481, 800), (17, 512, 512))
    ]


def test_a_stream_of_stacks_is_read_as_each_stack_alone_in_tile_order():
    reader = stand_in_reader(engine)
    alone = [reader.read(stack) for stack in stacks()]
    together = list(reader.maps((stack, tiles.measured(stack)) for stack in stacks()))
    assert len(together) == len(alone) and all(a.tobytes() == b.tobytes() for a, b in zip(alone, together))


def test_a_prepared_stack_keeps_the_original_tile_coverage():
    rng = np.random.default_rng(14)
    stack = rng.integers(1, 255, (17, 600, 820), dtype=np.uint8)
    stack[:, :, :550] = 0
    reader = stand_in_reader(engine)
    prepared = tiles.prepare_stack(stack)
    assert prepared.values[:, :, :512].any() and not prepared.coverage[:, :512].any()
    (read,) = reader.maps([(prepared, tiles.measured(stack))])
    assert np.array_equal(read, reader.read(stack))


def test_fullest_face_windows_keep_raster_order_for_equal_coverage():
    sheet = np.ones((769, 1281), dtype=bool)
    assert piece_reading.fullest(sheet) == ([(0, 0), (0, 512)], 512)


def test_reader_defers_heavy_imports_until_they_are_used():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ink.reading.piece_reading; assert not {'torch', 'scipy', 'tifffile'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_failed_model_loading_leaves_prefetched_piece_unwritten(tmp_path, monkeypatch):
    folder = tmp_path / "scroll" / "volume" / "pieces" / "unit" / "piece-0001"
    folder.mkdir(parents=True)
    y, z = np.meshgrid(np.arange(16, 24, dtype=np.float32), np.arange(16, 24, dtype=np.float32))
    for axis, grid in zip("xyz", (np.full_like(y, 24), y, z)):
        tifffile.imwrite(folder / f"{axis}.tif", grid)
    (folder / "meta.json").write_text(json.dumps({"stride": 1, "voxel_um": 9.4}))
    store = BrickStore(tmp_path / "bricks", (48, 48, 48), (48, 48, 48))
    store.put((0, 0, 0), np.full(store.brick, 128, dtype=np.uint8).tobytes())
    store.save()
    face = tmp_path / "piece-0001.png"
    windows = tmp_path / "windows"

    def fail_model(_):
        deadline = time.monotonic() + 10
        while not face.with_suffix(".waiting").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert face.with_suffix(".waiting").is_file()
        raise ValueError("model could not open")

    monkeypatch.setenv("HERCULEX_WINDOWS", str(windows))
    monkeypatch.setattr(piece_reading, "_model", None)
    monkeypatch.setattr(piece_reading, "SAMPLERS", 1)
    monkeypatch.setattr(piece_reading, "sampled_piece", mark_before_sampling_writes)
    monkeypatch.setattr(piece_reading, "lane_reader", fail_model)
    with pytest.raises(ValueError, match="model could not open"):
        piece_reading.read_all([(folder, face, None, 1, "model")], store.folder, False)
    assert not list(tmp_path.glob("piece-0001*.png")) and not windows.exists() and store.folder.is_dir()


def test_a_stack_the_engine_fails_on_gives_its_error_and_the_next_is_read():
    def failing_once(features, calls=[]):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("engine fault")
        return engine(features)

    first, second = stacks()[3], stacks()[0]
    maps = list(stand_in_reader(failing_once).maps([(first, tiles.measured(first)), (second, tiles.measured(second))]))
    assert isinstance(maps[0], ValueError) and maps[1].tobytes() == stand_in_reader(engine).read(second).tobytes()


class StandInNorm:
    def scale_and_shift(self):
        return torch.ones(STEM_CHANNELS), torch.zeros(STEM_CHANNELS)


class StandInSurveyReader(EngineReader):
    """Its shared norms always hold nothing, as if another piece had just been measured."""

    fixed = property(lambda self: (None, None), lambda self, value: None)

    def __init__(self):
        self.tiles, self.torch, self.measuring = tiles, torch, threading.Lock()
        self.norms, self.stem = [StandInNorm(), StandInNorm()], lambda image: image

    def logits(self, image, norms):
        return image.astype(np.float32).mean(0) * np.float32(0.3) - np.float32(0.1)


def read_tile_by_tile(reader, stack):
    """The survey's read as it stood, one tile at a time with each scaled in full."""
    scale, _ = reader.fix_to(stack)
    _, height, width = stack.shape
    blend = np.outer(np.hanning(SIDE + 2)[1:-1], np.hanning(SIDE + 2)[1:-1]).astype(np.float32)
    total, weight = np.zeros((2, height, width), dtype=np.float32)
    block = np.zeros((DEPTH, SIDE, SIDE), dtype=np.uint8)
    for y in sorted({*range(0, max(height - SIDE, 0) + 1, tiles.STEP), max(height - SIDE, 0)}):
        for x in sorted({*range(0, max(width - SIDE, 0) + 1, tiles.STEP), max(width - SIDE, 0)}):
            part = stack[:, y : y + SIDE, x : x + SIDE]
            h, w = part.shape[1:]
            block[:] = 0
            block[:, :h, :w] = part
            read = 1 / (1 + np.exp(-reader.logits(reader.scaled(block, scale), None)))
            total[y : y + h, x : x + w] += read[:h, :w] * blend[:h, :w]
            weight[y : y + h, x : x + w] += blend[:h, :w]
    return total / np.maximum(weight, 1e-6)


def test_the_survey_reader_equals_the_tile_by_tile_read_and_reads_side_by_side():
    reader = StandInSurveyReader()
    alone = [read_tile_by_tile(reader, stack) for stack in stacks()]
    with ThreadPoolExecutor(2) as readers:
        together = list(readers.map(reader.read, stacks()))
    assert all(a.tobytes() == b.tobytes() for a, b in zip(alone, together))


def test_fetch_ahead_skips_cached_windows_and_plans_all_misses_for_the_first_store(tmp_path, monkeypatch):
    reading = piece_reading
    monkeypatch.setattr(reading, "WINDOWS", tmp_path / "windows")

    def job(unit, piece, cached=False):
        folder = tmp_path / "s" / "v" / "pieces" / unit / piece
        out = tmp_path / f"{piece}.png"
        if cached:
            folder.mkdir(parents=True)
            (folder / "meta.json").touch()
            kept = reading.kept_window(folder)
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.touch()
            out.touch()
            reading.layer0(out).touch()
            reading.side_of(out).write_text(json.dumps({"written": "near", "turn": 1}))
        return folder, out, None, 1, "model"

    cached = job("a", "piece-0001", cached=True)
    mixed = job("b", "piece-0002", cached=True)
    first, later = job("b", "piece-0003"), job("b", "piece-0004")
    skipped = job("c", "piece-0005")
    monkeypatch.setattr(
        reading, "queue_of", lambda *args: [("a", [cached]), ("b", [mixed, first]), ("c", [skipped]), ("b", [later])]
    )
    fetched = []
    monkeypatch.setattr(reading, "fetch_bricks", lambda sample, volume, jobs, folder: fetched.append((folder, jobs)))
    reading.fetch_ahead("s", "v", 1)
    assert fetched == [("b", [first, later])]


def test_reader_resumes_missing_or_stale_images(tmp_path, monkeypatch):
    reading = piece_reading
    folder, out = tmp_path / "mesh", tmp_path / "piece-0001.png"
    folder.mkdir()
    meta = folder / "meta.json"
    meta.write_text('{"voxel_um": 9.4}')
    job = folder, out, None, -1, "model"
    reading.side_of(out).write_text('{"written": "near", "turn": -1, "model": "model"}')
    out.touch()
    taught = out.with_name("piece-0001-taught.png")
    taught.write_bytes(b"archived ink")
    assert not reading.done(job)
    scan = np.zeros((8, 8), dtype=np.uint8)
    monkeypatch.setattr(reading, "sampled", lambda *_: (scan, scan, None))
    assert "failed" not in reading.sampled_piece(job)["row"]
    assert taught.read_bytes() == b"archived ink"
    assert reading.done(job)
    assert not reading.done((folder, out, None, 1, "model"))
    taught.unlink()
    assert not reading.done(job)
    meta.write_text('{"voxel_um": 4.7}')
    os.utime(meta, (1, 1))
    assert not reading.done(job), "a piece finer than the reader's training scans still needs its ink"
    taught.write_bytes(b"archived ink")
    assert reading.done(job)
    os.utime(meta, (2e9, 2e9))
    assert not reading.done(job)


def test_earlier_maps_are_reused_only_for_the_equivalent_checkpoint(tmp_path, monkeypatch):
    reading = piece_reading
    weights = tmp_path / "weights.pth"
    weights.write_bytes(bytes(2**20 + 1))
    monkeypatch.setattr(reading, "WEIGHTS", weights)
    original = reading.digest()
    original_package = package_of(weights)
    stat = weights.stat()
    with weights.open("r+b") as f:
        f.seek(2**20)
        f.write(b"\x01")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement = reading.digest()
    assert original != replacement
    assert package_of(weights) != original_package

    new_file = weights.with_suffix(".new")
    new_file.write_bytes(bytes(2**20 + 1))
    os.utime(new_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    new_file.replace(weights)
    assert reading.digest() == original
    assert package_of(weights) == original_package

    monkeypatch.setattr(reading, "SAME_OUTPUT", {original: {"old-digest"}, replacement: {"other-model"}})
    record = tmp_path / "read.json"
    record.write_text(json.dumps({"model": "old-digest"}))
    assert reading.read_by(record, original)
    assert not reading.read_by(record, replacement)
    record.write_text(json.dumps({"model": "other-model"}))
    assert not reading.read_by(record, original)


def test_empty_columns_are_filled_from_the_sheet_and_the_sheet_is_left_alone():
    stack = np.random.default_rng(1).integers(1, 255, (5, 64, 48), dtype=np.uint8)
    assert tiles.filled(stack) is stack
    holed = stack.copy()
    holed[:, 10:30, 5:40] = 0
    filled = tiles.filled(holed)
    kept = holed.any(axis=0)
    assert np.array_equal(filled[:, kept], holed[:, kept]) and filled.any(axis=0).all()
    columns = {column.tobytes() for column in holed[:, kept].T}
    assert all(column.tobytes() in columns for column in filled[:, ~kept].T)
    assert np.array_equal(filled, tiles.filled(holed))
    with pytest.raises(ValueError, match="cannot reshape"):
        tiles.filled(holed, kept[:-1])


def test_a_sparse_stack_keeps_the_original_fill_when_the_tries_run_out():
    stack = np.random.default_rng(73).integers(1, 255, (5, 64, 48), dtype=np.uint8)
    stack[:, :61] = 0
    stack[:, :, :43] = 0
    filled = tiles.filled(np.asfortranarray(stack))
    assert (
        hashlib.sha256(filled.tobytes()).hexdigest()
        == "358a905e407ab9ebe11c0f0b47e9b5d415741b0af66a29bd8cb101ab137e546c"
    )


def test_arrays_cross_in_shared_memory_unchanged():
    arrays = [
        np.arange(12, dtype=np.uint8).reshape(3, 4),
        np.ones((2, 2), dtype=np.float32),
        np.zeros((0, 5), dtype=np.uint8),
    ]
    back = unparcelled(*parcelled(arrays))
    assert all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(arrays, back))
