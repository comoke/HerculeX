"""The Apple lane's ink reader on 512-pixel tiles: the depth stem as one Metal kernel, the body on the Neural Engine."""

import collections
import os
import queue
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from pathlib import Path

import numpy as np

from ink.model import stock_model
from ink.model.reader_apple.neural_engine import CPU_AND_GPU, SIDE, EngineBody
from ink.model.tiles import (
    DEPTH,
    STEM_CHANNELS,
    TILE,
    PreparedStack,
    Sums,
    fixed_norms,
    measured,
    prepare_stack,
    scaled,
    tile_starts,
)

TILES_AHEAD = 8
ENGINE_CALLS = 2
GPU_CALLS = 2
PENDING = 10
PAIR_GAP = 280
# which unit reads each 512 tile's body, by the tile's place and data: E the Neural Engine, G the GPU
LANES = os.environ.get("HERCULEX_BODY_LANES", "EEGEG")
if not LANES or set(LANES) - set("EG"):
    raise SystemExit(f"HERCULEX_BODY_LANES={LANES}: a pattern of E (Neural Engine) and G (GPU), such as EEGEG")
PROBES = ((128, 128), (128, 384), (384, 128), (384, 384))
# tensor: the stem on Metal 4 tensor operations (tensor_stem.metal); kernel: the one-pass kernel (depth_stem.metal)
STEM = os.environ.get("HERCULEX_STEM", "tensor")
ROOMS = 4


class DepthStem:
    """The network's depth stem as one Metal kernel, its norms fixed to the statistics of a stack."""

    def __init__(self, module, device="mps"):
        import torch

        self.torch, self.device, self.module = torch, device, module.to(device).half()
        features = module.features
        if isinstance(features[0], torch.nn.Conv3d):
            features[0], features[3] = stock_model.folded(features[0]), stock_model.folded(features[3])
            module.to(device).half()
        self.norms = fixed_norms(torch, module)
        self.source = (Path(__file__).parent / "depth_stem.metal").read_text()
        self.kernel = torch.mps.compile_shader(self.source)
        self.attention_bias = float(module.attention_logits.bias.float())
        self.scale, self.table = None, None

    @cached_property
    def raw_kernel(self):
        return self.torch.mps.compile_shader("#define RAW_INPUT\n" + self.source)

    @cached_property
    def edge_kernel(self):
        return self.torch.mps.compile_shader("#define RAW_INPUT\n#define TILE_EDGE\n" + self.source)

    def statistics(self, stack):
        return self.statistics_of(*measured(stack))

    def statistics_of(self, tiles, scale):
        """Raw scale and folded stem weights of a stack, as a patch's own norms would take them but the same everywhere on it."""
        torch = self.torch
        functional = torch.nn.functional
        with torch.inference_mode():
            features = self.module.features
            image = scaled(torch.from_numpy(tiles).to(self.device), scale)
            batch, depth, height, width = image.shape
            padded = functional.pad(image, (0, 0, 0, 0, 1, 1))
            stacked = torch.stack([padded[:, k : k + depth] for k in range(3)], 2).reshape(-1, 3, height, width)
            first = functional.conv2d(stacked, features[0].weight, padding=1)
            self.measure(self.norms[0], first)
            scale_1, shift_1 = self.norms[0].scale_and_shift()
            first = functional.leaky_relu(
                first * scale_1.reshape(1, -1, 1, 1).to(first.dtype) + shift_1.reshape(1, -1, 1, 1).to(first.dtype),
                features[2].negative_slope,
            )
            padded = functional.pad(first.reshape(batch, depth, -1, height, width), (0, 0, 0, 0, 0, 0, 1, 1))
            stacked = torch.cat([padded[:, k : k + depth] for k in range(3)], 2).reshape(batch * depth, -1, height, width)
            self.measure(self.norms[1], functional.conv2d(stacked, features[3].weight, padding=1))
        with torch.no_grad():
            return scale, stem_of(self.module)

    def measure(self, norm, flat):
        """A norm's statistics from a convolution's output with depth still in the batch."""
        values = flat.float()
        norm.mean, norm.variance = values.mean([0, 2, 3]), values.var([0, 2, 3], unbiased=False)

    def __call__(self, image, statistics):
        return self.features(self.kernel, image, statistics)

    def features(self, kernel, image, statistics, *extra):
        batch, _, height, width = image.shape
        out = self.torch.empty(batch, 32, height, width, device=self.device, dtype=self.torch.float16)
        sizes = self.torch.tensor(
            np.frombuffer(struct.pack("IIIf", batch, height, width, self.attention_bias), dtype=np.int32).copy(),
            device=self.device,
        )
        kernel.depth_stem(
            image,
            *statistics[1],
            out,
            sizes,
            *extra,
            threads=[(width + 15) // 16 * 128, (height + 7) // 8, batch],
            group_size=[128, 1, 1],
        )
        return out

    def tiles(self, block, statistics, gap):
        """Stem features for one tile or two sharing their convolution interior."""
        torch = self.torch
        if self.scale != statistics[0]:
            self.scale = statistics[0]
            self.table = scaled(torch.arange(256, dtype=torch.int32, device=self.device), self.scale)
        image = torch.from_numpy(np.ascontiguousarray(block, dtype=np.uint8)).to(self.device)[None]
        features = self.features(self.raw_kernel, image, statistics, self.table)
        if not gap:
            yield features
            return
        for offset in (0, gap):
            tile = features[:, :, :, offset : offset + TILE].contiguous()
            dims = struct.pack("IIIfIII", 1, TILE, TILE, self.attention_bias, TILE + gap, offset, int(offset == 0))
            sizes = torch.tensor(np.frombuffer(dims, dtype=np.int32).copy(), device=self.device)
            self.edge_kernel.depth_stem(
                image, *statistics[1], tile, sizes, self.table, threads=[128, TILE // 8, 1], group_size=[128, 1, 1]
            )
            yield tile


class TensorStem(DepthStem):
    """The stem on Metal 4 tensor operations: the kernel's arithmetic, its second convolution as matrix products."""

    def __init__(self, module, device="mps"):
        super().__init__(module, device)
        self.tensor_kernel = self.torch.mps.compile_shader((Path(__file__).parent / "tensor_stem.metal").read_text())
        self.rooms = collections.OrderedDict()
        features = self.module.features
        with self.torch.no_grad():
            self.raw_first = features[0].weight.float().reshape(16, 27).t().contiguous().half()
            self.raw_second = (
                features[3].weight.float().reshape(16, 3, 16, 3, 3).permute(1, 3, 4, 2, 0).reshape(27, 16, 16).contiguous().half()
            )

    def statistics_of(self, tiles, scale):
        """DepthStem's statistics, measured by the tensor kernels: each convolution's output summed per column on the GPU."""
        torch = self.torch
        with torch.inference_mode():
            if self.scale != scale:
                self.scale = scale
                self.table = scaled(torch.arange(256, dtype=torch.int32, device=self.device), scale)
            image = torch.from_numpy(np.ascontiguousarray(tiles, dtype=np.uint8)).to(self.device)
            batch, depth, height, width = image.shape
            volume, sizes = self.room(batch, height, width)
            columns, count = batch * height * width, batch * depth * height * width
            sums = torch.empty(columns, 2 * STEM_CHANNELS, device=self.device, dtype=torch.float32)
            self.tensor_kernel.first_statistics(
                image, self.raw_first, self.table, sums, sizes, threads=[width, height, batch], group_size=[32, 1, 1]
            )
            self.measured(self.norms[0], sums, count)
            scale_1, shift_1 = self.norms[0].scale_and_shift()
            first = self.module.features[0].weight.float().reshape(16, 3, 1, 3, 3)[:, :, 0] * scale_1.reshape(-1, 1, 1, 1)
            self.tensor_kernel.first_convolution(
                image,
                first.permute(1, 2, 3, 0).reshape(27, 16).contiguous().half(),
                shift_1.half(),
                self.table,
                volume,
                sizes,
                threads=[width, height, batch * DEPTH],
                group_size=[32, 1, 1],
            )
            sums = torch.empty(columns, 2 * STEM_CHANNELS, device=self.device, dtype=torch.float32)
            groups = -(-width // 32) * -(-batch * height // 4)
            self.tensor_kernel.second_statistics(
                volume, self.raw_second, sums, sizes, threads=[groups * 128, 1, 1], group_size=[128, 1, 1]
            )
            self.measured(self.norms[1], sums, count)
        with torch.no_grad():
            return scale, stem_of(self.module)

    def measured(self, norm, sums, count):
        """A norm's mean and variance from per-column sums and sums of squares."""
        total = sums.sum(0)
        norm.mean = total[:STEM_CHANNELS] / count
        norm.variance = total[STEM_CHANNELS:] / count - norm.mean * norm.mean

    def room(self, batch, height, width):
        """A zero-edged volume for the first convolution's output and the sizes of a batch of tiles, kept for its shape."""
        key = (batch, height, width)
        if key in self.rooms:
            self.rooms.move_to_end(key)
            return self.rooms[key]
        torch = self.torch
        padded = -(-width // 32) * 32 + 2
        sizes = struct.pack("IIIIf", batch, height, width, padded, self.attention_bias)
        while len(self.rooms) >= ROOMS:
            self.rooms.popitem(last=False)
        self.rooms[key] = (
            torch.zeros(batch, DEPTH + 2, height + 2, padded, STEM_CHANNELS, device=self.device, dtype=torch.float16),
            torch.tensor(np.frombuffer(sizes, dtype=np.int32).copy(), device=self.device),
        )
        return self.rooms[key]

    def tiles(self, block, statistics, gap):
        """Stem features for one tile, or for two cut from a block that holds both."""
        torch = self.torch
        if self.scale != statistics[0]:
            self.scale = statistics[0]
            self.table = scaled(torch.arange(256, dtype=torch.int32, device=self.device), self.scale)
        images = np.stack([block[:, :, :TILE], block[:, :, gap : gap + TILE]]) if gap else block[None]
        image = torch.from_numpy(np.ascontiguousarray(images, dtype=np.uint8)).to(self.device)
        batch, _, height, width = image.shape
        volume, sizes = self.room(batch, height, width)
        first, first_bias, second, second_bias, attention = statistics[1]
        out = torch.empty(batch, 2 * STEM_CHANNELS, height, width, device=self.device, dtype=torch.float16)
        self.tensor_kernel.first_convolution(
            image,
            first,
            first_bias,
            self.table,
            volume,
            sizes,
            threads=[width, height, batch * DEPTH],
            group_size=[32, 1, 1],
        )
        groups = -(-width // 32) * (batch * height // 4)
        self.tensor_kernel.second_convolution(
            volume, second, second_bias, attention, out, sizes, threads=[groups * 128, 1, 1], group_size=[128, 1, 1]
        )
        for k in range(batch):
            yield out[k : k + 1]


def tile_groups(starts, pair):
    """Keep blend order while pairing close tiles, starting with the last (often closest) pair."""
    if not pair:
        return [(x,) for x in starts]
    starts = starts.copy()
    groups = []
    while starts:
        x = starts.pop()
        groups.append((starts.pop(), x) if starts and x - starts[-1] <= PAIR_GAP else (x,))
    return groups[::-1]


class TileReader:
    def __init__(self, weights, device="mps"):
        import torch

        self.torch, self.device, self.weights = torch, device, weights
        self.stem = (TensorStem if STEM == "tensor" else DepthStem)(stock_model.taught_stem(weights, STEM_CHANNELS), device)
        self.engine = EngineBody(weights, load_body=lambda: self.body)

    @cached_property
    def gpu_engine(self):
        """The engine's own compiled body, read on the GPU."""
        return EngineBody(self.weights, units=CPU_AND_GPU)

    @cached_property
    def body(self):
        """The Torch body for uncompiled checkpoints and tiles smaller than the engine model."""
        return (
            stock_model.InkModel(names=tuple(stock_model.CHECKPOINTS)[:1], weights=[self.weights], device=self.device)
            .models[0]
            .model.model.requires_grad_(False)
            .network
        )

    def read(self, stack, fixed=False):
        """Probability of ink of a (depth, height, width) uint8 stack, tiles blended by a Hann window."""
        (read,) = self.maps([(stack, None if fixed else measured(stack))])
        if isinstance(read, Exception):
            raise read
        return read

    def maps(self, stacks):
        """Read each raw or prepared stack with its measurements, or None to reuse the last statistics."""
        tiles, stop, pending = queue.Queue(TILES_AHEAD), threading.Event(), collections.deque()
        with (
            ThreadPoolExecutor(1) as stem_thread,
            ThreadPoolExecutor(ENGINE_CALLS) as engine,
            ThreadPoolExecutor(GPU_CALLS) as gpu,
        ):
            lanes = {"E": (engine, self.engine), "G": (gpu, self.gpu_engine if "G" in LANES else None)}
            stems = stem_thread.submit(self.stems, stacks, tiles, stop)
            try:
                for sums, place, features, lane in iter(tiles.get, None):
                    if lane:
                        pool, body = lanes[lane]
                        features = pool.submit(self.ink_of, body, features)
                    pending.append((sums, place, features))
                    while pending and (len(pending) > PENDING or pending[0][1] is None):
                        yield from self.settled(pending)
                while pending:
                    yield from self.settled(pending)
            finally:
                stop.set()
            stems.result()

    def ink_of(self, body, features):
        return 1 / (1 + np.exp(-body(features)))

    def settled(self, pending):
        """The oldest tile added to its map, always in tile order, or the map once its stack has no tile left."""
        sums, place, read = pending.popleft()
        if place is None:
            yield sums.result()
        elif sums.failed is None:
            try:
                sums.add(*place, read.result() if isinstance(read, Future) else read)
            except Exception as error:
                sums.failed = error

    def stems(self, stacks, tiles, stop):
        """Every tile of every stack through the stem, in order, for the Neural Engine."""
        torch = self.torch

        def send(item):
            while not stop.is_set():
                try:
                    return tiles.put(item, timeout=1)
                except queue.Full:
                    continue
            return None

        try:
            with torch.inference_mode():
                for stack, measure in stacks:
                    _, height, width = stack.shape
                    sums = Sums(height, width)
                    try:
                        if measure is not None:
                            self.fixed = self.stem.statistics_of(*measure)
                        ty, tx = sums.tile_y, sums.tile_x
                        whole, coverage = stack if isinstance(stack, PreparedStack) else prepare_stack(stack)
                        on_engine = (ty, tx) == (SIDE, SIDE)
                        pair = on_engine and height >= ty
                        xs = tile_starts(width, tx)
                        for y in tile_starts(height, ty):
                            active = [x for x in xs if coverage[y : y + ty, x : x + tx].any()]
                            for group in tile_groups(active, pair):
                                x, last = group[0], group[-1]
                                block = whole[:, y : y + ty, x : last + tx]
                                h, w = block.shape[1:]
                                if h < ty or w < tx:
                                    block = np.pad(block, ((0, 0), (0, ty - h), (0, tx - w)))
                                for at, features in zip(group, self.stem.tiles(block, self.fixed, last - x)):
                                    place = (y, at, h, min(w, tx))
                                    if on_engine:
                                        send((sums, place, features.cpu().numpy(), lane_of(whole, y, at)))
                                    else:
                                        read = torch.sigmoid(self.body(features)["ink"].float())[0, 0].cpu().numpy()
                                        send((sums, place, read, None))
                    except Exception as error:
                        sums.failed = error
                    send((sums, None, None, None))
        finally:
            send(None)

    def fix_to(self, stack):
        self.fixed = self.stem.statistics(stack)


def lane_of(stack, y, x):
    """The unit that reads a 512 tile's body, fixed by where the tile is and what it holds, whatever the order of reads."""
    if len(LANES) == 1:
        return LANES
    _, height, width = stack.shape
    probe = sum(int(stack[DEPTH // 2, min(y + a, height - 1), min(x + b, width - 1)]) for a, b in PROBES)
    return LANES[(y // 128 + 3 * (x // 128) + probe) % len(LANES)]


def stem_of(stem):
    """The stem's weights with its norms folded in, as the Metal kernel reads them."""
    first, second = stem.features[1].scale_and_shift(), stem.features[4].scale_and_shift()
    w1 = stem.features[0].weight.float().reshape(16, 3, 1, 3, 3)[:, :, 0] * first[0].reshape(-1, 1, 1, 1)
    w2 = stem.features[3].weight.float().reshape(16, 3, 16, 3, 3) * second[0].reshape(-1, 1, 1, 1, 1)
    return (
        w1.permute(1, 2, 3, 0).reshape(27, 16).contiguous().half(),
        first[1].half(),
        w2.permute(1, 3, 4, 2, 0).reshape(27, 16, 16).contiguous().half(),
        second[1].half(),
        stem.attention_logits.weight.float().reshape(16).half(),
    )
