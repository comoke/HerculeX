"""The ink network's 2-D body compiled for the Neural Engine in half precision."""

import collections
import copy
import ctypes
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ink.model.stock_model import checkpoint_hash
from ink.model.tiles import DEPTH, STEM_CHANNELS
from runner.compiled_kernels import C_FLAGS, load
from storage.data_roots import NEURAL_ENGINE_BUILDS

COMPILED = NEURAL_ENGINE_BUILDS
SIDE = 512
CHANNELS = 32
CPU_AND_GPU = 1
CPU_AND_ENGINE = 3
ENGINE_CALLS = 2


def package_of(weights):
    """Where the compiled network of some weights is kept."""
    return COMPILED / f"{checkpoint_hash(weights).hexdigest()[:16]}-{SIDE}.mlpackage"


def compile_body(body, path):
    """A torch body as a Core ML program, written whole so two readers may race."""
    import coremltools as ct
    import torch

    class Logits(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, features):
            return self.network(features)["ink"]

    model = Logits(copy.deepcopy(body).float().cpu().eval())
    example = torch.zeros(1, CHANNELS, SIDE, SIDE)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    program = ct.convert(
        traced,
        inputs=[ct.TensorType(name="features", shape=example.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="ink", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.stem}-{os.getpid()}.part.mlpackage")
    program.save(str(part))
    if path.exists():
        shutil.rmtree(part)
    else:
        part.rename(path)


def engine_calls():
    calls = load(
        "engine-calls",
        [Path(__file__).parent / "engine_calls.m"],
        (*C_FLAGS, "-fobjc-arc", "-framework", "Foundation", "-framework", "CoreML"),
    )
    calls.hx_engine_open.restype = ctypes.c_void_p
    calls.hx_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    calls.hx_engine_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    calls.hx_engine_whole.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    return calls


class EngineBody:
    """Logits of one 512-pixel tile of stem features on the Neural Engine, called from C so a thread waiting on it holds nothing."""

    def __init__(self, weights, body=None, units=CPU_AND_ENGINE, *, load_body=None):
        path = package_of(weights)
        if not path.exists():
            compile_body(body if load_body is None else load_body(), path)
        self.calls = engine_calls()
        self.engine = self.calls.hx_engine_open(
            str(path).encode(), str(path.with_suffix(".mlmodelc")).encode(), units, False
        )
        if not self.engine:
            raise RuntimeError(f"no engine from {path}")

    def __call__(self, features):
        """Features as a (1, 32, 512, 512) float16 array."""
        logits = np.empty((SIDE, SIDE), dtype=np.float16)
        failed = self.calls.hx_engine_read(self.engine, features.ctypes.data, CHANNELS, SIDE, logits.ctypes.data)
        if failed:
            raise RuntimeError(f"engine call failed: {failed}")
        return logits.astype(np.float32)


def whole_network(stem, body):
    """Depth stem and body as one torch module taking a tile's 17 layers and the stack's norms."""
    import torch
    from torch.nn import functional

    depth, width = DEPTH, STEM_CHANNELS
    first = stem.features[0].weight.detach().reshape(width, 3, 1, 3, 3).permute(1, 0, 2, 3, 4)
    second = stem.features[3].weight.detach().reshape(width, 3, width, 3, 3).permute(1, 0, 2, 3, 4)
    attention = stem.attention_logits.weight.detach().reshape(1, width, 1, 1)

    class Whole(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = body
            self.register_buffer("first", first.repeat(1, depth, 1, 1, 1).contiguous())
            self.register_buffer("second", second.repeat(1, depth, 1, 1, 1).contiguous())
            self.register_buffer("attention", attention.repeat(depth, 1, 1, 1).contiguous())
            self.register_buffer("attention_bias", stem.attention_logits.bias.detach().repeat(depth))

        def across_depth(self, x, weights, channels):
            blank = torch.zeros_like(x[:, :channels])
            lower, upper = torch.cat([blank, x[:, :-channels]], 1), torch.cat([x[:, channels:], blank], 1)
            return sum(
                functional.conv2d(part, weight, padding=1, groups=depth)
                for part, weight in zip((lower, x, upper), weights)
            )

        def forward(self, image, scale_1, shift_1, scale_2, shift_2):
            spread = [v.repeat(1, depth, 1, 1) for v in (scale_1, shift_1, scale_2, shift_2)]
            a = functional.leaky_relu(self.across_depth(image, self.first, 1) * spread[0] + spread[1], 0.01)
            b = functional.leaky_relu(self.across_depth(a, self.second, width) * spread[2] + spread[3], 0.01)
            share = torch.softmax(functional.conv2d(b, self.attention, self.attention_bias, groups=depth), dim=1)
            pooled, peak = 0, None
            for z in range(depth):
                part = b[:, width * z : width * (z + 1)]
                pooled = pooled + part * share[:, z : z + 1]
                peak = part if peak is None else torch.maximum(peak, part)
            return self.body(torch.cat([pooled, peak], 1))["ink"]

    return Whole().eval()


def compile_whole(stem, body, path):
    """The whole network as a Core ML program, written whole so two readers may race."""
    import coremltools as ct
    import torch

    model = whole_network(stem, body)
    image = torch.zeros(1, DEPTH, SIDE, SIDE)
    norms = [torch.ones(1, STEM_CHANNELS, 1, 1), torch.zeros(1, STEM_CHANNELS, 1, 1)] * 2
    with torch.inference_mode():
        traced = torch.jit.trace(model, (image, *norms), strict=False)
    names = ("image", "scale_1", "shift_1", "scale_2", "shift_2")
    program = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=a.shape, dtype=np.float16) for n, a in zip(names, (image, *norms))],
        outputs=[ct.TensorType(name="ink", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.stem}-{os.getpid()}.part.mlpackage")
    program.save(str(part))
    if path.exists():
        shutil.rmtree(part)
    else:
        part.rename(path)


class EngineReader:
    """TileReader's read with no GPU: the stack's norms measured on CPU, every tile on the Neural Engine."""

    def __init__(self, weights):
        import torch

        from ink.model import stock_model, tiles

        self.tiles = tiles
        self.torch = torch
        net = (
            stock_model.InkModel(names=tuple(stock_model.CHECKPOINTS)[:1], weights=[weights], device="cpu")
            .models[0]
            .model.model.requires_grad_(False)
        )
        self.stem, body = net.depth_fusion.float().eval(), net.network.float().eval()
        path = package_of(weights).with_name(package_of(weights).stem + "-whole.mlpackage")
        if not path.exists():
            compile_whole(self.stem, body, path)
        self.calls = engine_calls()
        self.engine = self.calls.hx_engine_open(
            str(path).encode(), str(path.with_suffix(".mlmodelc")).encode(), CPU_AND_ENGINE, True
        )
        if not self.engine:
            raise RuntimeError(f"no engine from {path}")
        self.norms = tiles.fixed_norms(torch, self.stem)
        self.measuring = threading.Lock()

    def fix_to(self, stack):
        """The raw scale and the stem's norms of a stack, as a patch's own norms would take them."""
        torch = self.torch
        tiles = self.tiles.measured_tiles(stack)
        scale = self.tiles.raw_scale(tiles)
        with self.measuring:
            for norm in self.norms:
                norm.measuring = True
            with torch.inference_mode():
                self.stem(torch.from_numpy(self.scaled(np.stack(tiles), scale, np.float32))[:, None])
            for norm in self.norms:
                norm.measuring = False
            with torch.no_grad():
                parts = [
                    p.numpy().reshape(1, STEM_CHANNELS, 1, 1).astype(np.float16)
                    for norm in self.norms
                    for p in norm.scale_and_shift()
                ]
        fixed = scale, dict(zip(("scale_1", "shift_1", "scale_2", "shift_2"), parts))
        self.fixed = fixed
        return fixed

    def scaled(self, raw, scale, dtype=np.float16):
        low, high, centre, spread = scale
        return ((np.clip(raw.astype(np.float32), low, high) - centre) / spread).astype(dtype)

    def logits(self, image, norms):
        """Logits of one (17, 512, 512) float16 tile as a (512, 512) float32 array."""
        logits = np.empty((SIDE, SIDE), dtype=np.float16)
        failed = self.calls.hx_engine_whole(
            self.engine, image.ctypes.data, DEPTH, norms.ctypes.data, STEM_CHANNELS, SIDE, logits.ctypes.data
        )
        if failed:
            raise RuntimeError(f"engine call failed: {failed}")
        return logits.astype(np.float32)

    def read(self, stack, fixed=False):
        """Probability of ink of a (depth, height, width) uint8 stack, tiles blended by a Hann window, as TileReader reads it."""
        scale, norms = self.fixed if fixed else self.fix_to(stack)
        _, height, width = stack.shape
        blend = np.outer(np.hanning(SIDE + 2)[1:-1], np.hanning(SIDE + 2)[1:-1]).astype(np.float32)
        total, weight = np.zeros((2, height, width), dtype=np.float32)
        block = np.zeros((DEPTH, SIDE, SIDE), dtype=np.uint8)
        grey = self.scaled(np.arange(256, dtype=np.uint8), scale)
        norms = np.concatenate([part.ravel() for part in norms.values()])
        step = self.tiles.STEP
        pending = collections.deque()

        def settle():
            y, x, h, w, call = pending.popleft()
            read = 1 / (1 + np.exp(-call.result()))
            total[y : y + h, x : x + w] += read[:h, :w] * blend[:h, :w]
            weight[y : y + h, x : x + w] += blend[:h, :w]

        with ThreadPoolExecutor(ENGINE_CALLS) as engine:
            for y in sorted({*range(0, max(height - SIDE, 0) + 1, step), max(height - SIDE, 0)}):
                for x in sorted({*range(0, max(width - SIDE, 0) + 1, step), max(width - SIDE, 0)}):
                    part = stack[:, y : y + SIDE, x : x + SIDE]
                    if not part.any():
                        continue
                    h, w = part.shape[1:]
                    block[:] = 0
                    block[:, :h, :w] = part
                    pending.append((y, x, h, w, engine.submit(self.logits, grey[block], norms)))
                    if len(pending) > ENGINE_CALLS:
                        settle()
            while pending:
                settle()
        return total / np.maximum(weight, 1e-6)
