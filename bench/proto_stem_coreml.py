"""Prototype: the depth stem as one fused Core ML program on the GPU, norms as inputs. Speed per formulation and distance
of its features from the stock Metal kernel's. Run from a checkout root."""

import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
import coremltools as ct  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from ink.model import models, stock_model  # noqa: E402
from ink.model.reader import lane_reader  # noqa: E402
from ink.model.tiles import STEM_CHANNELS  # noqa: E402

cache, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
weights = models.chosen().weights
reader = lane_reader(weights)
kernel_stem = reader.stem

# the trained stem, float32 on CPU, raw (unfolded) weights
raw = stock_model.taught_stem(weights, STEM_CHANNELS)  # features[0], [3] are Folded modules holding conv2d-shaped weights
w1 = raw.features[0].weight.detach().float()  # [16, 3*1, 3, 3]  (out, dz*in, dy, dx)
w2 = raw.features[3].weight.detach().float()  # [16, 3*16, 3, 3]
att_w = raw.attention_logits.weight.detach().float().reshape(1, 16, 1, 1)
att_b = raw.attention_logits.bias.detach().float()
print("weights", tuple(w1.shape), tuple(w2.shape), flush=True)


class Conv3dStem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("k1", w1.reshape(16, 3, 1, 3, 3).permute(0, 2, 1, 3, 4).contiguous())  # [out, in, dz, dy, dx]
        self.register_buffer("k2", w2.reshape(16, 3, 16, 3, 3).permute(0, 2, 1, 3, 4).contiguous())
        self.register_buffer("aw", att_w.reshape(1, 16, 1, 1, 1))
        self.register_buffer("ab", att_b)

    def forward(self, image, scale_1, shift_1, scale_2, shift_2):
        x = image[:, None]  # [1, 1, 17, h, w]
        a = F.leaky_relu(F.conv3d(x, self.k1, padding=1) * scale_1[..., None] + shift_1[..., None], 0.01)
        b = F.leaky_relu(F.conv3d(a, self.k2, padding=1) * scale_2[..., None] + shift_2[..., None], 0.01)
        share = torch.softmax(F.conv3d(b, self.aw, self.ab), dim=2)
        return torch.cat([(b * share).sum(2), b.amax(2)], 1)


class FoldedStem(torch.nn.Module):
    """Depth as batch: each layer's three depth neighbours as channels."""

    def __init__(self):
        super().__init__()
        self.register_buffer("k1", w1.clone())
        self.register_buffer("k2", w2.clone())
        self.register_buffer("aw", att_w.clone())
        self.register_buffer("ab", att_b)

    def forward(self, image, scale_1, shift_1, scale_2, shift_2):
        x = image.permute(1, 0, 2, 3)  # [17, 1, h, w]
        z = x.shape[0]
        xp = F.pad(x, (0, 0, 0, 0, 0, 0, 1, 1))
        a = F.conv2d(torch.cat([xp[0:z], xp[1 : z + 1], xp[2 : z + 2]], 1), self.k1, padding=1)
        a = F.leaky_relu(a * scale_1 + shift_1, 0.01)
        ap = F.pad(a, (0, 0, 0, 0, 0, 0, 1, 1))
        b = F.conv2d(torch.cat([ap[0:z], ap[1 : z + 1], ap[2 : z + 2]], 1), self.k2, padding=1)
        b = F.leaky_relu(b * scale_2 + shift_2, 0.01)
        share = torch.softmax(F.conv2d(b, self.aw, self.ab), dim=0)
        return torch.cat([(b * share).sum(0, keepdim=True), b.amax(0, keepdim=True)], 1)


def converted(name, module):
    path = out_dir / f"stem-{name}.mlpackage"
    if not path.exists():
        image = torch.zeros(1, 17, 512, 512)
        norms = [torch.ones(1, 16, 1, 1), torch.zeros(1, 16, 1, 1)] * 2
        with torch.inference_mode():
            traced = torch.jit.trace(module.eval(), (image, *norms), strict=False)
        names = ("image", "scale_1", "shift_1", "scale_2", "shift_2")
        program = ct.convert(
            traced,
            inputs=[ct.TensorType(name=n, shape=a.shape, dtype=np.float16) for n, a in zip(names, (image, *norms))],
            outputs=[ct.TensorType(name="features", dtype=np.float16)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
            minimum_deployment_target=ct.target.macOS15,
        )
        program.save(str(path))
    return path


def norms_of(fixed_module_norms):
    parts = []
    for norm in fixed_module_norms:
        scale, shift = norm.scale_and_shift()
        parts += [scale.float().cpu().numpy().reshape(1, 16, 1, 1).astype(np.float16), shift.float().cpu().numpy().reshape(1, 16, 1, 1).astype(np.float16)]
    return parts


pieces = [pickle.load(open(f, "rb")) for f in sorted(cache.glob("*.pkl"))[:12:3]]
rows = []
with torch.inference_mode():
    for piece in pieces:
        (values, coverage), _ = piece["stacks"][-1]
        measure = next(m for _, m in piece["stacks"] if m is not None)
        if values.shape[1] < 512 or values.shape[2] < 512:
            continue
        fixed = kernel_stem.statistics_of(*measure)
        with torch.no_grad():
            norms = norms_of(kernel_stem.norms)
        low, high, centre, spread = fixed[0]
        grey = ((np.clip(np.arange(256, dtype=np.float32), low, high).astype(np.float16) - np.float16(centre)) / np.float16(spread)).astype(np.float16)
        y, x = (values.shape[1] - 512) // 2, (values.shape[2] - 512) // 2
        block = np.ascontiguousarray(values[:, y : y + 512, x : x + 512])
        want = next(iter(kernel_stem.tiles(block, fixed, 0))).float().cpu().numpy()
        rows.append((grey[block][None], norms, want))

for name, module in (("conv3d", Conv3dStem()), ("folded", FoldedStem())):
    try:
        path = converted(name, module)
    except Exception as error:  # noqa: BLE001
        print(name, "conversion failed:", repr(error)[:300], flush=True)
        continue
    for units_name, units in (("GPU", ct.ComputeUnit.CPU_AND_GPU), ("Neural Engine", ct.ComputeUnit.CPU_AND_NE)):
        try:
            model = ct.models.MLModel(str(path), compute_units=units)

            def run(k):
                image, norms, _ = rows[k % len(rows)]
                return model.predict({"image": image, "scale_1": norms[0], "shift_1": norms[1], "scale_2": norms[2], "shift_2": norms[3]})["features"]

            for _ in range(3):
                run(0)
            line = f"Core ML stem {name:7s} on {units_name:13s}"
            for workers in (1, 2):
                with ThreadPoolExecutor(workers) as pool:
                    t = time.perf_counter()
                    list(pool.map(run, range(20)))
                    line += f" {workers} at once {1000 * (time.perf_counter() - t) / 20:6.2f} ms/tile"
            worst, mean = 0.0, 0.0
            for k, (_, _, want) in enumerate(rows):
                d = np.abs(run(k).astype(np.float32) - want)
                worst, mean = max(worst, float(d.max())), mean + float(d.mean()) / len(rows)
            print(f"{line} | vs stock kernel features: max abs {worst:.4f} mean abs {mean:.6f}", flush=True)
        except Exception as error:  # noqa: BLE001
            print(name, units_name, "failed:", repr(error)[:300], flush=True)
