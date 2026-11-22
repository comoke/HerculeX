"""Alternative GPU formulations of the stem's heavy convolution (16->16 channels, 3x3x3, 17x512x512): PyTorch MPS and MLX."""

import os
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

sync = torch.mps.synchronize
dev = "mps"


def bench(label, fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    print(f"{label:78s} {1000 * (time.perf_counter() - t) / n:7.2f} ms", flush=True)


with torch.inference_mode():
    for dtype in (torch.float16, torch.float32):
        name = "fp16" if dtype == torch.float16 else "fp32"
        x2 = torch.randn(17, 48, 512, 512, device=dev, dtype=dtype)
        w2 = torch.randn(16, 48, 3, 3, device=dev, dtype=dtype)
        bench(f"torch MPS conv2d {name} [17,48,512,512]->16 (folded, 30.8 GMAC)", lambda: F.conv2d(x2, w2, padding=1))
        x2c = x2.contiguous(memory_format=torch.channels_last)
        w2c = w2.contiguous(memory_format=torch.channels_last)
        bench(f"torch MPS conv2d {name} channels_last", lambda: F.conv2d(x2c, w2c, padding=1))
        x3 = torch.randn(1, 16, 17, 512, 512, device=dev, dtype=dtype)
        w3 = torch.randn(16, 16, 3, 3, 3, device=dev, dtype=dtype)
        bench(f"torch MPS conv3d {name} [1,16,17,512,512]->16 (30.8 GMAC)", lambda: F.conv3d(x3, w3, padding=1), n=10)
        x1 = torch.randn(17, 3, 512, 512, device=dev, dtype=dtype)
        w1 = torch.randn(16, 3, 3, 3, device=dev, dtype=dtype)
        bench(f"torch MPS conv2d {name} [17,3,512,512]->16 (first conv, 1.9 GMAC)", lambda: F.conv2d(x1, w1, padding=1))
        # body-like layers for reference
        xb = torch.randn(1, 32, 512, 512, device=dev, dtype=dtype)
        wb = torch.randn(32, 32, 3, 3, device=dev, dtype=dtype)
        bench(f"torch MPS conv2d {name} [1,32,512,512]->32 (2.4 GMAC)", lambda: F.conv2d(xb, wb, padding=1))
        xd = torch.randn(1, 256, 64, 64, device=dev, dtype=dtype)
        wd = torch.randn(256, 256, 3, 3, device=dev, dtype=dtype)
        bench(f"torch MPS conv2d {name} [1,256,64,64]->256 (2.4 GMAC)", lambda: F.conv2d(xd, wd, padding=1))
        del x2, w2, x2c, w2c, x3, w3, x1, w1

try:
    import mlx.core as mx

    for dtype in (mx.float16, mx.float32):
        name = "fp16" if dtype == mx.float16 else "fp32"
        x = mx.random.normal((17, 512, 512, 48)).astype(dtype)
        w = mx.random.normal((16, 3, 3, 48)).astype(dtype)
        mx.eval(x, w)

        def run():
            mx.eval(mx.conv2d(x, w, padding=1))

        for _ in range(3):
            run()
        t = time.perf_counter()
        for _ in range(20):
            run()
        print(f"{'MLX conv2d ' + name + ' [17,512,512,48]->16 (folded, 30.8 GMAC)':78s} {1000 * (time.perf_counter() - t) / 20:7.2f} ms", flush=True)
        x3 = mx.random.normal((1, 17, 512, 512, 16)).astype(dtype)
        w3 = mx.random.normal((16, 3, 3, 3, 16)).astype(dtype)
        mx.eval(x3, w3)

        def run3():
            mx.eval(mx.conv3d(x3, w3, padding=1))

        for _ in range(2):
            run3()
        t = time.perf_counter()
        for _ in range(10):
            run3()
        print(f"{'MLX conv3d ' + name + ' [1,17,512,512,16]->16 (30.8 GMAC)':78s} {1000 * (time.perf_counter() - t) / 10:7.2f} ms", flush=True)
        xb = mx.random.normal((1, 64, 64, 256)).astype(dtype)
        wb = mx.random.normal((256, 3, 3, 256)).astype(dtype)
        mx.eval(xb, wb)

        def runb():
            mx.eval(mx.conv2d(xb, wb, padding=1))

        for _ in range(3):
            runb()
        t = time.perf_counter()
        for _ in range(20):
            runb()
        print(f"{'MLX conv2d ' + name + ' [1,64,64,256]->256 (2.4 GMAC)':78s} {1000 * (time.perf_counter() - t) / 20:7.2f} ms", flush=True)
except Exception as error:  # noqa: BLE001
    print("mlx failed", repr(error))
