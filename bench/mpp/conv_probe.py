"""Fused MPP conv: 3x3x3, 16->16 channels over a zero-haloed NHWC half volume, 9 accumulating matmuls (K=48) per tile
into a cooperative float tensor, stored once as half. Checked against numpy, then timed on 17x512x512 (30.8 GMAC)."""
import time
import numpy as np
import torch

SRC = r'''
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Dims { uint depth, height, width; };

kernel void conv(device half *a [[buffer(0)]], device half *w [[buffer(1)]], device half *d [[buffer(2)]],
                 constant Dims &dims [[buffer(3)]],
                 uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr auto mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto desc = matmul2d_descriptor(SM, 16, 48, false, false, false, mode);
    matmul2d<desc, execution_simdgroup> op;
    const int W = dims.width, H = dims.height;
    const int tiles_x = W / SM;
    const int x0 = int(group.x % tiles_x) * SM;
    const int row = int(group.x / tiles_x) * SIMDS + int(simd);
    const int y = row % H, z = row / H;
    const size_t pitch_x = 16, pitch_y = size_t(W + 2) * 16, pitch_z = size_t(H + 2) * pitch_y;
    auto first_a = tensor(a, dextents<int, 2>{48, SM}, array<int, 2>{1, 16});
    auto first_w = tensor(w, dextents<int, 2>{16, 48}, array<int, 2>{1, 16});
    auto sum = op.get_destination_cooperative_tensor<decltype(first_a), decltype(first_w), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < sum.get_capacity(); ++i) sum[i] = 0;
    for (int dz = 0; dz < 3; dz++)
        for (int dy = 0; dy < 3; dy++) {
            auto mA = tensor(a + (z + dz) * pitch_z + (y + dy) * pitch_y + x0 * pitch_x, dextents<int, 2>{48, SM}, array<int, 2>{1, 16});
            auto mW = tensor(w + (dz * 3 + dy) * 768, dextents<int, 2>{16, 48}, array<int, 2>{1, 16});
            op.run(mA, mW, sum);
        }
    auto narrow = op.get_destination_cooperative_tensor<decltype(first_a), decltype(first_w), half>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < sum.get_capacity(); ++i) narrow[i] = half(sum[i]);
    auto mD = tensor(d + ((size_t(z) * H + y) * W + x0) * 16, dextents<int, 2>{16, SM}, array<int, 2>{1, 16});
    narrow.store(mD);
}
'''
sync = torch.mps.synchronize

def run(sm, simds, a, w, d, depth, height, width):
    k = torch.mps.compile_shader(SRC.replace("SIMDS", str(simds)).replace("SM", str(sm)))
    dims = torch.tensor(np.array([depth, height, width], dtype=np.int32), device="mps")
    groups = (width // sm) * (depth * height // simds)
    def go():
        k.conv(a, w, d, dims, threads=[groups * simds * 32, 1, 1], group_size=[simds * 32, 1, 1])
    return go

rng = np.random.default_rng(0)
Z, H, W = 3, 64, 64
a_np = np.zeros((Z + 2, H + 2, W + 2, 16), dtype=np.float16)
a_np[1:-1, 1:-1, 1:-1] = (rng.standard_normal((Z, H, W, 16)) * 0.5).astype(np.float16)
w_np = (rng.standard_normal((3, 3, 3, 16, 16)) * 0.2).astype(np.float16)
want = np.zeros((Z, H, W, 16))
a64, w64 = a_np.astype(np.float64), w_np.astype(np.float64)
for dz in range(3):
    for dy in range(3):
        for dx in range(3):
            want += a64[dz : dz + Z, dy : dy + H, dx : dx + W] @ w64[dz, dy, dx]
a, w = torch.from_numpy(a_np).to("mps"), torch.from_numpy(w_np).to("mps")
d = torch.zeros(Z, H, W, 16, device="mps", dtype=torch.float16)
run(32, 4, a, w, d, Z, H, W)()
got = d.cpu().numpy()
print(f"check: max abs err {np.abs(got.astype(np.float64) - want).max():.5f} (values up to {np.abs(want).max():.1f}), equal to correctly rounded half: {np.mean(got == want.astype(np.float16)):.5f}", flush=True)

Z, H, W = 17, 512, 512
a = torch.randn(Z + 2, H + 2, W + 2, 16, device="mps", dtype=torch.float16)
w = torch.randn(3, 3, 3, 16, 16, device="mps", dtype=torch.float16)
d = torch.zeros(Z, H, W, 16, device="mps", dtype=torch.float16)
for sm, simds in ((32, 4), (32, 8), (64, 4), (16, 4), (32, 2), (32, 16), (64, 8), (128, 4)):
    try:
        go = run(sm, simds, a, w, d, Z, H, W)
        for _ in range(2):
            go()
        sync(); t = time.perf_counter()
        for _ in range(5):
            go()
        sync(); dt = (time.perf_counter() - t) / 5
        print(f"SM={sm:3d} simdgroups per threadgroup={simds:2d}: {1000 * dt:6.2f} ms per tile = {30.8 / dt / 1000:5.2f} TMAC/s", flush=True)
    except Exception as e:
        print(f"SM={sm} simds={simds}: FAILED {repr(e)[:500]}", flush=True)
