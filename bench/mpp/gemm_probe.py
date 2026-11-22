"""MPP matmul2d inside torch.mps.compile_shader: semantics check against numpy, then throughput on the stem's shape.
D[M x 16] = A[M x 48] @ B[48 x 16], rows of A overlapping in memory with stride LD (16 = NHWC conv taps along x)."""
import sys, time
import numpy as np
import torch

SRC = r'''
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Dims { uint rows; uint ld_a; };

kernel void gemm(device half *a [[buffer(0)]], device half *b [[buffer(1)]], device DTYPE *d [[buffer(2)]],
                 constant Dims &dims [[buffer(3)]],
                 uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr auto desc = matmul2d_descriptor(SM, 16, 48);
    matmul2d<desc, execution_simdgroup> op;
    const int m0 = (group.x * SIMDS + simd) * SM;
    const int ld = dims.ld_a;
    auto mA = tensor(a + size_t(m0) * ld, dextents<int, 2>{48, SM}, array<int, 2>{1, ld});
    auto mB = tensor(b, dextents<int, 2>{16, 48}, array<int, 2>{1, 16});
    auto mD = tensor(d + size_t(m0) * 16, dextents<int, 2>{16, SM}, array<int, 2>{1, 16});
    op.run(mA, mB, mD);
}
'''
sync = torch.mps.synchronize
def build(sm, simds, dtype):
    src = SRC.replace("SIMDS", str(simds)).replace("SM", str(sm)).replace("DTYPE", dtype)
    return torch.mps.compile_shader(src)

def run(kernel, a, b, d, rows, ld, sm, simds):
    dims = torch.tensor(np.array([rows, ld], dtype=np.int32), device="mps")
    groups = rows // (sm * simds)
    kernel.gemm(a, b, d, dims, threads=[groups * simds * 32, 1, 1], group_size=[simds * 32, 1, 1])

rng = np.random.default_rng(0)
for ld in (48, 16):
    for dtype, np_t, t_t in (("float", np.float32, torch.float32), ("half", np.float16, torch.float16)):
        sm, simds, rows = 32, 4, 32 * 4 * 64
        a_np = (rng.standard_normal(rows * ld + 64) * 0.5).astype(np.float16)
        b_np = (rng.standard_normal((48, 16)) * 0.5).astype(np.float16)
        want = np.stack([a_np[m * ld : m * ld + 48].astype(np.float64) @ b_np.astype(np.float64) for m in range(rows)])
        try:
            k = build(sm, simds, dtype)
            a, b = torch.from_numpy(a_np).to("mps"), torch.from_numpy(b_np).to("mps")
            d = torch.zeros(rows, 16, device="mps", dtype=t_t)
            run(k, a, b, d, rows, ld, sm, simds)
            got = d.cpu().numpy().astype(np.float64)
            err = np.abs(got - want)
            print(f"ld_a={ld} dest={dtype}: max abs err {err.max():.5f} (values up to {np.abs(want).max():.1f}), exact-rounded matches {np.mean(got.astype(np_t) == want.astype(np_t)):.4f}", flush=True)
        except Exception as e:
            print(f"ld_a={ld} dest={dtype}: FAILED {repr(e)[:700]}", flush=True)

# throughput on the stem conv shape: 17*512*512 pixels, 9 GEMMs of K=48 (30.8 GMAC)
rows = 17 * 512 * 512
for sm, simds in ((32, 4), (32, 1), (64, 4), (16, 4), (32, 8)):
    try:
        k = build(sm, simds, "float")
        a = torch.randn(rows * 16 + 64, device="mps", dtype=torch.float16)
        b = torch.randn(48, 16, device="mps", dtype=torch.float16)
        d = torch.zeros(rows, 16, device="mps", dtype=torch.float32)
        usable = rows // (sm * simds) * (sm * simds)
        for _ in range(2):
            run(k, a, b, d, usable, 16, sm, simds)
        sync(); t = time.perf_counter()
        for _ in range(9):
            run(k, a, b, d, usable, 16, sm, simds)
        sync(); dt = time.perf_counter() - t
        print(f"SM={sm} simdgroups/threadgroup={simds}: 9 GEMMs (30.8 GMAC, float dest, no accumulate) {1000 * dt:.2f} ms = {30.8 / dt / 1000:.2f} TMAC/s", flush=True)
    except Exception as e:
        print(f"SM={sm} simds={simds}: FAILED {repr(e)[:400]}", flush=True)
