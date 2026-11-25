import torch

src = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
kernel void probe(device float *out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
#if defined(__HAVE_TENSOR__)
    out[tid] = float(__METAL_VERSION__) + 0.5f;
#else
    out[tid] = float(__METAL_VERSION__);
#endif
}
"""
try:
    k = torch.mps.compile_shader(src)
    out = torch.zeros(4, device="mps")
    k.probe(out)
    print("compiled; __METAL_VERSION__ (+0.5 if tensors available):", out.cpu().tolist())
except Exception as e:
    print("compile failed:", repr(e)[:1500])
