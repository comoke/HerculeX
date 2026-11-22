#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// The depth stem of depth_stem.metal in two passes, for GPUs with Metal 4 tensor operations: the first 3x3x3
// convolution into a zero-edged volume with channels last, then the second as nine accumulating matrix products per
// layer, with attention over depth and max over depth on the sums.

constant constexpr int C = 16, Z = 17;
constant constexpr int SM = 32, ROWS = 4;

struct Dims {
    uint batch, height, width, padded_width;
    float attention_bias;
};

kernel void first_convolution(device const uchar *image [[buffer(0)]], device const half *first [[buffer(1)]],
                              device const half *first_bias [[buffer(2)]], device const half *values [[buffer(3)]],
                              device half *out [[buffer(4)]], constant Dims &d [[buffer(5)]],
                              uint3 at [[thread_position_in_grid]]) {
    const int H = d.height, W = d.width;
    const int x = at.x, y = at.y, z = at.z % Z, b = at.z / Z;
    device const uchar *volume = image + size_t(b) * Z * H * W;
    float acc[C];
    for (int c = 0; c < C; c++) acc[c] = float(first_bias[c]);
    for (int dz = 0; dz < 3; dz++)
        for (int dy = 0; dy < 3; dy++)
            for (int dx = 0; dx < 3; dx++) {
                int zz = z + dz - 1, yy = y + dy - 1, xx = x + dx - 1;
                if (zz < 0 || zz >= Z || yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
                float v = float(values[volume[(size_t(zz) * H + yy) * W + xx]]);
                device const half *w = first + ((dz * 3 + dy) * 3 + dx) * C;
                for (int c = 0; c < C; c++) acc[c] += v * float(w[c]);
            }
    device half *o = out + (((size_t(b) * (Z + 2) + z + 1) * (H + 2) + y + 1) * d.padded_width + x + 1) * C;
    for (int c = 0; c < C; c++) {
        float a = acc[c];
        o[c] = half(a > 0 ? a : 0.01f * a);
    }
}

kernel void second_convolution(device half *a [[buffer(0)]], device half *second [[buffer(1)]],
                               device const half *second_bias [[buffer(2)]], device const half *attention [[buffer(3)]],
                               device half *out [[buffer(4)]], constant Dims &d [[buffer(5)]],
                               uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half staged[ROWS * SM * C];
    constexpr auto mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto description = matmul2d_descriptor(SM, C, 3 * C, false, false, false, mode);
    matmul2d<description, execution_simdgroup> product;

    const int H = d.height, W = d.width;
    const int across = (W + SM - 1) / SM;
    const int x0 = int(group.x % across) * SM;
    const int row = int(group.x / across) * ROWS + int(simd);
    const int y = row % H, b = row / H, held = min(b, int(d.batch) - 1);
    const size_t pitch_y = size_t(d.padded_width) * C, pitch_z = size_t(H + 2) * pitch_y;
    device half *volume = a + size_t(held) * (Z + 2) * pitch_z + size_t(x0) * C;
    threadgroup half *mine = staged + simd * SM * C;

    auto taps = tensor(volume, dextents<int, 2>{3 * C, SM}, array<int, 2>{1, C});
    auto weights = tensor(second, dextents<int, 2>{C, 3 * C}, array<int, 2>{1, C});
    auto sums = product.get_destination_cooperative_tensor<decltype(taps), decltype(weights), float>();
    auto narrow = product.get_destination_cooperative_tensor<decltype(taps), decltype(weights), half>();
    auto stage = tensor(mine, dextents<int, 2>{C, SM}, array<int, 2>{1, C});

    float best = -INFINITY, total = 0;
    float pooled[C], peak[C];
    for (int c = 0; c < C; c++) {
        pooled[c] = 0;
        peak[c] = -INFINITY;
    }

    for (int z = 0; z < Z; z++) {
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sums.get_capacity(); ++i) sums[i] = 0;
        for (int dz = 0; dz < 3; dz++)
            for (int dy = 0; dy < 3; dy++) {
                auto left = tensor(volume + (z + dz) * pitch_z + (y + dy) * pitch_y, dextents<int, 2>{3 * C, SM},
                                   array<int, 2>{1, C});
                auto right = tensor(second + (dz * 3 + dy) * 3 * C * C, dextents<int, 2>{C, 3 * C}, array<int, 2>{1, C});
                product.run(left, right, sums);
            }
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sums.get_capacity(); ++i) narrow[i] = half(sums[i]);
        narrow.store(stage);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float logit = d.attention_bias;
        float value[C];
        for (int c = 0; c < C; c++) {
            float v = float(mine[lane * C + c]) + float(second_bias[c]);
            v = v > 0 ? v : 0.01f * v;
            value[c] = v;
            logit += v * float(attention[c]);
            peak[c] = max(peak[c], v);
        }
        float top = max(best, logit), fade = exp(best - top), weight = exp(logit - top);
        total = total * fade + weight;
        for (int c = 0; c < C; c++) pooled[c] = pooled[c] * fade + weight * value[c];
        best = top;
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int x = x0 + int(lane);
    if (x < W && b < int(d.batch)) {
        device half *o = out + size_t(b) * 2 * C * H * W + size_t(y) * W + x;
        for (int c = 0; c < C; c++) {
            o[size_t(c) * H * W] = half(pooled[c] / total);
            o[size_t(C + c) * H * W] = half(peak[c]);
        }
    }
}

// Per column of a batch of tiles, the sum and the sum of squares over depth of the first convolution's output before
// its norm, each value rounded to half precision first: what the norm's statistics are taken from.
kernel void first_statistics(device const uchar *image [[buffer(0)]], device const half *first [[buffer(1)]],
                             device const half *values [[buffer(2)]], device float *out [[buffer(3)]],
                             constant Dims &d [[buffer(4)]], uint3 at [[thread_position_in_grid]]) {
    const int H = d.height, W = d.width;
    const int x = at.x, y = at.y, b = at.z;
    device const uchar *volume = image + size_t(b) * Z * H * W;
    float sum[C], squares[C];
    for (int c = 0; c < C; c++) sum[c] = squares[c] = 0;
    for (int z = 0; z < Z; z++) {
        float acc[C];
        for (int c = 0; c < C; c++) acc[c] = 0;
        for (int dz = 0; dz < 3; dz++)
            for (int dy = 0; dy < 3; dy++)
                for (int dx = 0; dx < 3; dx++) {
                    int zz = z + dz - 1, yy = y + dy - 1, xx = x + dx - 1;
                    if (zz < 0 || zz >= Z || yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
                    float v = float(values[volume[(size_t(zz) * H + yy) * W + xx]]);
                    device const half *w = first + ((dz * 3 + dy) * 3 + dx) * C;
                    for (int c = 0; c < C; c++) acc[c] += v * float(w[c]);
                }
        for (int c = 0; c < C; c++) {
            float v = float(half(acc[c]));
            sum[c] += v;
            squares[c] += v * v;
        }
    }
    device float *o = out + ((size_t(b) * H + y) * W + x) * 2 * C;
    for (int c = 0; c < C; c++) {
        o[c] = sum[c];
        o[C + c] = squares[c];
    }
}

// The same for the second convolution, over the first's output in its zero-edged volume.
kernel void second_statistics(device half *a [[buffer(0)]], device half *second [[buffer(1)]],
                              device float *out [[buffer(2)]], constant Dims &d [[buffer(3)]],
                              uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half staged[ROWS * SM * C];
    constexpr auto mode = matmul2d_descriptor::mode::multiply_accumulate;
    constexpr auto description = matmul2d_descriptor(SM, C, 3 * C, false, false, false, mode);
    matmul2d<description, execution_simdgroup> product;

    const int H = d.height, W = d.width, rows = int(d.batch) * H;
    const int across = (W + SM - 1) / SM;
    const int x0 = int(group.x % across) * SM;
    const int row = int(group.x / across) * ROWS + int(simd), held = min(row, rows - 1);
    const int y = held % H, b = held / H;
    const size_t pitch_y = size_t(d.padded_width) * C, pitch_z = size_t(H + 2) * pitch_y;
    device half *volume = a + size_t(b) * (Z + 2) * pitch_z + size_t(x0) * C;
    threadgroup half *mine = staged + simd * SM * C;

    auto taps = tensor(volume, dextents<int, 2>{3 * C, SM}, array<int, 2>{1, C});
    auto weights = tensor(second, dextents<int, 2>{C, 3 * C}, array<int, 2>{1, C});
    auto sums = product.get_destination_cooperative_tensor<decltype(taps), decltype(weights), float>();
    auto narrow = product.get_destination_cooperative_tensor<decltype(taps), decltype(weights), half>();
    auto stage = tensor(mine, dextents<int, 2>{C, SM}, array<int, 2>{1, C});

    float sum[C], squares[C];
    for (int c = 0; c < C; c++) sum[c] = squares[c] = 0;
    for (int z = 0; z < Z; z++) {
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sums.get_capacity(); ++i) sums[i] = 0;
        for (int dz = 0; dz < 3; dz++)
            for (int dy = 0; dy < 3; dy++) {
                auto left = tensor(volume + (z + dz) * pitch_z + (y + dy) * pitch_y, dextents<int, 2>{3 * C, SM},
                                   array<int, 2>{1, C});
                auto right = tensor(second + (dz * 3 + dy) * 3 * C * C, dextents<int, 2>{C, 3 * C}, array<int, 2>{1, C});
                product.run(left, right, sums);
            }
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < sums.get_capacity(); ++i) narrow[i] = half(sums[i]);
        narrow.store(stage);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < C; c++) {
            float v = float(mine[lane * C + c]);
            sum[c] += v;
            squares[c] += v * v;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int x = x0 + int(lane);
    if (x < W && row < rows) {
        device float *o = out + ((size_t(b) * H + y) * W + x) * 2 * C;
        for (int c = 0; c < C; c++) {
            o[c] = sum[c];
            o[C + c] = squares[c];
        }
    }
}
