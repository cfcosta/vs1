#include <cuda_bf16.h>

// One 128-thread block per value head; all arithmetic between casts is F32.
template <typename Weight>
__device__ __forceinline__ void normalize_rms_gated(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ output,
    float eps, const Weight *__restrict__ weight
) {
    const unsigned int col = threadIdx.x;
    const unsigned int i = blockIdx.x * 128 + col;
    const float value = __bfloat162float(x[i]);
    float sum = __fmul_rn(value, value);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, mask));
    }
    __shared__ float warp_sums[4];
    const unsigned int lane = col % 32;
    if (lane == 0) warp_sums[col / 32] = sum;
    __syncthreads();
    sum = lane < 4 ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, mask));
    }
    // Keep Candle's separate mean, epsilon, sqrt and reciprocal results.
    // The F32 reduction tree can differ from Candle's fast_sum_f32.
    const float mean = __fmul_rn(sum, 1.f / 128.f);
    const float inv_rms = 1.0 / __fsqrt_rn(__fadd_rn(mean, eps));
    const float normalized = __bfloat162float(
        __float2bfloat16_rn(__fmul_rn(value, inv_rms)));
    float weighted = __fmul_rn(normalized, static_cast<float>(weight[col]));
    if constexpr (sizeof(Weight) == sizeof(__nv_bfloat16)) {
        weighted = __bfloat162float(__float2bfloat16_rn(weighted));
    }
    const float gate_value = __bfloat162float(gate[i]);
    const float silu = gate_value / __fadd_rn(1.f, expf(-gate_value));
    output[i] = __float2bfloat16_rn(__fmul_rn(weighted, silu));
}

extern "C" __global__ void normalize_rms_gated_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ output,
    float eps, const __nv_bfloat16 *__restrict__ weight
) {
    normalize_rms_gated(x, gate, output, eps, weight);
}

extern "C" __global__ void normalize_rms_gated_bf16_f32_weight(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ output,
    float eps, const float *__restrict__ weight
) {
    normalize_rms_gated(x, gate, output, eps, weight);
}
