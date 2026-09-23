__device__ __forceinline__ float as_float(unsigned short value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

// One 256-thread block per row, including strided query rows between gates.
extern "C" __global__ void normalize_rms_bf16(
    const unsigned short *__restrict__ x,
    const unsigned short *__restrict__ weight,
    unsigned short *__restrict__ output,
    unsigned int cols, unsigned int row_stride, float mean_scale, float eps
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    float sum = 0.f;
    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        const float value = as_float(x[row * row_stride + col]);
        // Candle materializes the F32 squares before reducing them.
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, mask));
    }
    __shared__ float warp_sums[8];
    const unsigned int lane = tid % 32;
    if (lane == 0) warp_sums[tid / 32] = sum;
    __syncthreads();
    sum = lane < 8 ? warp_sums[lane] : 0.f;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, mask));
    }
    // Preserve mean, epsilon, sqrt and reciprocal as separate F32 results.
    // Candle's recipg(float) spells the reciprocal as 1.0 / value.
    const float mean = __fmul_rn(sum, mean_scale);
    const float inv_rms = 1.0 / __fsqrt_rn(__fadd_rn(mean, eps));
    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        const float normalized = __fmul_rn(
            as_float(x[row * row_stride + col]), inv_rms);
        const float scale = __fadd_rn(as_float(weight[col]), 1.f);
        const float value = __fmul_rn(normalized, scale);
        unsigned short result;
        asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(result) : "f"(value));
        output[row * cols + col] = result;
    }
}
