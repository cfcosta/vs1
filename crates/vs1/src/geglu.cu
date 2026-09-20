// Preserve candle-kernels 0.11's BF16 normal-CDF rounding and both
// multiplications. Fusing memory traffic must not fuse the rounding steps.
#include <math.h>
__device__ __forceinline__ unsigned short bf16(float x) {
    unsigned short result;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ unsigned short mul_bf16(unsigned short a, unsigned short b) {
    unsigned short result;
    // Matches cuda_bf16.hpp's Ampere __hmul, including signed zero.
    asm("{ .reg .b16 z; mov.b16 z, 0x8000; fma.rn.bf16 %0, %1, %2, z; }"
        : "=h"(result) : "h"(a), "h"(b));
    return result;
}

extern "C" __global__ void geglu_bf16(
    const unsigned short *activation,
    const unsigned short *gate,
    unsigned short *output,
    unsigned int count
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        unsigned short a = activation[i];
        float x = __uint_as_float((unsigned int)a << 16);
        unsigned short cdf = bf16(normcdff(x));
        unsigned short gelu = mul_bf16(a, cdf);
        output[i] = mul_bf16(gelu, gate[i]);
    }
}

__device__ __forceinline__ unsigned int mul_pair(unsigned int a, unsigned int b) {
    unsigned int result;
    asm("{ .reg .b32 z; mov.b32 z, 0x80008000; fma.rn.bf16x2 %0, %1, %2, z; }"
        : "=r"(result) : "r"(a), "r"(b));
    return result;
}

extern "C" __global__ void geglu_bf16_pair(
    const unsigned short *activation,
    const unsigned short *gate,
    unsigned short *output,
    unsigned int count
) {
    unsigned int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i + 1 < count) {
        unsigned int a = reinterpret_cast<const unsigned int *>(activation)[i / 2];
        unsigned int g = reinterpret_cast<const unsigned int *>(gate)[i / 2];
        unsigned int c0 = bf16(normcdff(__uint_as_float((a & 0xffff) << 16)));
        unsigned int c1 = bf16(normcdff(__uint_as_float(a & 0xffff0000)));
        unsigned int gelu = mul_pair(a, c0 | (c1 << 16));
        reinterpret_cast<unsigned int *>(output)[i / 2] = mul_pair(gelu, g);
    } else if (i < count) {
        unsigned short a = activation[i];
        unsigned short cdf = bf16(normcdff(__uint_as_float((unsigned int)a << 16)));
        output[i] = mul_bf16(mul_bf16(a, cdf), gate[i]);
    }
}
