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
