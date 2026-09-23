// Candle's BF16 `broadcast_add(bias)` (FMA with 1.0 on Ampere) and its
// `relu` (__hmax_nan against zero), two BF16 values per instruction.
__device__ __forceinline__ unsigned int add_pair(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("{ .reg .b32 one; mov.b32 one, 0x3f803f80; fma.rn.bf16x2 %0,%1,one,%2; }"
        : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ unsigned int relu_pair(unsigned int a) {
    unsigned int r;
    asm("{ .reg .b32 zero; mov.b32 zero, 0; max.NaN.bf16x2 %0,%1,zero; }"
        : "=r"(r) : "r"(a));
    return r;
}

// `x` is (rows, cols) and `bias` is (cols); both views start on 8-value
// boundaries and cols is a multiple of 8. One uint4 (8 values) per thread.
template <bool relu>
__device__ __forceinline__ void bias_act(
    const uint4 *__restrict__ x, const uint4 *__restrict__ bias,
    uint4 *__restrict__ out, unsigned int count8, unsigned int cols8
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count8) return;
    const uint4 a = x[i], b = bias[i % cols8];
    uint4 r = make_uint4(
        add_pair(a.x, b.x), add_pair(a.y, b.y),
        add_pair(a.z, b.z), add_pair(a.w, b.w));
    if (relu) {
        r = make_uint4(relu_pair(r.x), relu_pair(r.y), relu_pair(r.z), relu_pair(r.w));
    }
    out[i] = r;
}

extern "C" __global__ void bias_add_bf16(
    const uint4 *__restrict__ x, const uint4 *__restrict__ bias,
    uint4 *__restrict__ out, unsigned int count8, unsigned int cols8
) {
    bias_act<false>(x, bias, out, count8, cols8);
}

extern "C" __global__ void bias_relu_bf16(
    const uint4 *__restrict__ x, const uint4 *__restrict__ bias,
    uint4 *__restrict__ out, unsigned int count8, unsigned int cols8
) {
    bias_act<true>(x, bias, out, count8, cols8);
}
