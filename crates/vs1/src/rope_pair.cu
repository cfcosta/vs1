// Same BF16 rounding boundaries as candle-kernels' rope_thd.
__device__ __forceinline__ unsigned short mul(unsigned short a, unsigned short b) {
    unsigned short r;
    asm("{ .reg .b16 z; mov.b16 z, 0x8000; fma.rn.bf16 %0,%1,%2,z; }"
        : "=h"(r) : "h"(a), "h"(b));
    return r;
}
__device__ __forceinline__ unsigned short add(unsigned short a, unsigned short b) {
    unsigned short r;
    asm("{ .reg .b16 one; mov.b16 one, 0x3f80; fma.rn.bf16 %0,%1,one,%2; }"
        : "=h"(r) : "h"(a), "h"(b));
    return r;
}
__device__ __forceinline__ unsigned short sub(unsigned short a, unsigned short b) {
    unsigned short r;
    asm("{ .reg .b16 neg; mov.b16 neg, 0xbf80; fma.rn.bf16 %0,%2,neg,%1; }"
        : "=h"(r) : "h"(a), "h"(b));
    return r;
}
extern "C" __global__ void rope_pair_bf16(
    const unsigned short *q, const unsigned short *k,
    const unsigned short *cos, const unsigned short *sin,
    unsigned short *out, unsigned int count, unsigned int heads, unsigned int dim
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count / 2) return;
    unsigned int row = i / (dim / 2), col = i % (dim / 2);
    unsigned int a = row * dim + col, b = a + dim / 2;
    unsigned int cs = (row / heads) * (dim / 2) + col;
    unsigned short c = cos[cs], s = sin[cs];
    unsigned short q0 = q[a], q1 = q[b], k0 = k[a], k1 = k[b];
    out[a] = sub(mul(q0, c), mul(q1, s));
    out[b] = add(mul(q0, s), mul(q1, c));
    out[count + a] = sub(mul(k0, c), mul(k1, s));
    out[count + b] = add(mul(k0, s), mul(k1, c));
}
