// Match Candle's BF16 add followed by its layernorm (width <= 1024).
__device__ __forceinline__ unsigned short add_bf16(unsigned short a, unsigned short b) {
    unsigned short r;
    asm("{ .reg .b16 one; mov.b16 one, 0x3f80; fma.rn.bf16 %0,%1,one,%2; }"
        : "=h"(r) : "h"(a), "h"(b));
    return r;
}
__device__ __forceinline__ float as_float(unsigned short a) {
    return __uint_as_float(static_cast<unsigned int>(a) << 16);
}
extern "C" __global__ void residual_norm_bf16(
    const unsigned short *x, const unsigned short *y, const unsigned short *weight,
    unsigned short *out, unsigned int count, int ncols, float eps
) {
    const unsigned int start = blockIdx.x * ncols;
    const int tid = threadIdx.x;
    float2 mean_var = make_float2(0.f, 0.f);
    for (int col = tid; col < ncols; col += blockDim.x) {
        const unsigned short sum = add_bf16(x[start + col], y[start + col]);
        out[start + col] = sum;
        const float xi = as_float(sum);
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }
    // Same lane traversal and XOR shuffle order as candle-kernels/reduce.cu.
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        mean_var.x += __shfl_xor_sync(0xffffffff, mean_var.x, mask, 32);
        mean_var.y += __shfl_xor_sync(0xffffffff, mean_var.y, mask, 32);
    }
    if (blockDim.x > 32) {
        __shared__ float2 sums[32];
        const int warp = tid / 32, lane = tid % 32;
        if (lane == 0) sums[warp] = mean_var;
        __syncthreads();
        mean_var = sums[lane];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            mean_var.x += __shfl_xor_sync(0xffffffff, mean_var.x, mask, 32);
            mean_var.y += __shfl_xor_sync(0xffffffff, mean_var.y, mask, 32);
        }
    }
    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);
    for (int col = tid; col < ncols; col += blockDim.x) {
        const float lhs = (as_float(out[start + col]) - mean) * inv_std;
        // The trunk LayerNorm has a persistent +0 bias. Preserve its FMA.
        const float value = __fmaf_rn(lhs, as_float(weight[col]), 0.f);
        unsigned short result;
        asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(result) : "f"(value));
        out[count + start + col] = result;
    }
}

__device__ __forceinline__ unsigned int add_pair(unsigned int a, unsigned int b) {
    unsigned int r;
    asm("{ .reg .b32 one; mov.b32 one, 0x3f803f80; fma.rn.bf16x2 %0,%1,one,%2; }"
        : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Constant bounds keep both arrays in registers rather than local memory.
template <int mask>
__device__ __forceinline__ void fold(float *s, float *q) {
#pragma unroll
    for (int i = 0; i < mask; ++i) {
        s[i] += s[i + mask];
        q[i] += q[i + mask];
    }
}

// Width 1024 only, one warp per row. Candle reduces that row with 1024
// threads: element t = 32 * w + l sits in lane l of warp w, lanes are
// XOR-folded (16, 8, 4, 2, 1), then the 32 warp sums are XOR-folded the
// same way. Here lane w holds Candle's warp w as 32 contiguous values, so
// the first fold happens in registers with identical operand pairs and the
// second is the same shuffle sequence. Float addition is commutative, so
// every intermediate matches bit for bit while loads become 16 bytes wide.
extern "C" __global__ void residual_norm_1024_bf16(
    const uint4 *__restrict__ x, const uint4 *__restrict__ y,
    const uint4 *__restrict__ weight, uint4 *__restrict__ out,
    unsigned int rows, float eps
) {
    const unsigned int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const unsigned int lane = threadIdx.x;
    // Four uint4 (32 BF16 values) per lane, 128 uint4 per row.
    const unsigned int base = row * 128 + lane * 4;
    unsigned int sum[16];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const uint4 a = x[base + v], b = y[base + v];
        sum[4 * v] = add_pair(a.x, b.x);
        sum[4 * v + 1] = add_pair(a.y, b.y);
        sum[4 * v + 2] = add_pair(a.z, b.z);
        sum[4 * v + 3] = add_pair(a.w, b.w);
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        out[base + v] = make_uint4(
            sum[4 * v], sum[4 * v + 1], sum[4 * v + 2], sum[4 * v + 3]);
    }
    float s[32], q[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const float xi = __uint_as_float(
            i & 1 ? sum[i / 2] & 0xffff0000u : sum[i / 2] << 16);
        // Candle's per-thread loop runs once from zero: keep 0 + xi and
        // the contracted 0 + xi * xi exactly.
        float2 mean_var = make_float2(0.f, 0.f);
        mean_var.x += xi;
        mean_var.y += xi * xi;
        s[i] = mean_var.x;
        q[i] = mean_var.y;
    }
    fold<16>(s, q);
    fold<8>(s, q);
    fold<4>(s, q);
    fold<2>(s, q);
    fold<1>(s, q);
    float2 mean_var = make_float2(s[0], q[0]);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        mean_var.x += __shfl_xor_sync(0xffffffff, mean_var.x, mask, 32);
        mean_var.y += __shfl_xor_sync(0xffffffff, mean_var.y, mask, 32);
    }
    // Candle's PTX leaves `mean_var.y / ncols - mean * mean` unrounded,
    // and ptxas fuses it into FFMA(-mean, mean, y / ncols) in its SASS
    // (this matters when mean * mean alone would overflow or cancel).
    // Spell that out with explicit rounding so ptxas cannot pick another
    // fusion here. Scaling by 2^-10 is exactly division by 1024.
    const float mean = __fmul_rn(mean_var.x, 0.0009765625f);
    const float var = __fmaf_rn(-mean, mean, __fmul_rn(mean_var.y, 0.0009765625f));
    const float inv_std = rsqrtf(__fadd_rn(var, eps));
    const unsigned int count4 = rows * 128;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const uint4 w4 = weight[lane * 4 + v];
        const unsigned int w[4] = {w4.x, w4.y, w4.z, w4.w};
        unsigned int packed[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned int pair = sum[4 * v + j];
            unsigned short lo, hi;
            const float l0 = __fmul_rn(__fsub_rn(__uint_as_float(pair << 16), mean), inv_std);
            const float l1 = __fmul_rn(__fsub_rn(__uint_as_float(pair & 0xffff0000u), mean), inv_std);
            const float v0 = __fmaf_rn(l0, __uint_as_float(w[j] << 16), 0.f);
            const float v1 = __fmaf_rn(l1, __uint_as_float(w[j] & 0xffff0000u), 0.f);
            asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(lo) : "f"(v0));
            asm("cvt.rn.bf16.f32 %0,%1;" : "=h"(hi) : "f"(v1));
            packed[j] = lo | (static_cast<unsigned int>(hi) << 16);
        }
        out[count4 + base + v] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}
