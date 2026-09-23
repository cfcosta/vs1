#include <cuda_bf16.h>

// One thread per [time, channel] output or saved pre-convolution row.
extern "C" __global__ void convolve_causally_with_silu_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ initial_state,
    __nv_bfloat16 *__restrict__ output,
    unsigned int count, unsigned int channels, unsigned int kernel,
    unsigned int should_save_state
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int state_count = (kernel - 1) * channels;
    if (i >= count) {
        if (should_save_state && i - count < state_count) {
            // The padded sequence's last rows begin at the chunk length.
            output[i] = i >= state_count ? x[i - state_count]
                : (initial_state ? initial_state[i] : __float2bfloat16_rn(0.f));
        }
        return;
    }
    const unsigned int time = i / channels;
    const unsigned int channel = i % channels;
    float sum = 0.f;
    for (unsigned int tap = 0; tap < kernel; ++tap) {
        const unsigned int lag = kernel - 1 - tap;
        const float value = time >= lag
            ? __bfloat162float(x[(time - lag) * channels + channel])
            : (initial_state ? __bfloat162float(initial_state[(time + tap) * channels + channel]) : 0.f);
        const float scale = __bfloat162float(weight[channel * kernel + tap]);
        // Candle materializes each F32 product before adding taps in order.
        sum = __fadd_rn(sum, __fmul_rn(value, scale));
    }
    const __nv_bfloat16 value = __float2bfloat16_rn(sum);
    // Candle's silu_fwd uses BF16 hexp, addition and division, not F32 SiLU.
    output[i] = value / (static_cast<__nv_bfloat16>(1) + hexp(-value));
}
