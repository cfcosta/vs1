// One block per head, one thread per value column. Unrolling keeps each
// state column in registers; rounded intrinsics preserve the CPU loop's order.
extern "C" __global__ void apply_delta_rule_f32(
    const float *__restrict__ query, const float *__restrict__ key,
    const float *__restrict__ value, const float *__restrict__ beta,
    const float *__restrict__ decay, float *__restrict__ output,
    unsigned int seq, unsigned int heads, unsigned int value_dim
) {
    constexpr unsigned int key_dim = 128;
    // Reload keys for the update instead of keeping a second column in registers.
    __shared__ volatile float token_key[key_dim];
    __shared__ volatile float token_query[key_dim];
    float state[key_dim];
    #pragma unroll
    for (unsigned int i = 0; i < key_dim; ++i) state[i] = 0.0f;

    const unsigned int head = blockIdx.x;
    const unsigned int j = threadIdx.x;
    for (unsigned int t = 0; t < seq; ++t) {
        const unsigned int token_head = t * heads + head;
        for (unsigned int i = j; i < key_dim; i += blockDim.x) {
            token_key[i] = key[token_head * key_dim + i];
            token_query[i] = query[token_head * key_dim + i];
        }
        __syncthreads();

        const float token_decay = decay[token_head];
        #pragma unroll
        for (unsigned int i = 0; i < key_dim; ++i) {
            state[i] = __fmul_rn(state[i], token_decay);
        }
        float memory = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < key_dim; ++i) {
            memory = __fadd_rn(memory, __fmul_rn(token_key[i], state[i]));
        }
        const unsigned int offset = token_head * value_dim + j;
        const float delta = __fmul_rn(
            __fsub_rn(value[offset], memory), beta[token_head]);
        float out = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < key_dim; ++i) {
            state[i] = __fadd_rn(state[i], __fmul_rn(token_key[i], delta));
            out = __fadd_rn(out, __fmul_rn(token_query[i], state[i]));
        }
        output[offset] = out;
        // All columns must finish reading before loading the next token.
        __syncthreads();
    }
}
