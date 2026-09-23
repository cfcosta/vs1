// One block per head, one thread per value column. Unrolling keeps each
// state column in registers; rounded intrinsics preserve the CPU loop's order.
extern "C" __global__ void apply_delta_rule_f32(
    const float *__restrict__ query, const float *__restrict__ key,
    const float *__restrict__ value, const float *__restrict__ beta,
    const float *__restrict__ decay, const float *__restrict__ initial_state,
    float *__restrict__ output, unsigned int seq, unsigned int heads,
    unsigned int value_dim, unsigned int should_save_state
) {
    constexpr unsigned int key_dim = 128;
    // Reload keys for the update instead of keeping a second column in registers.
    __shared__ volatile float token_key[key_dim];
    __shared__ volatile float token_query[key_dim];
    float state[key_dim];
    const unsigned int head = blockIdx.x;
    const unsigned int j = threadIdx.x;
    #pragma unroll
    for (unsigned int i = 0; i < key_dim; ++i) {
        state[i] = initial_state ? initial_state[(head * key_dim + i) * value_dim + j] : 0.0f;
    }
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
    if (should_save_state) {
        #pragma unroll
        for (unsigned int i = 0; i < key_dim; ++i) {
            output[seq * heads * value_dim + (head * key_dim + i) * value_dim + j] = state[i];
        }
    }
}

// Eight adjacent lanes share a state column; each block handles 32 columns.
// The shuffle reductions change the CPU loop's summation order.
extern "C" __global__ void __launch_bounds__(256) apply_delta_rule_parallel_f32(
    const float *__restrict__ query, const float *__restrict__ key,
    const float *__restrict__ value, const float *__restrict__ beta,
    const float *__restrict__ decay, const float *__restrict__ initial_state,
    float *__restrict__ output, unsigned int seq, unsigned int heads,
    unsigned int value_dim, unsigned int should_save_state
) {
    constexpr unsigned int key_dim = 128;
    constexpr unsigned int lanes_per_column = 8;
    constexpr unsigned int rows_per_lane = key_dim / lanes_per_column;
    __shared__ float token_key[key_dim];
    __shared__ float token_query[key_dim];
    float state[rows_per_lane];
    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x % lanes_per_column;
    const unsigned int j = blockIdx.y * 32 + threadIdx.x / lanes_per_column;
    #pragma unroll
    for (unsigned int i = 0; i < rows_per_lane; ++i) {
        const unsigned int row = i * lanes_per_column + lane;
        state[i] = initial_state ? initial_state[(head * key_dim + row) * value_dim + j] : 0.0f;
    }
    for (unsigned int t = 0; t < seq; ++t) {
        const unsigned int token_head = t * heads + head;
        if (threadIdx.x < key_dim) {
            token_key[threadIdx.x] = key[token_head * key_dim + threadIdx.x];
            token_query[threadIdx.x] = query[token_head * key_dim + threadIdx.x];
        }
        __syncthreads();

        const float token_decay = decay[token_head];
        float memory = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < rows_per_lane; ++i) {
            state[i] = __fmul_rn(state[i], token_decay);
            memory = __fadd_rn(memory, __fmul_rn(
                token_key[i * lanes_per_column + lane], state[i]));
        }
        #pragma unroll
        for (unsigned int offset = lanes_per_column / 2; offset > 0; offset /= 2) {
            memory = __fadd_rn(memory, __shfl_xor_sync(
                0xffffffff, memory, offset, lanes_per_column));
        }
        const unsigned int offset = token_head * value_dim + j;
        const float delta = __fmul_rn(
            __fsub_rn(value[offset], memory), beta[token_head]);
        float out = 0.0f;
        #pragma unroll
        for (unsigned int i = 0; i < rows_per_lane; ++i) {
            const unsigned int row = i * lanes_per_column + lane;
            state[i] = __fadd_rn(state[i], __fmul_rn(token_key[row], delta));
            out = __fadd_rn(out, __fmul_rn(token_query[row], state[i]));
        }
        #pragma unroll
        for (unsigned int offset = lanes_per_column / 2; offset > 0; offset /= 2) {
            out = __fadd_rn(out, __shfl_xor_sync(
                0xffffffff, out, offset, lanes_per_column));
        }
        if (lane == 0) output[offset] = out;
        __syncthreads();
    }
    if (should_save_state) {
        #pragma unroll
        for (unsigned int i = 0; i < rows_per_lane; ++i) {
            const unsigned int row = i * lanes_per_column + lane;
            output[seq * heads * value_dim + (head * key_dim + row) * value_dim + j] = state[i];
        }
    }
}
