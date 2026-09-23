// CUTLASS GeGLU: preserve Candle's BF16 GeGLU rounding in the epilogue.
#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cuda_runtime.h>
using B = cutlass::bfloat16_t;
__device__ unsigned short mul_bf16(unsigned short a,unsigned short b) {
    unsigned short c;
    asm("{ .reg .b16 z; mov.b16 z, 0x8000; fma.rn.bf16 %0,%1,%2,z; }" : "=h"(c) : "h"(a),"h"(b));
    return c;
}
struct RoundedGeGlu {
    using ElementOutput=B; using ElementSource=B; using ElementAccumulator=float;
    using ElementCompute=float; using ElementScalar=float; using ElementC=B; using ElementD=B;
    static int const kCount=8;
    using FragmentOutput=cutlass::Array<B,kCount>;
    using FragmentSource=FragmentOutput;
    using FragmentAccumulator=cutlass::Array<float,kCount>;
    using FragmentCompute=FragmentAccumulator;
    struct Params {};
    CUTLASS_HOST_DEVICE RoundedGeGlu(Params const &) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const {return true;}
    CUTLASS_HOST_DEVICE void set_k_partition(int,int) {} // split-K disabled
    CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& a,FragmentSource const& gate) const {
        FragmentOutput out;
        for(int i=0;i<kCount;++i) {
            B rounded(a[i]);
            B cdf(normcdff(float(rounded)));
            out[i]=B::bitcast(mul_bf16(mul_bf16(rounded.raw(),cdf.raw()),B(gate[i]).raw()));
        }
        return out;
    }
    CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& a) const {
        FragmentSource ones;for(int i=0;i<kCount;++i)ones[i]=B(1.0f);return (*this)(a,ones);
    }
};
template<class Epilogue>
using Gemm=cutlass::gemm::device::Gemm<B,cutlass::layout::RowMajor,B,cutlass::layout::ColumnMajor,
    B,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<16,8,16>,Epilogue,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3>;
template<class Op>
int run(void const*x,void const*w,void const*g,void*out,int m,int n,int k,void*stream,typename Op::EpilogueOutputOp::Params epilogue) {
    typename Op::Arguments args({m,n,k},{static_cast<B const*>(x),k},{static_cast<B const*>(w),k},
        {static_cast<B const*>(g),n},{static_cast<B*>(out),n},epilogue,1);
    Op op;
    auto valid=op.can_implement(args); if(valid!=cutlass::Status::kSuccess)return int(valid);
    return int(op(args,nullptr,static_cast<cudaStream_t>(stream)));
}
extern "C" int vs1_cutlass_geglu(void const*x,void const*w,void const*g,void*out,int m,int n,int k,void*stream,int fused) {
    if(fused)return run<Gemm<RoundedGeGlu>>(x,w,g,out,m,n,k,stream,{});
    using Plain=cutlass::epilogue::thread::LinearCombination<B,8,float,float>;
    return run<Gemm<Plain>>(x,w,g,out,m,n,k,stream,{1.0f,0.0f});
}

// Dual GEMM (vendored CUTLASS example 45): one kernel computes both the
// activation and gate products from shared A tiles. Each accumulator is
// rounded to BF16 on its own, exactly like two separate GEMM outputs, and
// only the rounded GeGLU product is stored.
#include "cutlass_dual/device/dual_gemm.h"
struct RoundedGeGluPair {
    using ElementOutput=B; using ElementAccumulator=B; using ElementCompute=float;
    static int const kCount=8;
    using FragmentOutput=cutlass::Array<B,kCount>;
    using FragmentAccumulator=cutlass::Array<B,kCount>;
    struct Params {};
    CUTLASS_HOST_DEVICE RoundedGeGluPair(Params const &) {}
    CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& act,FragmentAccumulator const& gate) const {
        FragmentOutput out;
        CUTLASS_PRAGMA_UNROLL
        for(int i=0;i<kCount;++i) {
            B a=act[i], g=gate[i];
            B cdf(normcdff(float(a)));
            out[i]=B::bitcast(mul_bf16(mul_bf16(a.raw(),cdf.raw()),g.raw()));
        }
        return out;
    }
};
// Round each F32 accumulator to BF16 with no scaling and no source.
using RoundOnly=cutlass::epilogue::thread::LinearCombination<B,8,float,float,
    cutlass::epilogue::thread::ScaleType::Nothing>;
using DualGeGlu=cutlass::gemm::device::DualGemm<B,cutlass::layout::RowMajor,B,
    cutlass::layout::ColumnMajor,cutlass::layout::ColumnMajor,B,cutlass::layout::RowMajor,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,64,32>,cutlass::gemm::GemmShape<64,32,32>,
    cutlass::gemm::GemmShape<16,8,16>,RoundOnly,RoundOnly,RoundedGeGluPair,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3,false,false,false>;
extern "C" int vs1_cutlass_dual_geglu(void const*x,void const*wa,void const*wg,void*out,int m,int n,int k,void*stream) {
    auto A=static_cast<B const*>(x);
    auto D=static_cast<B*>(out);
    // C is never read (RoundOnly needs no source); unstored D0/D1 are null.
    typename DualGeGlu::Arguments args(cutlass::gemm::DualGemmMode::kGemm,{m,n,k},
        {A,k},{static_cast<B const*>(wa),k},{D,n},{nullptr,n},
        {static_cast<B const*>(wg),k},{D,n},{nullptr,n},{D,n});
    DualGeGlu op;
    auto valid=op.can_implement(args); if(valid!=cutlass::Status::kSuccess)return int(valid);
    valid=op.initialize(args,nullptr,static_cast<cudaStream_t>(stream));
    if(valid!=cutlass::Status::kSuccess)return 100+int(valid);
    return int(op(static_cast<cudaStream_t>(stream)));
}
