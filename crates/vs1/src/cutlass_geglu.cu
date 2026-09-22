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
