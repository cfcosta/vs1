//! Guarded output-tile adjustment for short packed BF16 encoder products.
use std::{cell::RefCell, collections::BTreeMap, ffi::c_void};

use anyhow::{Result, ensure};
use candle_core::{
    CpuStorage,
    CudaStorage,
    CustomOp2,
    DType,
    Layout,
    Shape,
    Tensor,
    backend::BackendStorage,
    cuda_backend::{
        CudaStorageSlice,
        cudarc::{
            cublas::sys::cublasOperation_t,
            cublaslt::{result as lt, sys},
            driver::{
                CudaEvent,
                CudaSlice,
                DevicePtr,
                DevicePtrMut,
                sys::CUevent_flags,
            },
        },
    },
};
use candle_nn::Linear;

const WORKSPACE: usize = 64 * 1024 * 1024;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

struct Engine {
    handle: sys::cublasLtHandle_t,
    workspace: RefCell<CudaSlice<u8>>,
    completion: CudaEvent,
    last_thread: std::cell::Cell<Option<std::thread::ThreadId>>,
}
impl Engine {
    fn new(xs: &Tensor) -> Result<Self> {
        let stream = xs.device().as_cuda_device()?.cuda_stream();
        stream.context().bind_to_thread()?;
        // SAFETY: workspace is scratch memory, never read by Rust.
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE)? };
        Ok(Self {
            handle: lt::create_handle()?,
            workspace: RefCell::new(workspace),
            completion: stream
                .context()
                .new_event(Some(CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
            last_thread: std::cell::Cell::new(None),
        })
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        // A model can be dropped on a different host thread from its last use.
        // Finish the last scratch-buffer use before its originating stream frees it.
        if self.last_thread.get().is_some() {
            self.completion.synchronize().unwrap();
        }
        // SAFETY: this is the unique owner of this handle.
        unsafe { lt::destroy_handle(self.handle).unwrap() };
    }
}

struct Matrix(sys::cublasLtMatrixLayout_t);
impl Matrix {
    fn new(rows: usize, cols: usize) -> Result<Self> {
        Ok(Self(lt::create_matrix_layout(
            sys::cudaDataType::CUDA_R_16BF,
            rows as u64,
            cols as u64,
            rows as i64,
        )?))
    }
}
impl Drop for Matrix {
    fn drop(&mut self) {
        // SAFETY: unique descriptor ownership.
        unsafe { lt::destroy_matrix_layout(self.0).unwrap() };
    }
}

struct Desc(sys::cublasLtMatmulDesc_t);
impl Drop for Desc {
    fn drop(&mut self) {
        // SAFETY: unique descriptor ownership.
        unsafe { lt::destroy_matmul_desc(self.0).unwrap() };
    }
}
struct Pref(sys::cublasLtMatmulPreference_t);
impl Drop for Pref {
    fn drop(&mut self) {
        // SAFETY: unique descriptor ownership.
        unsafe { lt::destroy_matmul_pref(self.0).unwrap() };
    }
}

struct Plan {
    shape: [usize; 3], // tokens, output, input
    desc: Desc,
    a: Matrix,
    b: Matrix,
    c: Matrix,
}
impl Plan {
    fn new([m, n, k]: [usize; 3]) -> Result<Self> {
        let desc = Desc(lt::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType::CUDA_R_32F,
        )?);
        let trans = cublasOperation_t::CUBLAS_OP_T;
        // Same column-major interpretation as Candle: W^T * X, yielding
        // row-major X * W^T. No padding, fusion, batching, or dtype changes.
        unsafe {
            lt::set_matmul_desc_attribute(desc.0, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&trans as *const cublasOperation_t).cast(), size_of_val(&trans))?;
        }
        Ok(Self {
            shape: [m, n, k],
            desc,
            a: Matrix::new(k, n)?,
            b: Matrix::new(k, m)?,
            c: Matrix::new(n, m)?,
        })
    }
    fn algorithms(
        &self,
        engine: &Engine,
    ) -> Result<Vec<sys::cublasLtMatmulHeuristicResult_t>> {
        let pref = Pref(lt::create_matmul_pref()?);
        let mut results = vec![
            std::mem::MaybeUninit::<
                sys::cublasLtMatmulHeuristicResult_t,
            >::uninit();
            64
        ];
        let mut count = 0;
        // SAFETY: live descriptors, correctly typed attribute, and space for
        // all 64 results requested. Preference and descriptors outlive the call.
        unsafe {
            lt::set_matmul_pref_attribute(pref.0, sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&WORKSPACE as *const usize).cast(), size_of::<usize>())?;
            sys::cublasLtMatmulAlgoGetHeuristic(
                engine.handle,
                self.desc.0,
                self.a.0,
                self.b.0,
                self.c.0,
                self.c.0,
                pref.0,
                64,
                results.as_mut_ptr().cast(),
                &mut count,
            )
            .result()?;
        }
        // SAFETY: cuBLASLt initialized exactly the returned count of records.
        let mut results: Vec<_> = results
            .into_iter()
            .take(count as usize)
            .map(|r| unsafe { r.assume_init() })
            .collect();
        results
            .retain(|r| r.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        Ok(results)
    }
}

struct Op<'a> {
    engine: &'a Engine,
    plan: &'a Plan,
    algo: &'a sys::cublasLtMatmulAlgo_t,
}
impl CustomOp2 for Op<'_> {
    fn name(&self) -> &'static str {
        "vs1_bf16_retiled_gemm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("CUDA only")
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let [m, n, k] = self.plan.shape;
        let (CudaStorageSlice::BF16(xb), CudaStorageSlice::BF16(wb)) =
            (&x.slice, &w.slice)
        else {
            candle_core::bail!("BF16 only")
        };
        if !xl.is_contiguous()
            || !wl.is_contiguous()
            || xl.dims() != [m, k]
            || wl.dims() != [n, k]
        {
            candle_core::bail!("invalid GEMM layouts")
        }
        let dev = x.device();
        let stream = dev.cuda_stream();
        stream.context().bind_to_thread().map_err(err)?;
        let thread = std::thread::current().id();
        if self
            .engine
            .last_thread
            .get()
            .is_some_and(|last| last != thread)
        {
            // CUDA's per-thread default stream has a different queue on each
            // host thread, even when the CudaStream Arc is shared.
            stream.wait(&self.engine.completion).map_err(err)?;
        }
        let xv = xb.slice(xl.start_offset()..xl.start_offset() + m * k);
        let wv = wb.slice(wl.start_offset()..wl.start_offset() + n * k);
        // SAFETY: beta=0 and GEMM initializes every output element.
        let mut output = unsafe { dev.alloc(m * n)? };
        let (xp, _x_guard) = xv.device_ptr(&stream);
        let (wp, _w_guard) = wv.device_ptr(&stream);
        let (out, _out_guard) = output.device_ptr_mut(&stream);
        let mut workspace = self.engine.workspace.borrow_mut();
        let (scratch, _scratch_guard) = workspace.device_ptr_mut(&stream);
        // SAFETY: shapes checked above, buffers and stream guards live through
        // launch, and beta=0 means the aliased C/D input need not be initialized.
        unsafe {
            lt::matmul(
                self.engine.handle,
                self.plan.desc.0,
                (&1.0f32 as *const f32).cast(),
                (&0.0f32 as *const f32).cast(),
                wp as *const c_void,
                self.plan.a.0,
                xp as *const c_void,
                self.plan.b.0,
                out as *const c_void,
                self.plan.c.0,
                out as *mut c_void,
                self.plan.c.0,
                self.algo,
                scratch as *mut c_void,
                WORKSPACE,
                stream.cu_stream().cast(),
            )
            .map_err(err)?;
        }
        self.engine.completion.record(&stream).map_err(err)?;
        self.engine.last_thread.set(Some(thread));
        drop(_out_guard);
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(output),
                device: dev.clone(),
            },
            (m, n).into(),
        ))
    }
}

fn output(
    engine: &Engine,
    plan: &Plan,
    algo: &sys::cublasLtMatmulAlgo_t,
    xs: &Tensor,
    linear: &Linear,
) -> Result<Tensor> {
    Ok(xs.apply_op2_no_bwd(linear.weight(), &Op { engine, plan, algo })?)
}

fn config(algo: &sys::cublasLtMatmulAlgo_t) -> Result<[u32; 9]> {
    use sys::cublasLtMatmulAlgoConfigAttributes_t::*;
    let mut values = [0; 9];
    for (value, attr) in values.iter_mut().zip([
        CUBLASLT_ALGO_CONFIG_ID,
        CUBLASLT_ALGO_CONFIG_TILE_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        CUBLASLT_ALGO_CONFIG_STAGES_ID,
        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    ]) {
        let mut written = 0;
        // SAFETY: each queried attribute is a 32-bit integer.
        unsafe {
            sys::cublasLtMatmulAlgoConfigGetAttribute(
                algo,
                attr,
                (value as *mut u32).cast(),
                size_of::<u32>(),
                &mut written,
            )
            .result()?;
        }
        ensure!(written == size_of::<u32>());
    }
    // These two attributes have 16-bit storage, unlike the preceding seven.
    for (value, attr) in values[7..].iter_mut().zip([
        CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
        CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
    ]) {
        let mut small = 0u16;
        let mut written = 0;
        // SAFETY: the buffer and size match the documented uint16_t type.
        unsafe {
            sys::cublasLtMatmulAlgoConfigGetAttribute(
                algo,
                attr,
                (&mut small as *mut u16).cast(),
                size_of::<u16>(),
                &mut written,
            )
            .result()?;
        }
        ensure!(written == size_of::<u16>());
        *value = u32::from(small);
    }
    Ok(values)
}

struct CachedPlan {
    plan: Plan,
    algorithm: sys::cublasLtMatmulAlgo_t,
}
struct State {
    plans: BTreeMap<[usize; 3], Option<CachedPlan>>,
    engine: Engine,
}
// SAFETY: Retile's mutex exclusively owns these host descriptors, handle and
// scratch allocation. Calls bind their CUDA context. The completion event
// orders scratch reuse across per-thread default streams and before teardown.
unsafe impl Send for State {}

#[derive(Default)]
pub(crate) struct Retile {
    state: std::sync::Mutex<Option<State>>,
    disabled: std::sync::atomic::AtomicBool,
}
#[cfg(test)]
pub(crate) static REFERENCE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
static HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

impl Retile {
    pub(crate) fn linear(
        &self,
        xs: &Tensor,
        linear: &Linear,
    ) -> candle_core::Result<Tensor> {
        #[cfg(test)]
        if REFERENCE.load(std::sync::atomic::Ordering::Relaxed) {
            return xs.apply(linear);
        }
        // Most calls have another shape. Reject those before inspecting the
        // device, weights, alignment, or shared state.
        if xs.rank() != 2 {
            return xs.apply(linear);
        }
        let (m, k) = xs.dims2()?;
        if !(129..=136).contains(&m) || ![1024, 2624].contains(&k) {
            return xs.apply(linear);
        }
        if self.disabled.load(std::sync::atomic::Ordering::Relaxed)
            || xs.dtype() != DType::BF16
            || !xs.device().is_cuda()
            || linear.bias().is_some()
            || !xs.is_contiguous()
            || !linear.weight().is_contiguous()
            || !xs.layout().start_offset().is_multiple_of(128)
            || !linear.weight().layout().start_offset().is_multiple_of(128)
        {
            return xs.apply(linear);
        }
        let n = linear.weight().dim(0)?;
        if n != 1024 {
            return xs.apply(linear);
        }
        let attempt = || -> Result<Option<Tensor>> {
            let mut guard = self
                .state
                .lock()
                .map_err(|e| anyhow::anyhow!("retile lock: {e}"))?;
            if guard.is_none() {
                let stream = xs.device().as_cuda_device()?.cuda_stream();
                stream.context().bind_to_thread()?;
                // Algorithm IDs and the original cuBLAS heuristic are not
                // portable. Restrict this measured exception to the tested GPU
                // and library release; other devices keep Candle's dispatcher.
                if stream.context().name()? != "NVIDIA GeForce RTX 3080 Ti"
                    || unsafe { sys::cublasLtGetVersion() } != 120901
                {
                    self.disabled
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    return Ok(None);
                }
                *guard = Some(State {
                    plans: BTreeMap::new(),
                    engine: Engine::new(xs)?,
                });
            }
            let state = guard.as_mut().unwrap();
            let shape = [m, n, k];
            if !state.plans.contains_key(&shape) {
                let plan = Plan::new(shape)?;
                let algorithms = plan.algorithms(&state.engine)?;
                // Tile 18 is 128x64, tile 15 is 64x64. Keep the algorithm,
                // K partitions, reduction, stages, swizzle, custom option,
                // inner shape and cluster shape.
                // Changing those can change the BF16 result bits.
                let chosen = if let Some(first) = algorithms.first() {
                    let mut target = config(&first.algo)?;
                    if target[..7] == [21, 18, 4, 4, 12, 0, 0] {
                        target[1] = 15;
                        algorithms.iter().find_map(|a| match config(&a.algo) {
                            Ok(candidate) if candidate == target => {
                                Some(a.algo)
                            }
                            _ => None,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };
                state.plans.insert(
                    shape,
                    chosen.map(|algorithm| CachedPlan { plan, algorithm }),
                );
            }
            let Some(cached) = state.plans.get(&shape).unwrap() else {
                return Ok(None);
            };
            let result = output(
                &state.engine,
                &cached.plan,
                &cached.algorithm,
                xs,
                linear,
            )?;
            #[cfg(test)]
            HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Some(result))
        };
        match attempt().map_err(err)? {
            Some(output) => Ok(output),
            None => xs.apply(linear),
        }
    }
}

#[cfg(test)]
pub(crate) mod bench {
    use std::time::Instant;

    use anyhow::Context;
    use candle_core::cuda_backend::cudarc::driver::{
        LaunchConfig,
        PushKernelArg,
    };
    use serde_json::{Value, json};

    use super::*;
    fn bits(xs: &Tensor) -> Result<Vec<u32>> {
        Ok(xs
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    struct Mismatches;
    impl CustomOp2 for Mismatches {
        fn name(&self) -> &'static str {
            "gemm_bit_mismatches"
        }
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> candle_core::Result<(CpuStorage, Shape)> {
            candle_core::bail!("CUDA only")
        }
        fn cuda_fwd(
            &self,
            a: &CudaStorage,
            al: &Layout,
            b: &CudaStorage,
            bl: &Layout,
        ) -> candle_core::Result<(CudaStorage, Shape)> {
            let (CudaStorageSlice::BF16(av), CudaStorageSlice::BF16(bv)) =
                (&a.slice, &b.slice)
            else {
                candle_core::bail!("BF16 only")
            };
            if !al.is_contiguous()
                || !bl.is_contiguous()
                || al.shape() != bl.shape()
            {
                candle_core::bail!("invalid comparison shape")
            }
            let n = u32::try_from(al.shape().elem_count()).map_err(err)?;
            static PTX: std::sync::OnceLock<String> =
                std::sync::OnceLock::new();
            let ptx = PTX.get_or_init(|| candle_core::cuda_backend::cudarc::nvrtc::compile_ptx(r#"
            extern "C" __global__ void bit_mismatches(const unsigned short *a, const unsigned short *b, unsigned int *count, unsigned int n) {
                unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n && a[i] != b[i]) atomicAdd(count, 1u);
            }
        "#).expect("compile diagnostic counter").to_src());
            let dev = a.device();
            let func = dev.get_or_load_custom_func(
                "bit_mismatches",
                "vs1_gemm_check",
                ptx,
            )?;
            let av =
                av.slice(al.start_offset()..al.start_offset() + n as usize);
            let bv =
                bv.slice(bl.start_offset()..bl.start_offset() + n as usize);
            let mut count = dev.alloc_zeros::<u32>(1)?;
            let mut launch = func.builder();
            launch.arg(&av).arg(&bv).arg(&mut count).arg(&n);
            // SAFETY: equally sized contiguous buffers, checked 32-bit length, and
            // an initialized atomic counter. This compares storage bits, not floats.
            unsafe { launch.launch(LaunchConfig::for_num_elems(n)) }
                .map_err(err)?;
            Ok((
                CudaStorage {
                    slice: CudaStorageSlice::U32(count),
                    device: dev.clone(),
                },
                Shape::from(()),
            ))
        }
    }
    fn mismatch_count(actual: &Tensor, expected: &Tensor) -> Result<usize> {
        Ok(actual
            .apply_op2_no_bwd(expected, &Mismatches)?
            .to_scalar::<u32>()? as usize)
    }
    fn report_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/cuda-optimization")
            .join(name)
    }

    // Change only the output tile, preserving the algorithm, K partitioning,
    // reduction scheme, staging, swizzle, and custom options of the first heuristic.
    fn retile_index(
        algorithms: &[sys::cublasLtMatmulHeuristicResult_t],
    ) -> Result<Option<usize>> {
        let Some(first) = algorithms.first() else {
            return Ok(None);
        };
        let mut target = description(&first.algo)?;
        if target["tile"] != 18
            || target["split_k"] != 4
            || target["reduction"] != 4
        {
            return Ok(None);
        }
        target["tile"] = json!(15);
        for (i, candidate) in algorithms.iter().enumerate() {
            if description(&candidate.algo)? == target {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    #[test]
    #[ignore = "requires CUDA; run alone"]
    fn retile_preserves_adversarial_products() -> Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        let seed = Tensor::zeros((1, 1), DType::BF16, &device)?;
        let engine = Engine::new(&seed)?;
        let runtime = Retile::default();
        eprintln!("cuBLASLt version: {}", unsafe { sys::cublasLtGetVersion() });
        let mut results = vec![];
        let mut state = 0x5a93c62du32;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for m in 1..=512 {
            for k in [1024, 2624] {
                let plan = Plan::new([m, 1024, k])?;
                let algorithms = plan.algorithms(&engine)?;
                let Some(index) = retile_index(&algorithms)? else {
                    continue;
                };
                for pattern in 0..3 {
                    let values =
                        |count: usize, random: &mut dyn FnMut() -> u32| {
                            (0..count)
                                .map(|_| {
                                    let r = random();
                                    match pattern {
                                        0 => {
                                            ((r & 65535) as f32 - 32768.0)
                                                / 32768.0
                                        }
                                        1 => {
                                            let value = 2.0f32.powi(
                                                ((r >> 16) % 17) as i32 - 8,
                                            );
                                            if r & 1 == 0 {
                                                value
                                            } else {
                                                -value
                                            }
                                        }
                                        _ => [
                                            16384.0, -16384.0, 1.0, -1.0,
                                            0.0001, -0.0001, 0.0, -0.0,
                                        ]
                                            [r as usize % 8],
                                    }
                                })
                                .collect::<Vec<_>>()
                        };
                    let xs = Tensor::from_vec(
                        values(m * k, &mut random),
                        (m, k),
                        &device,
                    )?
                    .to_dtype(DType::BF16)?;
                    let weights = Tensor::from_vec(
                        values(1024 * k, &mut random),
                        (1024, k),
                        &device,
                    )?
                    .to_dtype(DType::BF16)?;
                    let linear = Linear::new(weights, None);
                    let expected = xs.apply(&linear)?;
                    let actual = output(
                        &engine,
                        &plan,
                        &algorithms[index].algo,
                        &xs,
                        &linear,
                    )?;
                    let mismatches = mismatch_count(&actual, &expected)?;
                    let hits = HITS.load(std::sync::atomic::Ordering::Relaxed);
                    ensure!(
                        mismatch_count(
                            &runtime.linear(&xs, &linear)?,
                            &expected
                        )? == 0,
                        "runtime retile output drift"
                    );
                    ensure!(
                        HITS.load(std::sync::atomic::Ordering::Relaxed) > hits,
                        "runtime fell back for shape {:?}",
                        plan.shape
                    );
                    let row = json!({"shape":plan.shape,"pattern":pattern,"mismatches":mismatches,"elements":actual.elem_count(),"default":description(&algorithms[0].algo)?,"candidate":description(&algorithms[index].algo)?});
                    eprintln!("{row}");
                    results.push(row);
                }
            }
        }
        std::fs::write(
            report_path("15-retile-adversarial.json"),
            serde_json::to_vec_pretty(&results)?,
        )?;
        ensure!(!results.is_empty(), "retile never exercised");
        ensure!(
            HITS.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "runtime never exercised"
        );
        ensure!(
            results.iter().all(|r| r["mismatches"] == 0),
            "retile changed BF16 outputs"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn retile_model_outputs_and_latency() -> Result<()> {
        use std::sync::atomic::Ordering;

        use crate::{Question, SystemOne, SystemOneRequest};
        let device = candle_core::Device::new_cuda(0)?;
        let model: SystemOne = SystemOne::from(crate::DEFAULT_REPO_ID)
            .with_device(device)
            .with_dtype(DType::BF16)
            .try_into()?;
        let mut extra = vec![];
        for target in 128..=137 {
            for (kind, question) in [
                ("noul", Question::noul("Is this relevant?")),
                (
                    "choice",
                    Question::choice(
                        "Choose a topic",
                        [("travel", "travel"), ("code", "software")],
                    ),
                ),
                (
                    "score",
                    Question::score("How relevant?", ["low", "medium", "high"]),
                ),
            ] {
                let request = (0..512)
                    .find_map(|n| {
                        let request = SystemOneRequest::new("hotel ".repeat(n))
                            .question("q", question.clone());
                        let state = model.encode_state(&request.state).unwrap();
                        let item = model
                            .build_sequence(&state, "q", &question)
                            .unwrap();
                        (item.ids.len() == target).then_some(request)
                    })
                    .context("construct exact token count")?;
                REFERENCE.store(true, Ordering::Relaxed);
                let expected = snapshot(&[model.system_one(&request)?]);
                REFERENCE.store(false, Ordering::Relaxed);
                ensure!(
                    snapshot(&[model.system_one(&request)?]) == expected,
                    "{target} {kind} output drift"
                );
                if (kind == "noul" && [128, 130, 136, 137].contains(&target))
                    || (kind == "choice" && target == 131)
                    || (kind == "score" && target == 132)
                {
                    extra.push((
                        format!("tokens{target}_{kind}"),
                        vec![request],
                    ));
                }
            }
        }
        ensure!(HITS.load(Ordering::Relaxed) > 0, "runtime never exercised");
        // Preserve a cache across alternating host threads and compare results;
        // this exercises the scratch-buffer completion event on PTDS queues.
        let request = extra
            .iter()
            .find(|(name, _)| name == "tokens130_noul")
            .unwrap()
            .1[0]
            .clone();
        let expected = snapshot(&[model.system_one(&request)?]);
        for _ in 0..3 {
            std::thread::scope(|scope| {
                for _ in 0..2 {
                    let model = &model;
                    let request = &request;
                    let expected = &expected;
                    scope.spawn(move || {
                        assert_eq!(
                            &snapshot(&[model.system_one(request).unwrap()]),
                            expected
                        );
                    });
                }
            });
        }
        drop(model);
        if std::env::var_os("VS1_RETILE_CHECK_ONLY").is_some() {
            return Ok(());
        }
        let mut cases = crate::model::batch_bench::cases();
        cases.extend(extra);
        crate::model::batch_bench::run_paired_cases(&REFERENCE, cases)
    }
    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
    }
    fn timing(
        xs: &Tensor,
        mut f: impl FnMut() -> Result<Tensor>,
    ) -> Result<f64> {
        for _ in 0..2 {
            std::hint::black_box(f()?);
        }
        let stream = xs.device().as_cuda_device()?.cuda_stream();
        let mut samples = vec![];
        for _ in 0..3 {
            stream.synchronize()?;
            let start =
                stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            for _ in 0..8 {
                std::hint::black_box(f()?);
            }
            let end =
                stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            end.synchronize()?;
            samples.push(f64::from(start.elapsed_ms(&end)?) / 8.0);
        }
        Ok(median(&mut samples))
    }
    fn description(algo: &sys::cublasLtMatmulAlgo_t) -> Result<Value> {
        let mut fields = serde_json::Map::new();
        for (name, value) in [
            "id",
            "tile",
            "split_k",
            "reduction",
            "stages",
            "swizzle",
            "custom",
            "inner_shape",
            "cluster_shape",
        ]
        .into_iter()
        .zip(config(algo)?)
        {
            fields.insert(name.into(), json!(value));
        }
        Ok(Value::Object(fields))
    }

    struct Candidate {
        algo: sys::cublasLtMatmulAlgo_t,
        ms: f64,
        row: usize,
    }
    struct Entry {
        plan: Plan,
        candidates: Vec<Candidate>,
        baseline_ms: f64,
        rows: Vec<Value>,
        checked: usize,
    }
    impl Entry {
        fn discover(
            engine: &Engine,
            xs: &Tensor,
            linear: &Linear,
            expected: &Tensor,
        ) -> Result<Self> {
            let (m, k) = xs.dims2()?;
            let n = linear.weight().dim(0)?;
            let plan = Plan::new([m, n, k])?;
            let baseline_ms = timing(xs, || Ok(xs.apply(linear)?))?;
            let mut candidates = vec![];
            let mut rows = vec![];
            for heuristic in plan.algorithms(engine)? {
                let actual =
                    output(engine, &plan, &heuristic.algo, xs, linear)?;
                let mismatches = mismatch_count(&actual, expected)?;
                // Cross-check the diagnostic counter on a real projection of
                // every shape, independently of its dedicated finite-value test.
                if rows.is_empty() {
                    ensure!(
                        mismatches
                            == bits(&actual)?
                                .iter()
                                .zip(bits(expected)?)
                                .filter(|(a, b)| **a != *b)
                                .count()
                    );
                }
                let ms = timing(xs, || {
                    output(engine, &plan, &heuristic.algo, xs, linear)
                })?;
                let row = rows.len();
                rows.push(json!({"algorithm":description(&heuristic.algo)?, "ms":ms, "workspace":heuristic.workspaceSize,
                "initial_mismatches":mismatches, "later_mismatches":0, "elements":actual.elem_count()}));
                if mismatches == 0 {
                    candidates.push(Candidate {
                        algo: heuristic.algo,
                        ms,
                        row,
                    });
                }
            }
            candidates.sort_by(|a, b| a.ms.total_cmp(&b.ms));
            eprintln!(
                "GEMM {:?}: baseline {:.4}ms, {} / {} exact, fastest exact {:?}ms",
                plan.shape,
                baseline_ms,
                candidates.len(),
                rows.len(),
                candidates.first().map(|c| c.ms)
            );
            Ok(Self {
                plan,
                candidates,
                baseline_ms,
                rows,
                checked: 1,
            })
        }
        fn validate(
            &mut self,
            engine: &Engine,
            xs: &Tensor,
            linear: &Linear,
            expected: &Tensor,
        ) -> Result<()> {
            let mut surviving = vec![];
            for candidate in self.candidates.drain(..) {
                let actual =
                    output(engine, &self.plan, &candidate.algo, xs, linear)?;
                let mismatches = mismatch_count(&actual, expected)?;
                if mismatches == 0 {
                    surviving.push(candidate);
                } else {
                    self.rows[candidate.row]["later_mismatches"] =
                        json!(mismatches);
                }
            }
            self.candidates = surviving;
            self.checked += 1;
            Ok(())
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Reference,
        Discover,
        Candidate,
    }
    struct Session {
        engine: Engine,
        mode: Mode,
        entries: BTreeMap<[usize; 3], Entry>,
        hits: usize,
    }
    thread_local! { static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) }; }

    pub(crate) fn linear(
        xs: &Tensor,
        linear: &Linear,
    ) -> Option<candle_core::Result<Tensor>> {
        if xs.dtype() != DType::BF16
            || !xs.device().is_cuda()
            || xs.rank() != 2
            || linear.bias().is_some()
        {
            return None;
        }
        SESSION.with(|cell| {
            let mut guard = cell.borrow_mut();
            let session = guard.as_mut()?;
            if session.mode == Mode::Reference {
                return Some(xs.apply(linear));
            }
            Some(
                (|| -> Result<Tensor> {
                    let (m, k) = xs.dims2()?;
                    let shape = [m, linear.weight().dim(0)?, k];
                    if session.mode == Mode::Candidate {
                        if let Some(entry) = session.entries.get(&shape)
                            && let Some(candidate) = entry
                                .candidates
                                .first()
                                .filter(|c| c.ms < entry.baseline_ms * 0.98)
                        {
                            session.hits += 1;
                            return output(
                                &session.engine,
                                &entry.plan,
                                &candidate.algo,
                                xs,
                                linear,
                            );
                        }
                        return Ok(xs.apply(linear)?);
                    }
                    let reference = xs.apply(linear)?;
                    if let Some(entry) = session.entries.get_mut(&shape) {
                        entry.validate(
                            &session.engine,
                            xs,
                            linear,
                            &reference,
                        )?;
                    } else {
                        session.entries.insert(
                            shape,
                            Entry::discover(
                                &session.engine,
                                xs,
                                linear,
                                &reference,
                            )?,
                        );
                    }
                    Ok(reference)
                })()
                .map_err(err),
            )
        })
    }

    fn set_mode(mode: Mode) {
        SESSION.with(|s| s.borrow_mut().as_mut().unwrap().mode = mode);
    }
    fn snapshot(responses: &[crate::SystemOneResponse]) -> Value {
        json!({"responses":responses, "actions":responses.iter().map(|r| r.answers.values().map(|a| a.action().map(|a| a.act_probability)).collect::<Vec<_>>()).collect::<Vec<_>>()})
    }

    #[test]
    #[ignore = "requires CUDA and checkpoint; run alone"]
    fn tune_encoder_gemms() -> Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        // Check every finite BF16 encoding, signed zero, offsets, and launch tails
        // against the original host bit comparison before using the GPU counter.
        let values: Vec<f32> = (0u32..=u16::MAX as u32)
            .filter(|b| b & 0x7f80 != 0x7f80)
            .map(|b| f32::from_bits(b << 16))
            .collect();
        let all = Tensor::from_vec(values.clone(), values.len(), &device)?
            .to_dtype(DType::BF16)?;
        for offset in [0, 1, 127, 8191] {
            let shifted = Tensor::from_vec(
                (0..values.len())
                    .map(|i| values[(i + offset) % values.len()])
                    .collect::<Vec<_>>(),
                values.len(),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            for (start, len) in [(0, values.len()), (17, 1003)] {
                let a = all.narrow(0, start, len)?;
                let b = shifted.narrow(0, start, len)?;
                let expected = bits(&a)?
                    .iter()
                    .zip(bits(&b)?)
                    .filter(|(a, b)| **a != *b)
                    .count();
                ensure!(
                    mismatch_count(&a, &b)? == expected,
                    "diagnostic counter drift"
                );
            }
        }
        let zero = Tensor::from_vec(vec![0.0f32, -0.0], 2, &device)?
            .to_dtype(DType::BF16)?;
        ensure!(
            mismatch_count(&zero.narrow(0, 0, 1)?, &zero.narrow(0, 1, 1)?)?
                == 1
        );
        ensure!(
            report_path("14-gemm-search.json")
                .parent()
                .unwrap()
                .is_dir(),
            "missing report directory"
        );
        let model: crate::SystemOne =
            crate::SystemOne::from(crate::DEFAULT_REPO_ID)
                .with_device(device.clone())
                .with_dtype(DType::BF16)
                .try_into()?;
        let seed = Tensor::zeros((1, 1), DType::BF16, &device)?;
        SESSION.with(|s| {
            *s.borrow_mut() = Some(Session {
                engine: Engine::new(&seed).unwrap(),
                mode: Mode::Reference,
                entries: BTreeMap::new(),
                hits: 0,
            })
        });
        let cases = crate::model::batch_bench::cases();
        let mut expected = vec![];
        for (_, requests) in &cases {
            expected.push(snapshot(&model.system_one_batch(requests)?));
        }
        set_mode(Mode::Discover);
        for (i, (name, requests)) in cases.iter().enumerate() {
            eprintln!("Discover {name}");
            ensure!(
                snapshot(&model.system_one_batch(requests)?) == expected[i]
            );
            let changed = serde_json::from_str::<Vec<crate::SystemOneRequest>>(
                &serde_json::to_string(requests)?
                    .replace("changed", "removed")
                    .replace("One way", "Two way"),
            )?;
            model.system_one_batch(&changed)?;
        }
        let report = SESSION.with(|s| {
        s.borrow().as_ref().unwrap().entries.values().map(|entry| {
            let selected = entry.candidates.first().filter(|c| c.ms < entry.baseline_ms * 0.98);
            json!({"shape":entry.plan.shape, "baseline_ms":entry.baseline_ms, "checked_inputs":entry.checked,
                "surviving":entry.candidates.len(), "selected_row":selected.map(|c|c.row), "algorithms":entry.rows})
        }).collect::<Vec<_>>()
    });
        std::fs::write(
            report_path("14-gemm-search.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        let mut paired = vec![];
        for (i, (name, requests)) in cases.iter().enumerate() {
            for j in 0..6 {
                set_mode(if j % 2 == 0 {
                    Mode::Reference
                } else {
                    Mode::Candidate
                });
                ensure!(
                    snapshot(&model.system_one_batch(requests)?) == expected[i],
                    "warmup drift: {name}"
                );
            }
            let (mut baseline, mut candidate, mut ratios) =
                (vec![], vec![], vec![]);
            let mut faster = 0;
            for iteration in 0..40 {
                let mut times = [0.0; 2];
                for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                    set_mode(if index == 0 {
                        Mode::Reference
                    } else {
                        Mode::Candidate
                    });
                    device.synchronize()?;
                    let start = Instant::now();
                    let result =
                        model.system_one_batch(requests).with_context(
                            || format!("{name} iteration {iteration}"),
                        )?;
                    device.synchronize()?;
                    times[index] = start.elapsed().as_secs_f64() * 1000.0;
                    ensure!(
                        snapshot(&result) == expected[i],
                        "output drift: {name} iteration {iteration}"
                    );
                }
                baseline.push(times[0]);
                candidate.push(times[1]);
                ratios.push(times[1] / times[0]);
                faster += usize::from(times[1] < times[0]);
            }
            let row = json!({"name":name, "reference_p50_ms":median(&mut baseline), "candidate_p50_ms":median(&mut candidate),
            "paired_change_percent":100.0*(median(&mut ratios)-1.0), "faster_pairs":faster, "pairs":40, "outputs_exact":true});
            eprintln!("{row}");
            paired.push(row);
            std::fs::write(
                report_path("14-paired.json"),
                serde_json::to_vec_pretty(&paired)?,
            )?;
        }
        SESSION.with(|s| {
            let mut session = s.borrow_mut();
            eprintln!("candidate GEMMs: {}", session.as_ref().unwrap().hits);
            *session = None;
        });
        Ok(())
    }
}
