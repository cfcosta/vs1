//! Warmed model trace: VS1_PROFILE_NSYS=1 for Nsight's cudaProfilerApi range,
//! or preload kernel_trace.cpp's collector for a standalone CUPTI trace.
#[cfg(feature = "flash-attn")]
fn main() -> anyhow::Result<()> {
    use std::ffi::{CString, c_char, c_void};

    use anyhow::ensure;
    use candle_core::{DType, Device};
    use vs1::{Question, SystemOne, SystemOneRequest};

    unsafe extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "kernel_profile CASE OUT.tsv");
    // SAFETY: the collector exports exactly these C signatures. Null means
    // the explicitly required collector has not been preloaded.
    let nsys = std::env::var_os("VS1_PROFILE_NSYS").is_some();
    let callbacks = if nsys {
        None
    } else {
        Some(unsafe {
            let start =
                dlsym(std::ptr::null_mut(), c"vs1_trace_start".as_ptr());
            let stop = dlsym(std::ptr::null_mut(), c"vs1_trace_stop".as_ptr());
            ensure!(
                !start.is_null() && !stop.is_null(),
                "preload kernel_trace.so"
            );
            (
                std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(
                    start,
                ),
                std::mem::transmute::<
                    *mut c_void,
                    unsafe extern "C" fn(*const c_char),
                >(stop),
            )
        })
    };
    let device = Device::new_cuda(0)?;
    let model: SystemOne = SystemOne::from(vs1::DEFAULT_REPO_ID)
        .with_device(device.clone())
        .with_dtype(DType::BF16)
        .try_into()?;
    let requests = match args[1].as_str() {
        "browser_call5" => vec![serde_json::from_str(include_str!(
            "../tests/fixtures/jev/call5_request.json"
        ))?],
        n => (0..n.parse::<usize>()?).map(|i| {
            SystemOneRequest::new(format!("Document {i}. {}", "The indexing pipeline compares content hashes and updates changed documents. Unchanged files are skipped. ".repeat(5)))
                .question("relevant", Question::noul("Does this explain how changed files are selected?"))
        }).collect(),
    };
    for _ in 0..5 {
        model.system_one_batch(&requests)?;
    }
    device.synchronize()?;
    // SAFETY: callback lifetime covers these synchronized inference calls.
    if let Some((start, _)) = callbacks {
        unsafe { start() };
    } else {
        unsafe {
            candle_core::cuda_backend::cudarc::driver::sys::cuProfilerStart()
                .result()?
        };
    }
    for _ in 0..10 {
        std::hint::black_box(model.system_one_batch(&requests)?);
    }
    device.synchronize()?;
    let path = CString::new(args[2].as_str())?;
    // SAFETY: path remains valid until the collector has finished writing.
    if let Some((_, stop)) = callbacks {
        unsafe { stop(path.as_ptr()) };
    } else {
        unsafe {
            candle_core::cuda_backend::cudarc::driver::sys::cuProfilerStop()
                .result()?
        };
    }
    Ok(())
}

#[cfg(not(feature = "flash-attn"))]
fn main() {
    eprintln!("kernel_profile requires --features flash-attn");
}
