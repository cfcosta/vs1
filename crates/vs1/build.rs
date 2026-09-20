use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo::rerun-if-changed=src/geglu.cu");
    println!("cargo::rerun-if-changed=src/rope_pair.cu");
    println!("cargo::rerun-if-env-changed=CUDA_PATH");
    println!("cargo::rerun-if-env-changed=NVCC");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = env::var_os("NVCC")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("CUDA_PATH")
                .map(|path| PathBuf::from(path).join("bin/nvcc"))
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    let output_dir =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    // Candle's BF16 CUDA kernels already require Ampere or newer.
    // Shipping PTX keeps forward compatibility and avoids runtime NVRTC.
    for kernel in ["geglu", "rope_pair"] {
        let status = Command::new(&nvcc)
            .args([
                "--ptx",
                "--gpu-architecture=compute_80",
                "-O3",
                "--std=c++17",
            ])
            .arg(format!("src/{kernel}.cu"))
            .arg("-o")
            .arg(output_dir.join(format!("{kernel}.ptx")))
            .status()
            .expect("CUDA builds require nvcc (set NVCC or CUDA_PATH)");
        assert!(status.success(), "compiling {kernel} failed");
    }
}
