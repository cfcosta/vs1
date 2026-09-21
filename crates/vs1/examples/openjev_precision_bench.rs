//! Paired F32/BF16 latency, excluding model loading and tokenization.
//! openjev_precision_bench CHECKPOINT REFERENCE
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    use anyhow::ensure;
    use candle_core::{DType, Device};
    use vs1::{OpenJev, SystemOneRequest};
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "expected CHECKPOINT REFERENCE");
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let device = Device::new_cuda(0)?;
    let f32: OpenJev = OpenJev::from(&args[1])
        .with_device(device.clone())
        .with_dtype(DType::F32)
        .try_into()?;
    let bf16: OpenJev = OpenJev::from(&args[1])
        .with_device(device)
        .with_dtype(DType::BF16)
        .try_into()?;
    let inputs = cases
        .iter()
        .take(13)
        .map(|case| {
            let request: SystemOneRequest =
                serde_json::from_value(case["request"].clone())?;
            Ok(f32.build_input(
                &request.state.render(),
                "decision",
                &request.questions["decision"],
            )?)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut results = Vec::new();
    for (name, input) in [("short", &inputs[..1]), ("mixed_13", &inputs[..])] {
        for _ in 0..3 {
            f32.predict(input)?;
            bf16.predict(input)?;
        }
        let mut times = [Vec::new(), Vec::new()];
        let models = [&f32, &bf16];
        for iteration in 0..40 {
            for i in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                let started = Instant::now();
                std::hint::black_box(models[i].predict(input)?);
                times[i].push(started.elapsed().as_secs_f64() * 1000.);
            }
        }
        for time in &mut times {
            time.sort_by(f64::total_cmp);
        }
        results.push(serde_json::json!({"scenario":name,"pairs":40,
            "f32_median_ms":times[0][20],"bf16_median_ms":times[1][20],
            "speedup":times[0][20]/times[1][20],"flash_attn":cfg!(feature="flash-attn")}));
    }
    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("requires --features cuda or flash-attn")
}
