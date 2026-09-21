//! Keep Laya and OpenJev resident and alternate typed requests between them.
//! models_together LAYA_CHECKPOINT OPENJEV_CHECKPOINT cpu|cuda
use anyhow::ensure;
use candle_core::Device;
use vs1::{DecisionModel, OpenJev, SystemOne, SystemOneRequest};

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "expected LAYA_CHECKPOINT OPENJEV_CHECKPOINT cpu|cuda"
    );
    let device = match args[3].as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        _ => anyhow::bail!("unsupported device"),
    };
    let openjev: OpenJev = OpenJev::from(&args[2])
        .with_device(device.clone())
        .try_into()?;
    let request: SystemOneRequest = serde_json::from_value(
        serde_json::json!({
            "state":"The customer asks to reset a forgotten password.",
            "questions":{"route":{"type":"choice","instructions":"What should support do?",
                "criteria":{"reset":"reset the password","refund":"refund a purchase"}}}
        }),
    )?;
    let before = serde_json::to_value(openjev.system_one(&request)?)?;
    let laya: SystemOne =
        SystemOne::from(&args[1]).with_device(device).try_into()?;
    let models: [DecisionModel; 2] = [laya.into(), openjev.into()];
    let baseline = models
        .iter()
        .map(|m| Ok(serde_json::to_value(m.system_one(&request)?)?))
        .collect::<anyhow::Result<Vec<_>>>()?;
    ensure!(before == baseline[1], "loading Laya changed OpenJev output");
    for _ in 0..3 {
        for (model, expected) in models.iter().zip(&baseline) {
            ensure!(
                serde_json::to_value(model.system_one(&request)?)? == *expected,
                "alternating models changed {} output",
                model.model_name()
            );
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "resident_models":models.iter().map(DecisionModel::model_name).collect::<Vec<_>>(),
            "alternating_runs_per_model":3,"stable":true,"responses":baseline
        }))?
    );
    Ok(())
}
