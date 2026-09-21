use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use clap::Parser;
use vs1::{DecisionModel, SystemOne};
use vs1_email::{
    Config,
    dry_run_single_choice_with_progress,
    dry_run_with_progress,
    read_maildir,
    request_fits,
};

#[derive(Parser)]
#[command(
    version,
    about = "Categorize a local Maildir mailbox using caller-selected laya or Jev models. Only --dry-run is implemented."
)]
struct Args {
    /// Print proposed categories as JSON without changing the mailbox.
    #[arg(long)]
    dry_run: bool,
    /// Write completed classifications as JSONL while running; file must not exist.
    #[arg(long)]
    progress_jsonl: Option<PathBuf>,
    /// TOML file containing [[rules]] and optional [owner] context.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Local Maildir or sync root; includes all descendant Maildir folders.
    #[arg(long)]
    mailbox: Option<PathBuf>,
    /// Global maximum messages across all folders, in sorted file-path order.
    #[arg(long, default_value = "100", value_parser = positive)]
    limit: usize,
    /// Explicit backend; Jev sends email content to TypeSafe and requires TYPESAFE_API_KEY.
    #[arg(long, default_value="laya",value_parser=["laya","jev"])]
    backend: String,
    /// Local checkpoint or hosted model ID; defaults to laya's checkpoint / jev-latest.
    #[arg(long)]
    model: Option<String>,
    #[arg(long, default_value = "")]
    subfolder: String,
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "cuda", "metal"])]
    device: String,
    #[arg(long, value_parser = ["f32", "bf16", "f16"])]
    dtype: Option<String>,
    /// Messages per model batch; reduce if GPU memory is insufficient.
    #[arg(long, default_value = "16", value_parser = positive)]
    batch_size: usize,
    /// Sequence token budget; defaults to checkpoint configuration. Bodies are chunked.
    #[arg(long, value_parser = positive)]
    max_len: Option<usize>,
    /// Question-header budget; defaults to checkpoint configuration.
    #[arg(long, value_parser = positive)]
    head_max_len: Option<usize>,
}

fn positive(raw: &str) -> std::result::Result<usize, String> {
    raw.parse::<usize>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| "must be an integer greater than zero".into())
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Keep this guard ahead of all file, credential, network and model access.
    ensure!(
        args.dry_run,
        "mailbox changes are not implemented; use --dry-run"
    );
    if args.backend == "jev" {
        ensure!(
            args.device == "cpu"
                && args.dtype.is_none()
                && args.subfolder.is_empty()
                && args.max_len.is_none()
                && args.head_max_len.is_none(),
            "device, dtype, subfolder and token-budget options are local-only"
        );
    }
    let path = args.config.context("--config is required with --dry-run")?;
    if let (Some(head), Some(max)) = (args.head_max_len, args.max_len) {
        ensure!(head < max, "--head-max-len must be smaller than --max-len");
    }
    let config = Config::parse(
        &fs::read_to_string(path).context("cannot read rules file")?,
    )?;
    let mailbox_path = args
        .mailbox
        .context("--mailbox must specify a local Maildir or sync root")?;
    let mailbox = read_maildir(&mailbox_path, args.limit)?;

    let mut progress = args
        .progress_jsonl
        .as_ref()
        .map(|path| {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .with_context(|| {
                    format!("cannot create progress file {}", path.display())
                })
        })
        .transpose()?
        .map(io::BufWriter::new);
    eprintln!(
        "read {} messages; {} read/decode failures",
        mailbox.emails.len(),
        mailbox.failures.len()
    );
    let started = std::time::Instant::now();
    let mut completed = 0usize;
    // An empty mailbox needs neither weights nor inference.
    let model: Option<DecisionModel> = if mailbox.emails.is_empty() {
        None
    } else if args.backend == "jev" {
        #[cfg(feature = "jev")]
        {
            Some(
                vs1::JevClient::new(
                    std::env::var("TYPESAFE_API_KEY")
                        .context("TYPESAFE_API_KEY is required for Jev")?,
                    args.model.as_deref().unwrap_or("jev-latest"),
                )?
                .with_concurrency(args.batch_size)?
                .into(),
            )
        }
        #[cfg(not(feature = "jev"))]
        {
            anyhow::bail!("Jev requires building with --features jev")
        }
    } else {
        let device = match args.device.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::new_cuda(0).context(
                "CUDA requires the cuda feature and a supported GPU",
            )?,
            "metal" => Device::new_metal(0).context(
                "Metal requires the metal feature and a supported GPU",
            )?,
            _ => unreachable!("clap validates device"),
        };
        let mut builder = SystemOne::from(
            args.model.as_deref().unwrap_or(vs1::DEFAULT_REPO_ID),
        )
        .with_subfolder(&args.subfolder)
        .with_device(device);
        if let Some(n) = args.max_len {
            builder = builder.with_max_len(n);
        }
        if let Some(n) = args.head_max_len {
            builder = builder.with_head_max_len(n);
        }
        if let Some(dtype) = args.dtype {
            builder = builder.with_dtype(match dtype.as_str() {
                "f32" => DType::F32,
                "bf16" => DType::BF16,
                "f16" => DType::F16,
                _ => unreachable!("clap validates dtype"),
            });
        }
        builder = builder.with_batch_size(args.batch_size);
        let model: SystemOne = builder.try_into()?;
        eprintln!(
            "loaded {} on {:?} as {:?}",
            model.model_name(),
            model.device(),
            model.dtype()
        );
        Some(model.into())
    };
    let mut decide=|requests:&[vs1::SystemOneRequest]|->Result<Vec<vs1::SystemOneResponse>> {
        let response=model.as_ref().context("model not loaded")?.system_one_batch(requests)?;
        let questions:usize=requests.iter().map(|r|r.questions.len()).sum();
        completed+=questions;
        if completed/100 != (completed-questions)/100 {eprintln!("evaluated {completed} questions in {:.1}s",started.elapsed().as_secs_f64());}
        Ok(response)
    };
    let mut on_progress =
        |classification: &vs1_email::Classification| -> Result<()> {
            if let Some(writer) = progress.as_mut() {
                serde_json::to_writer(&mut *writer, classification)?;
                writeln!(writer)?;
                writer.flush()?;
            }
            Ok(())
        };
    let result = if args.backend == "jev" {
        dry_run_single_choice_with_progress(
            &config,
            &mailbox,
            args.batch_size,
            &mut decide,
            &mut on_progress,
        )
    } else {
        dry_run_with_progress(
            &config,
            &mailbox,
            args.batch_size,
            &mut |r| {
                request_fits(
                    model
                        .as_ref()
                        .and_then(|m| m.local())
                        .context("local model not loaded")?,
                    r,
                )
            },
            &mut decide,
            &mut on_progress,
        )
    };
    #[cfg(feature = "jev")]
    if let Some(DecisionModel::Jev(client)) = &model {
        eprintln!("Jev calls: {}", serde_json::to_string(&client.stats())?);
    }
    let report = result?;
    eprintln!(
        "completed {} emails from {} chunks and {completed} questions in {:.1}s",
        report.classifications.len(),
        report
            .classifications
            .iter()
            .map(|c| c.chunks.len())
            .sum::<usize>(),
        started.elapsed().as_secs_f64()
    );
    let mut stdout = io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, &report)?;
    writeln!(stdout)?;
    Ok(())
}
