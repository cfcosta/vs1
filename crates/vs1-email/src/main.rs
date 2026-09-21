use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use clap::Parser;
use vs1::SystemOne;
use vs1_email::{Config, dry_run_with_progress, read_maildir, request_fits};

#[derive(Parser)]
#[command(
    version,
    about = "Categorize a local Maildir mailbox using local laya models. Only --dry-run is implemented."
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
    /// Hugging Face checkpoint or local checkpoint directory.
    #[arg(long, default_value = vs1::DEFAULT_REPO_ID)]
    model: String,
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
    let model = if mailbox.emails.is_empty() {
        None
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
        let mut builder = SystemOne::from(&args.model)
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
        Some(model)
    };
    let report = dry_run_with_progress(
        &config,
        &mailbox,
        args.batch_size,
        &mut |request| {
            request_fits(model.as_ref().context("model not loaded")?, request)
        },
        &mut |requests| {
            let model = model.as_ref().context("model not loaded")?;
            let response = model.system_one_batch(requests)?;
            let questions: usize =
                requests.iter().map(|r| r.questions.len()).sum();
            completed += questions;
            if completed / 100 != (completed - questions) / 100 {
                eprintln!(
                    "evaluated {completed} questions in {:.1}s",
                    started.elapsed().as_secs_f64()
                );
            }
            Ok(response)
        },
        &mut |classification| {
            if let Some(writer) = progress.as_mut() {
                serde_json::to_writer(&mut *writer, classification)?;
                writeln!(writer)?;
                writer.flush()?;
            }
            Ok(())
        },
    )?;
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
