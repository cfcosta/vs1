use std::{
    fs,
    io::{self, Write},
    net::{TcpStream, ToSocketAddrs},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device};
use clap::Parser;
use vs1::SystemOne;
use vs1_email::{Config, dry_run, read_mailbox};

#[derive(Parser)]
#[command(
    version,
    about = "Categorize an IMAP mailbox using local laya models. Only --dry-run is implemented."
)]
struct Args {
    /// Print proposed categories as JSON without changing the mailbox.
    #[arg(long)]
    dry_run: bool,
    /// TOML file containing [[rules]] and optional [owner] context.
    #[arg(long)]
    config: Option<PathBuf>,
    /// IMAP TLS hostname.
    #[arg(long, env = "VS1_EMAIL_HOST")]
    host: Option<String>,
    #[arg(long, default_value_t = 993)]
    port: u16,
    #[arg(long, env = "VS1_EMAIL_USERNAME")]
    username: Option<String>,
    #[arg(long, default_value = "INBOX")]
    mailbox: String,
    /// IMAP SEARCH expression, e.g. UNSEEN or SINCE 01-Sep-2026.
    #[arg(long, default_value = "ALL")]
    search: String,
    /// Maximum number of messages, in ascending UID order.
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
    #[arg(long, value_parser = positive)]
    batch_size: Option<usize>,
    /// Sequence token budget; long messages are truncated by laya.
    #[arg(long, default_value = "4096", value_parser = positive)]
    max_len: usize,
    /// Shared instruction/category token budget, smaller than --max-len.
    #[arg(long, default_value = "2048", value_parser = positive)]
    head_max_len: usize,
}

fn positive(raw: &str) -> std::result::Result<usize, String> {
    raw.parse::<usize>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| "must be an integer greater than zero".into())
}

fn connect(
    host: &str,
    port: u16,
) -> Result<
    imap::Client<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>,
> {
    let timeout = Duration::from_secs(30);
    let addresses = (host, port)
        .to_socket_addrs()
        .context("cannot resolve IMAP host")?;
    let mut socket = None;
    let mut last_error = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(stream) => {
                socket = Some(stream);
                break;
            }
            Err(error) => last_error = Some(error),
        }
    }
    let socket = match socket {
        Some(socket) => socket,
        None => bail!("cannot connect to IMAP host: {last_error:?}"),
    };
    socket.set_read_timeout(Some(timeout))?;
    socket.set_write_timeout(Some(timeout))?;
    let roots = rustls::RootCertStore::from_iter(
        webpki_roots::TLS_SERVER_ROOTS.iter().cloned(),
    );
    let tls = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_root_certificates(roots)
    .with_no_client_auth();
    let name = rustls::pki_types::ServerName::try_from(host.to_owned())?;
    let connection = rustls::ClientConnection::new(Arc::new(tls), name)?;
    let mut client =
        imap::Client::new(rustls::StreamOwned::new(connection, socket));
    client
        .read_greeting()
        .context("cannot read IMAP TLS greeting")?;
    Ok(client)
}

fn main() -> Result<()> {
    let args = Args::parse();
    // Keep this guard ahead of all file, credential, network and model access.
    ensure!(
        args.dry_run,
        "mailbox changes are not implemented; use --dry-run"
    );
    let path = args.config.context("--config is required with --dry-run")?;
    ensure!(
        args.head_max_len < args.max_len,
        "--head-max-len must be smaller than --max-len"
    );
    let config = Config::parse(
        &fs::read_to_string(path).context("cannot read rules file")?,
    )?;
    let host = args.host.context("--host or VS1_EMAIL_HOST is required")?;
    let username = args
        .username
        .context("--username or VS1_EMAIL_USERNAME is required")?;
    let password = std::env::var("VS1_EMAIL_PASSWORD").context(
        "VS1_EMAIL_PASSWORD is required (IMAP password or app password)",
    )?;
    let client = connect(&host, args.port)?;
    let mut session = client
        .login(&username, &password)
        .map_err(|(error, _)| error)
        .context("IMAP login failed")?;
    let mailbox =
        read_mailbox(&mut session, &args.mailbox, &args.search, args.limit);
    // LOGOUT is safe for an examined mailbox; never issue CLOSE/EXPUNGE.
    let logout = session.logout();
    let mailbox = mailbox?;
    logout.context("IMAP logout failed")?;

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
            .with_device(device)
            .with_max_len(args.max_len)
            .with_head_max_len(args.head_max_len);
        if let Some(dtype) = args.dtype {
            builder = builder.with_dtype(match dtype.as_str() {
                "f32" => DType::F32,
                "bf16" => DType::BF16,
                "f16" => DType::F16,
                _ => unreachable!("clap validates dtype"),
            });
        }
        if let Some(batch_size) = args.batch_size {
            builder = builder.with_batch_size(batch_size);
        }
        let model: SystemOne = builder.try_into()?;
        eprintln!(
            "loaded {} on {:?} as {:?}",
            model.model_name(),
            model.device(),
            model.dtype()
        );
        Some(model)
    };
    let report = dry_run(
        &config,
        &host,
        &username,
        &args.mailbox,
        &mailbox,
        &mut |request| {
            let model = model.as_ref().context("model not loaded")?;
            Ok(model.system_one(request)?)
        },
    )?;
    let mut stdout = io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, &report)?;
    writeln!(stdout)?;
    Ok(())
}
