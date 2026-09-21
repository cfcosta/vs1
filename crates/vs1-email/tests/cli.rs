use std::process::Command;

fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vs1-email"))
        .args(args)
        .env_remove("VS1_EMAIL_HOST")
        .env_remove("VS1_EMAIL_USERNAME")
        .env_remove("VS1_EMAIL_PASSWORD")
        .output()
        .unwrap()
}

#[test]
fn real_execution_errors_before_config_network_or_model_loading() {
    for args in [
        vec![],
        vec![
            "--config",
            "/does/not/exist",
            "--mailbox",
            "/does/not/exist",
        ],
    ] {
        let out = cli(&args);
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        let error = String::from_utf8(out.stderr).unwrap();
        assert!(
            error
                .contains("mailbox changes are not implemented; use --dry-run"),
            "{error}"
        );
    }
}

#[test]
fn help_explains_dry_run_and_model_options() {
    let out = cli(&["--help"]);
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("Maildir"));
    assert!(!text.contains("--host"));
    for flag in [
        "--dry-run",
        "--config",
        "--mailbox",
        "--limit",
        "--model",
        "--subfolder",
        "--device",
        "--dtype",
        "--batch-size",
        "--max-len",
        "--head-max-len",
    ] {
        assert!(text.contains(flag), "missing {flag}");
    }
}

#[test]
fn dry_run_requires_config_and_rejects_zero_limits() {
    let out = cli(&["--dry-run"]);
    assert!(!out.status.success());
    assert!(String::from_utf8(out.stderr).unwrap().contains("--config"));
    for flag in ["--limit", "--batch-size", "--max-len", "--head-max-len"] {
        let out = cli(&["--dry-run", "--config", "/missing", flag, "0"]);
        assert!(!out.status.success());
        let error = String::from_utf8(out.stderr).unwrap();
        assert!(error.contains("greater than zero"), "{error}");
    }
}

#[test]
fn empty_local_maildir_needs_no_credentials_or_model() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["cur", "new", "tmp"] {
        std::fs::create_dir(dir.path().join(name)).unwrap();
    }
    let config = dir.path().join("rules.toml");
    std::fs::write(&config, "[[rules]]\ncategory='a'\nwhat='A'\n[[rules]]\ncategory='other'\nwhat='Other'").unwrap();
    let out = cli(&[
        "--dry-run",
        "--config",
        config.to_str().unwrap(),
        "--mailbox",
        dir.path().to_str().unwrap(),
        "--model",
        "/missing-model",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["mailbox"], dir.path().to_str().unwrap());
    assert_eq!(report["classifications"], serde_json::json!([]));
    assert!(report.get("host").is_none());
}

#[test]
fn progress_file_is_created_without_overwriting_existing_results() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["cur", "new", "tmp"] {
        std::fs::create_dir(dir.path().join(name)).unwrap();
    }
    let config = dir.path().join("rules.toml");
    std::fs::write(
        &config,
        "[[rules]]\ncategory='a'\nwhat='A'\n[[rules]]\ncategory='b'\nwhat='B'",
    )
    .unwrap();
    let progress = dir.path().join("progress.jsonl");
    let args = [
        "--dry-run",
        "--config",
        config.to_str().unwrap(),
        "--mailbox",
        dir.path().to_str().unwrap(),
        "--progress-jsonl",
        progress.to_str().unwrap(),
    ];
    let out = cli(&args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(&progress).unwrap(), b"");
    std::fs::write(&progress, "previous results").unwrap();
    assert!(!cli(&args).status.success());
    assert_eq!(
        std::fs::read_to_string(&progress).unwrap(),
        "previous results"
    );
}
#[test]
fn jev_is_explicit_and_rejects_local_tuning_flags() {
    let help = cli(&["--help"]);
    assert!(String::from_utf8_lossy(&help.stdout).contains("--backend"));
    let out = cli(&[
        "--dry-run",
        "--backend",
        "jev",
        "--device",
        "cuda",
        "--config",
        "/missing",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("local-only"));
}
