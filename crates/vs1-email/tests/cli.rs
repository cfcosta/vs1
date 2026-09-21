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
            "--host",
            "unreachable.invalid",
            "--username",
            "test",
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
    for flag in [
        "--dry-run",
        "--config",
        "--host",
        "--mailbox",
        "--search",
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
