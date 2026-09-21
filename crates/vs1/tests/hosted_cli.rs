#![cfg(feature = "jev")]
use std::process::Command;
#[test]
fn jev_cli_requires_credentials_and_rejects_local_token_dump() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/jev/call3_request.json"
    );
    for dump in [false, true] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_vs1"));
        cmd.args(["--backend", "jev", "--model", "jev-1.13.0"]);
        if dump {
            cmd.arg("--dump-ids");
        }
        cmd.arg(path).env_remove("TYPESAFE_API_KEY");
        let out = cmd.output().unwrap();
        assert!(!out.status.success());
        let error = String::from_utf8_lossy(&out.stderr);
        assert!(
            error.contains(if dump {
                "local-only"
            } else {
                "TYPESAFE_API_KEY"
            }),
            "{error}"
        );
    }
}
