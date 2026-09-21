use std::fs;

use vs1_email::read_maildir;

fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for name in ["cur", "new", "tmp"] {
        fs::create_dir(dir.path().join(name)).unwrap();
    }
    dir
}

#[test]
fn reads_cur_and_new_in_order_without_changing_contents_names_or_flags() {
    let dir = fixture();
    for name in ["cur/b:2,S", "new/a", "tmp/in-progress"] {
        fs::write(
            dir.path().join(name),
            format!("Subject: {name}\r\n\r\nMessage body"),
        )
        .unwrap();
    }
    fs::write(dir.path().join(".uidvalidity"), "123").unwrap();
    let before =
        ["cur/b:2,S", "new/a", "tmp/in-progress", ".uidvalidity"].map(|name| {
            let p = dir.path().join(name);
            (
                p.clone(),
                fs::read(&p).unwrap(),
                fs::metadata(p).unwrap().modified().unwrap(),
            )
        });
    let mailbox = read_maildir(dir.path(), 100).unwrap();
    assert_eq!(mailbox.path, dir.path().canonicalize().unwrap());
    assert_eq!(mailbox.emails.len(), 2);
    assert_eq!(mailbox.emails[0].path, dir.path().join("cur/b:2,S"));
    assert_eq!(mailbox.emails[1].path, dir.path().join("new/a"));
    assert_eq!(mailbox.emails[1].body, "Message body");
    for (path, data, modified) in before {
        assert_eq!(fs::read(&path).unwrap(), data);
        assert_eq!(fs::metadata(path).unwrap().modified().unwrap(), modified);
    }
    assert_eq!(fs::read_dir(dir.path().join("new")).unwrap().count(), 1);
    assert_eq!(fs::read_dir(dir.path().join("cur")).unwrap().count(), 1);
    assert_eq!(read_maildir(dir.path(), 1).unwrap().emails.len(), 1);
}

#[test]
fn empty_maildir_is_valid_but_missing_directories_and_zero_limit_are_errors() {
    let dir = fixture();
    assert!(read_maildir(dir.path(), 100).unwrap().emails.is_empty());
    assert!(read_maildir(dir.path(), 0).is_err());
    fs::remove_dir(dir.path().join("new")).unwrap();
    assert!(read_maildir(dir.path(), 100).is_err());
    assert!(read_maildir(&dir.path().join("missing"), 100).is_err());
}

#[test]
fn skips_subdirectories_and_symlink_entries() {
    let dir = fixture();
    fs::create_dir(dir.path().join("cur/subdir")).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("/does/not/exist", dir.path().join("new/link"))
        .unwrap();
    assert!(read_maildir(dir.path(), 100).unwrap().emails.is_empty());
}

fn add_maildir(root: &std::path::Path, folder: &str, message: &str) {
    let path = root.join(folder);
    for name in ["cur", "new", "tmp"] {
        fs::create_dir_all(path.join(name)).unwrap();
    }
    fs::write(path.join(message), b"Subject: Nested\r\n\r\nBody").unwrap();
}

#[test]
fn sync_root_reads_all_nested_and_hidden_maildirs_with_one_global_limit() {
    let root = tempfile::tempdir().unwrap();
    add_maildir(root.path(), "INBOX", "new/a");
    add_maildir(root.path(), "Archive/2026", "cur/b:2,S");
    add_maildir(root.path(), ".Sent", "new/c");
    fs::create_dir(root.path().join("metadata")).unwrap();
    fs::write(root.path().join("metadata/state"), b"not an email").unwrap();
    let expected = [".Sent/new/c", "Archive/2026/cur/b:2,S", "INBOX/new/a"]
        .map(|p| root.path().join(p));
    let all = read_maildir(root.path(), 100).unwrap();
    assert_eq!(all.path, root.path().canonicalize().unwrap());
    assert_eq!(
        all.emails.iter().map(|e| &e.path).collect::<Vec<_>>(),
        expected.iter().collect::<Vec<_>>()
    );
    let limited = read_maildir(root.path(), 2).unwrap();
    assert_eq!(
        limited.emails.iter().map(|e| &e.path).collect::<Vec<_>>(),
        expected[..2].iter().collect::<Vec<_>>()
    );
}

#[test]
fn maildir_root_includes_its_subfolders_but_never_descends_into_message_storage()
 {
    let root = fixture();
    fs::write(root.path().join("new/root"), b"Subject: Root\r\n\r\nBody")
        .unwrap();
    add_maildir(root.path(), ".Archive", "cur/child:2,S");
    for storage in ["cur", "new", "tmp"] {
        add_maildir(root.path(), &format!("{storage}/ignored"), "new/not-mail");
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(root.path(), root.path().join("loop")).unwrap();
    let all = read_maildir(root.path(), 100).unwrap();
    assert_eq!(all.emails.len(), 2);
    assert_eq!(
        all.emails[0].path,
        root.path().join(".Archive/cur/child:2,S")
    );
    assert_eq!(all.emails[1].path, root.path().join("new/root"));
}

#[test]
fn root_without_maildirs_or_with_incomplete_nested_maildir_is_an_error() {
    let root = tempfile::tempdir().unwrap();
    assert!(read_maildir(root.path(), 100).is_err());
    add_maildir(root.path(), "INBOX", "new/a");
    fs::create_dir_all(root.path().join("Broken/cur")).unwrap();
    assert!(read_maildir(root.path(), 100).is_err());
}

#[test]
fn malformed_message_is_reported_without_blocking_other_messages() {
    let root = fixture();
    fs::write(
        root.path().join("new/bad"),
        b"Subject: Broken\r\nContent-Transfer-Encoding: base64\r\n\r\n%%%%",
    )
    .unwrap();
    fs::write(root.path().join("new/good"), b"Subject: Good\r\n\r\nHello")
        .unwrap();
    let mailbox = read_maildir(root.path(), 100).unwrap();
    assert_eq!(mailbox.emails.len(), 1);
    assert_eq!(mailbox.failures.len(), 1);
    assert_eq!(mailbox.failures[0].path, root.path().join("new/bad"));
    assert!(mailbox.failures[0].error.contains("Base64"));
}
