use vs1_email::split_body;

#[test]
fn chunks_cover_all_text_on_utf8_boundaries_and_fit_budget() {
    let body = "Olá mundo.\n\nRecebido: ação 😀. Final sentence.";
    let chunks =
        split_body(body, &mut |s| Ok(s.chars().count() <= 16)).unwrap();
    assert!(chunks.len() > 1);
    assert_eq!(chunks.concat(), body);
    assert!(
        chunks
            .iter()
            .all(|s| !s.is_empty() && s.chars().count() <= 16)
    );
    assert!(chunks[0].ends_with(char::is_whitespace));
}

#[test]
fn chunks_handle_empty_bodies_unbroken_strings_and_insufficient_budget() {
    assert_eq!(split_body("", &mut |_| Ok(true)).unwrap(), [""]);
    let body = "x".repeat(40);
    let chunks = split_body(&body, &mut |s| Ok(s.len() <= 7)).unwrap();
    assert_eq!(chunks.concat(), body);
    assert!(chunks.iter().all(|s| s.len() <= 7));
    assert!(split_body("x", &mut |_| Ok(false)).is_err());
    assert!(split_body("x", &mut |s| Ok(s.is_empty())).is_err());
}
