use anyhow::{Result, ensure};
use vs1::{State, SystemOne, SystemOneRequest};

/// Nonoverlapping UTF-8 chunks, preferring whitespace boundaries. `fits` must
/// measure the whole request, including the repeated headers and owner context.
pub fn split_body<'a>(
    body: &'a str,
    fits: &mut impl FnMut(&str) -> Result<bool>,
) -> Result<Vec<&'a str>> {
    ensure!(fits("")?, "email metadata exceeds the model state budget");
    if body.is_empty() {
        return Ok(vec![body]);
    }
    let mut chunks = Vec::new();
    let mut rest = body;
    while !rest.is_empty() {
        // Bound tokenization work for exceptionally large bodies/HTML exports.
        let mut boundaries = rest
            .char_indices()
            .take(8192)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if boundaries.len() < 8192 {
            boundaries.push(rest.len());
        }
        let mut low = 0;
        let mut high = boundaries.len() - 1;
        while low < high {
            let mid = (low + high).div_ceil(2);
            if fits(&rest[..boundaries[mid]])? {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let mut end = boundaries[low];
        ensure!(
            end > 0,
            "email metadata leaves no room for even one body character"
        );
        if end < rest.len()
            && let Some((i, c)) = rest[..end]
                .char_indices()
                .rev()
                .find(|(i, c)| *i >= end / 2 && c.is_whitespace())
        {
            end = i + c.len_utf8();
        }
        // Token counts can change non-monotonically at word boundaries.
        while !fits(&rest[..end])? {
            end = rest[..end].char_indices().next_back().map_or(0, |(i, _)| i);
            ensure!(
                end > 0,
                "cannot fit a nonempty chunk in the model state budget"
            );
        }
        chunks.push(&rest[..end]);
        rest = &rest[end..];
    }
    Ok(chunks)
}

/// Checks exact state token capacity after the model's question header.
/// No message chunk is accepted if build_sequence would truncate its state.
pub fn request_fits(
    model: &SystemOne,
    request: &SystemOneRequest,
) -> Result<bool> {
    let empty = model.encode_state(&State::from(""))?;
    let state = model.encode_state(&request.state)?;
    for (id, question) in &request.questions {
        let overhead = model.build_sequence(&empty, id, question)?.ids.len();
        if overhead + state.len() > model.config().max_len {
            return Ok(false);
        }
    }
    Ok(true)
}
