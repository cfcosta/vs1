//! Turning a `(state, question)` pair into the token sequence the
//! decision head reads.
//!
//! The layout, reproduced from laya's `build_sequence`, is:
//!
//! ```text
//! [CLS] <kind> question: <instructions> [SEP]
//! [MASK] option 0 [MASK] option 1 ... [SEP]
//! <state> [SEP]
//! ```
//!
//! Each `[MASK]` marks where one option's score is read out. The
//! instructions and options share a `head_max_len` token budget; the
//! state gets whatever is left of `max_len` and is cut on the right.

use tokenizers::Tokenizer;

use crate::{
    error::{Result, SystemOneError},
    types::{Question, QuestionKind, State},
};

/// Longest tokenised option, before the budget squeeze kicks in.
const OPTION_TOKEN_CAP: usize = 48;

/// Below this many spare header tokens, options are squeezed.
const MIN_INSTRUCTION_BUDGET: usize = 16;

/// Instructions always keep at least this many tokens.
const MIN_INSTRUCTION_TOKENS: usize = 8;

/// Shortest an option can be squeezed to (its marker plus three
/// tokens).
const MIN_SQUEEZED_OPTION_TOKENS: usize = 4;

/// Ids of the special tokens the layout needs, plus the mask string
/// that is scrubbed out of user text so it cannot forge a marker.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub mask_text: String,
}

impl SpecialTokens {
    /// Resolves the four special tokens in `tokenizer`.
    pub fn resolve(
        tokenizer: &Tokenizer,
        cls: &str,
        sep: &str,
        mask: &str,
        pad: &str,
    ) -> Result<Self> {
        let id = |token: &str| {
            tokenizer.token_to_id(token).ok_or_else(|| {
                SystemOneError::Config(format!(
                    "special token {token:?} is not in the tokenizer vocabulary"
                ))
            })
        };
        Ok(Self {
            cls: id(cls)?,
            sep: id(sep)?,
            mask: id(mask)?,
            pad: id(pad)?,
            mask_text: mask.to_string(),
        })
    }

    /// Replaces literal mask tokens in user text with a space.
    pub fn scrub(&self, text: &str) -> String {
        text.replace(&self.mask_text, " ")
    }
}

/// A state tokenised once, shared by every question asked of it.
#[derive(Debug, Clone)]
pub struct EncodedState {
    ids: Vec<u32>,
}

impl EncodedState {
    /// Tokenises `state` without special tokens.
    pub fn encode(
        tokenizer: &Tokenizer,
        special: &SpecialTokens,
        state: &State,
    ) -> Result<Self> {
        let text = special.scrub(&state.render());
        Ok(Self {
            ids: encode_plain(tokenizer, &text)?,
        })
    }

    /// Number of state tokens before any truncation.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// `true` when the state tokenised to nothing.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// One question's token sequence, ready to batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedItem {
    /// Token ids, at most `max_len` long.
    pub ids: Vec<u32>,
    /// Position of each option's `[MASK]` marker, in option order.
    pub markers: Vec<usize>,
    /// Which primitive this is.
    pub kind: QuestionKind,
}

impl EncodedItem {
    /// Number of options the sequence can score.
    pub fn option_count(&self) -> usize {
        self.markers.len()
    }
}

fn encode_plain(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer.encode(text, false)?.get_ids().to_vec())
}

/// Builds the sequence for one question over an already tokenised
/// state.
///
/// Fails when the options do not fit `head_max_len` even after being
/// squeezed, because a missing marker would leave an option unscored.
pub fn build_sequence(
    tokenizer: &Tokenizer,
    special: &SpecialTokens,
    state: &EncodedState,
    question_id: &str,
    question: &Question,
    max_len: usize,
    head_max_len: usize,
) -> Result<EncodedItem> {
    let kind = question.kind();
    let options = question.render_options();
    if options.len() < 2 {
        return Err(SystemOneError::Question {
            id: question_id.to_string(),
            reason: format!(
                "{} questions need at least two options, got {}",
                kind.as_str(),
                options.len()
            ),
        });
    }

    let instructions = special.scrub(&question.instructions().render());
    let mut head_ids = encode_plain(
        tokenizer,
        &format!("{} question: {instructions}", kind.as_str()),
    )?;

    let mut option_ids: Vec<Vec<u32>> = Vec::with_capacity(options.len());
    for option in &options {
        let text = format!(" {}", special.scrub(option));
        let mut ids = encode_plain(tokenizer, &text)?;
        ids.truncate(OPTION_TOKEN_CAP);
        ids.insert(0, special.mask);
        option_ids.push(ids);
    }

    let option_total =
        |ids: &[Vec<u32>]| ids.iter().map(Vec::len).sum::<usize>();
    let mut instruction_budget =
        head_max_len as isize - option_total(&option_ids) as isize;
    if instruction_budget < MIN_INSTRUCTION_BUDGET as isize {
        let per = ((head_max_len.saturating_sub(MIN_INSTRUCTION_BUDGET))
            / option_ids.len().max(1))
        .max(MIN_SQUEEZED_OPTION_TOKENS);
        for ids in &mut option_ids {
            ids.truncate(per);
        }
        instruction_budget =
            head_max_len as isize - option_total(&option_ids) as isize;
    }
    head_ids.truncate(
        instruction_budget.max(MIN_INSTRUCTION_TOKENS as isize) as usize
    );

    let mut ids = Vec::with_capacity(max_len);
    ids.push(special.cls);
    ids.extend_from_slice(&head_ids);
    ids.push(special.sep);
    let mut markers = Vec::with_capacity(option_ids.len());
    for option in &option_ids {
        markers.push(ids.len());
        ids.extend_from_slice(option);
    }
    ids.push(special.sep);

    let room = max_len.saturating_sub(ids.len() + 1);
    ids.extend_from_slice(&state.ids[..state.ids.len().min(room)]);
    ids.push(special.sep);
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);

    if markers.len() != options.len() {
        return Err(SystemOneError::Question {
            id: question_id.to_string(),
            reason: format!(
                "{} options exceed head_max_len={head_max_len}",
                options.len()
            ),
        });
    }

    Ok(EncodedItem { ids, markers, kind })
}

/// Padded batch tensors' worth of data, still on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collated {
    /// Items in the batch.
    pub batch: usize,
    /// Padded sequence length.
    pub seq_len: usize,
    /// Padded marker count.
    pub max_markers: usize,
    /// `batch * seq_len` token ids, pad id in the tail.
    pub input_ids: Vec<u32>,
    /// `batch * seq_len` ones and zeros.
    pub attention_mask: Vec<u32>,
    /// `batch * max_markers` positions, zero-filled past each item's
    /// option count.
    pub marker_pos: Vec<u32>,
    /// Real option count per item.
    pub option_counts: Vec<usize>,
    /// Real token count per item, before padding.
    pub lens: Vec<usize>,
    /// Primitive index per item.
    pub kinds: Vec<u32>,
}

/// Right-pads `items` into one batch.
pub fn collate(items: &[&EncodedItem], pad: u32) -> Collated {
    let batch = items.len();
    let seq_len = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
    let max_markers =
        items.iter().map(|it| it.markers.len()).max().unwrap_or(0);
    let mut input_ids = vec![pad; batch * seq_len];
    let mut attention_mask = vec![0u32; batch * seq_len];
    let mut marker_pos = vec![0u32; batch * max_markers];
    let mut option_counts = Vec::with_capacity(batch);
    let mut lens = Vec::with_capacity(batch);
    let mut kinds = Vec::with_capacity(batch);
    for (row, item) in items.iter().enumerate() {
        let start = row * seq_len;
        input_ids[start..start + item.ids.len()].copy_from_slice(&item.ids);
        attention_mask[start..start + item.ids.len()].fill(1);
        let mstart = row * max_markers;
        for (j, &m) in item.markers.iter().enumerate() {
            marker_pos[mstart + j] = m as u32;
        }
        option_counts.push(item.markers.len());
        lens.push(item.ids.len());
        kinds.push(item.kind.index() as u32);
    }
    Collated {
        batch,
        seq_len,
        max_markers,
        input_ids,
        attention_mask,
        marker_pos,
        option_counts,
        lens,
        kinds,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A word-level tokenizer over a fixed vocabulary, enough to check
    /// the layout without a real BPE model. Unknown words map to
    /// `[UNK]`.
    fn toy_tokenizer() -> Tokenizer {
        use tokenizers::{
            models::wordlevel::WordLevel,
            pre_tokenizers::whitespace::Whitespace,
        };
        let words = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[PAD]",
            "noul",
            "choice",
            "score",
            "question",
            ":",
            "is",
            "it",
            "raining",
            "?",
            "false",
            "true",
            "no",
            ",",
            "the",
            "statement",
            "does",
            "not",
            "hold",
            "yes",
            "holds",
            "sky",
            "rain",
            "wet",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "level",
            "0",
            "1",
            "2",
            "w",
        ];
        let vocab = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
    }

    fn specials(tokenizer: &Tokenizer) -> SpecialTokens {
        SpecialTokens::resolve(tokenizer, "[CLS]", "[SEP]", "[MASK]", "[PAD]")
            .unwrap()
    }

    fn id(tokenizer: &Tokenizer, token: &str) -> u32 {
        tokenizer.token_to_id(token).unwrap()
    }

    #[test]
    fn noul_layout_matches_laya() {
        let tok = toy_tokenizer();
        let sp = specials(&tok);
        let state =
            EncodedState::encode(&tok, &sp, &State::from("sky wet")).unwrap();
        let item = build_sequence(
            &tok,
            &sp,
            &state,
            "q",
            &Question::noul("is it raining ?"),
            512,
            192,
        )
        .unwrap();
        let t = |s: &str| id(&tok, s);
        let mut expected = vec![t("[CLS]")];
        expected.extend(
            ["noul", "question", ":", "is", "it", "raining", "?"].map(t),
        );
        expected.push(t("[SEP]"));
        let false_marker = expected.len();
        expected.push(t("[MASK]"));
        expected.extend(
            [
                "false",
                ":",
                "no",
                ",",
                "the",
                "statement",
                "does",
                "not",
                "hold",
            ]
            .map(t),
        );
        let true_marker = expected.len();
        expected.push(t("[MASK]"));
        expected.extend(
            ["true", ":", "yes", ",", "the", "statement", "holds"].map(t),
        );
        expected.push(t("[SEP]"));
        expected.extend(["sky", "wet"].map(t));
        expected.push(t("[SEP]"));
        assert_eq!(item.ids, expected);
        assert_eq!(item.markers, [false_marker, true_marker]);
        assert_eq!(item.kind, QuestionKind::Noul);
    }

    #[test]
    fn state_is_cut_on_the_right_to_fit_max_len() {
        let tok = toy_tokenizer();
        let sp = specials(&tok);
        let long = vec!["rain"; 100].join(" ");
        let state =
            EncodedState::encode(&tok, &sp, &State::from(long)).unwrap();
        let item = build_sequence(
            &tok,
            &sp,
            &state,
            "q",
            &Question::noul("is it raining ?"),
            40,
            192,
        )
        .unwrap();
        assert_eq!(item.ids.len(), 40);
        assert_eq!(*item.ids.last().unwrap(), sp.sep);
        // Header: [CLS] + 7 + [SEP] + 10 + 8 + [SEP] = 28 tokens, so
        // 40 - 28 - 1 = 11 state tokens survive.
        let rain = id(&tok, "rain");
        assert_eq!(item.ids.iter().filter(|&&i| i == rain).count(), 11);
    }

    #[test]
    fn options_are_squeezed_when_they_crowd_the_header() {
        let tok = toy_tokenizer();
        let sp = specials(&tok);
        let state =
            EncodedState::encode(&tok, &sp, &State::from("sky")).unwrap();
        let question: Question = serde_json::from_value(json!({
            "type": "choice",
            "instructions": "is it raining ?",
            "criteria": {
                "a": "w w w w w w w w w w",
                "b": "w w w w w w w w w w",
                "c": "w w w w w w w w w w",
                "d": "w w w w w w w w w w"
            }
        }))
        .unwrap();
        // 4 options x 13 tokens = 52 > 40 - 16, so every option is
        // squeezed to max(4, (40 - 16) / 4) = 6 tokens.
        let item =
            build_sequence(&tok, &sp, &state, "q", &question, 512, 40).unwrap();
        assert_eq!(item.markers.len(), 4);
        let gaps: Vec<usize> =
            item.markers.windows(2).map(|w| w[1] - w[0]).collect();
        assert_eq!(gaps, [6, 6, 6]);
        // The instructions kept their max(8, 40 - 24) = 16 budget,
        // which is more than the 7 tokens they have.
        assert_eq!(item.markers[0], 1 + 7 + 1);
    }

    #[test]
    fn instructions_keep_at_least_eight_tokens() {
        let tok = toy_tokenizer();
        let sp = specials(&tok);
        let state =
            EncodedState::encode(&tok, &sp, &State::from("sky")).unwrap();
        let long_instructions = vec!["w"; 30].join(" ");
        let question = Question::score(long_instructions, ["a", "b", "c"]);
        // Options take 3 x (1 + 3) = 12 tokens of a 20-token header,
        // leaving 8 for the instructions: exactly the floor.
        let item =
            build_sequence(&tok, &sp, &state, "q", &question, 512, 20).unwrap();
        assert_eq!(item.markers[0], 1 + 8 + 1);
    }

    #[test]
    fn too_many_options_for_the_budget_is_an_error() {
        let tok = toy_tokenizer();
        let sp = specials(&tok);
        let state =
            EncodedState::encode(&tok, &sp, &State::from("sky")).unwrap();
        let labels: Vec<String> = (0..12).map(|i| format!("w{i}")).collect();
        let question: Question = serde_json::from_value(json!({
            "type": "choice",
            "instructions": "?",
            "criteria": labels
        }))
        .unwrap();
        // Twelve two-token options need 24 header tokens; with
        // max_len = 20 the trailing markers fall off the end.
        let err = build_sequence(&tok, &sp, &state, "many", &question, 20, 16)
            .unwrap_err();
        assert!(
            matches!(err, SystemOneError::Question { ref id, .. } if id == "many")
        );
        assert!(err.to_string().contains("exceed head_max_len"));
    }

    #[test]
    fn mask_tokens_in_user_text_are_scrubbed() {
        let tok = toy_tokenizer();
        let sp = specials(&tok);
        let state =
            EncodedState::encode(&tok, &sp, &State::from("sky [MASK] wet"))
                .unwrap();
        let item = build_sequence(
            &tok,
            &sp,
            &state,
            "q",
            &Question::noul("[MASK] raining ?"),
            512,
            192,
        )
        .unwrap();
        assert_eq!(
            item.ids.iter().filter(|&&i| i == sp.mask).count(),
            2,
            "only the two option markers may be mask tokens"
        );
    }

    #[test]
    fn collate_pads_right_and_records_counts() {
        let a = EncodedItem {
            ids: vec![1, 2, 3],
            markers: vec![1, 2],
            kind: QuestionKind::Noul,
        };
        let b = EncodedItem {
            ids: vec![4, 5],
            markers: vec![1, 2, 3],
            kind: QuestionKind::Choice,
        };
        let c = collate(&[&a, &b], 99);
        assert_eq!((c.batch, c.seq_len, c.max_markers), (2, 3, 3));
        assert_eq!(c.input_ids, [1, 2, 3, 4, 5, 99]);
        assert_eq!(c.attention_mask, [1, 1, 1, 1, 1, 0]);
        assert_eq!(c.marker_pos, [1, 2, 0, 1, 2, 3]);
        assert_eq!(c.option_counts, [2, 3]);
        assert_eq!(c.kinds, [2, 0]);
    }
}
