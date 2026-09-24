use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::{Result, SystemOneError};

const SYSTEM_PROMPT: &str = concat!(
    "You are a one-pass computer-use decision model. You are shown the ",
    "current state of a screen and a fixed, closed list of candidate ",
    "(element, action) options, each given a single letter. Choose exactly ",
    "one option: the single best next action to take. Answer with ONLY that ",
    "option's letter -- no words, no punctuation, no explanation."
);

/// One candidate (element, action), matching `cua_s1.four_b.Option`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuaS1Option {
    pub element_id: String,
    pub role: String,
    pub label: String,
    pub action: String,
    pub entity_id: Option<String>,
}

/// Tokens for one forward pass, with answer letters in option order.
#[derive(Debug, Clone, Serialize)]
pub struct CuaS1Input {
    pub chat_text: String,
    pub input_ids: Vec<u32>,
    pub letters: Vec<char>,
    pub letter_ids: Vec<u32>,
}

pub(super) struct PromptPrefix {
    chat_text: String,
    pub input_ids: Vec<u32>,
}

impl PromptPrefix {
    pub(super) fn encode(
        tokenizer: &Tokenizer,
        app: &str,
        task_family: &str,
        ax_tree: &str,
        goal: Option<&str>,
    ) -> Result<Self> {
        let chat_text = build_prompt_prefix(app, task_family, ax_tree, goal);
        let input_ids = tokenizer
            .encode(chat_text.as_str(), true)?
            .get_ids()
            .to_vec();
        Ok(Self {
            chat_text,
            input_ids,
        })
    }

    pub(super) fn encode_suffix(
        &self,
        tokenizer: &Tokenizer,
        input: &CuaS1Input,
        dropped_state_tokens: usize,
    ) -> Result<Option<Vec<u32>>> {
        if dropped_state_tokens > 0 || self.input_ids.is_empty() {
            return Ok(None);
        }
        let Some(suffix) = input.chat_text.strip_prefix(&self.chat_text) else {
            return Ok(None);
        };
        let suffix_ids = tokenizer.encode(suffix, false)?.get_ids().to_vec();
        if suffix_ids.is_empty()
            || !self
                .input_ids
                .iter()
                .chain(&suffix_ids)
                .eq(&input.input_ids)
        {
            return Ok(None);
        }
        Ok(Some(suffix_ids))
    }
}

impl CuaS1Input {
    /// Builds a text prompt using the Qwen3.5 `tokenizer.json` tokenizer.
    /// Requires 1..=26 options per pass, a nonempty tree, and one token per letter.
    pub fn encode(
        tokenizer: &Tokenizer,
        options: &[CuaS1Option],
        app: &str,
        task_family: &str,
        ax_tree: &str,
        goal: Option<&str>,
    ) -> Result<Self> {
        if ax_tree.is_empty() {
            return Err(SystemOneError::Config(
                "Cua-S1 text modality requires ax_tree".into(),
            ));
        }
        let (chat_text, letters) =
            build_prompt(options, app, task_family, ax_tree, goal)?;
        let letter_ids = encode_letters(tokenizer, &letters)?;
        // Python calls tokenizer(chat_text) with special tokens enabled.
        let input_ids = tokenizer
            .encode(chat_text.as_str(), true)?
            .get_ids()
            .to_vec();
        Ok(Self {
            chat_text,
            input_ids,
            letters,
            letter_ids,
        })
    }

    /// Drops only the state's tail, then checks the complete chat prompt.
    pub(super) fn encode_with_max_len(
        tokenizer: &Tokenizer,
        options: &[CuaS1Option],
        app: &str,
        task_family: &str,
        ax_tree: &str,
        goal: Option<&str>,
        max_len: usize,
    ) -> Result<(Self, usize)> {
        let mut input =
            Self::encode(tokenizer, options, app, task_family, ax_tree, goal)?;
        if input.input_ids.len() <= max_len {
            return Ok((input, 0));
        }
        let (empty_prompt, _) =
            build_prompt(options, app, task_family, "", goal)?;
        let empty_len = tokenizer.encode(empty_prompt.as_str(), true)?.len();
        if empty_len > max_len {
            return Err(SystemOneError::Question {
                id: task_family.into(),
                reason: format!(
                    "Cua-S1 prompt requires {empty_len} tokens even with an empty state, exceeding max_len ({max_len}); shorten the goal or options, or increase max_len"
                ),
            });
        }
        let state = tokenizer.encode(ax_tree, false)?;
        let state_ids = state.get_ids();
        let mut kept = (max_len - empty_len).min(state_ids.len());
        loop {
            let prefix = tokenizer.decode(&state_ids[..kept], false)?;
            // A byte-level token prefix can end inside a Unicode character.
            if !ax_tree.starts_with(&prefix) {
                kept = kept.saturating_sub(1);
                continue;
            }
            let (chat_text, _) =
                build_prompt(options, app, task_family, &prefix, goal)?;
            let input_ids = tokenizer
                .encode(chat_text.as_str(), true)?
                .get_ids()
                .to_vec();
            if input_ids.len() <= max_len {
                input.chat_text = chat_text;
                input.input_ids = input_ids;
                return Ok((input, state_ids.len() - kept));
            }
            kept = kept.saturating_sub(input_ids.len() - max_len);
        }
    }
}

fn describe_option(letter: char, option: &CuaS1Option) -> String {
    let mut action = option.action.clone();
    if option.action == "fill"
        && let Some(entity) =
            option.entity_id.as_deref().filter(|s| !s.is_empty())
    {
        action.push_str(&format!(" (with entity '{entity}')"));
    }
    format!("{letter}. {} \"{}\" -> {action}", option.role, option.label)
}

/// Hard-codes Qwen/Qwen3.5-4B's chat template for exactly [system, user]
/// text messages with add_generation_prompt=True and default thinking.
fn build_prompt(
    options: &[CuaS1Option],
    app: &str,
    task_family: &str,
    ax_tree: &str,
    goal: Option<&str>,
) -> Result<(String, Vec<char>)> {
    if !(1..=26).contains(&options.len()) {
        return Err(SystemOneError::Config(
            "Cua-S1 prompts require between 1 and 26 options per pass; use CuaS1::score_options for larger sets".into(),
        ));
    }
    let letters: Vec<char> = ('A'..='Z').take(options.len()).collect();
    let option_lines = letters
        .iter()
        .zip(options)
        .map(|(&letter, option)| describe_option(letter, option))
        .collect::<Vec<_>>()
        .join("\n");
    let prefix = build_prompt_prefix(app, task_family, ax_tree, goal);
    Ok((
        format!(
            "{prefix}{option_lines}\n\nAnswer with a single letter.<|im_end|>\n\
         <|im_start|>assistant\n<think>\n"
        ),
        letters,
    ))
}

fn build_prompt_prefix(
    app: &str,
    task_family: &str,
    ax_tree: &str,
    goal: Option<&str>,
) -> String {
    let mut user = String::new();
    if let Some(goal) = goal.filter(|s| !s.is_empty()) {
        user.push_str(&format!("Goal: {goal}\n\n"));
    }
    user.push_str(&format!(
        "App: {app}\nTask family: {task_family}\n\n\
         Accessibility tree:\n{ax_tree}\n\n\
         Options:\n"
    ));
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{user}"
    )
}

fn encode_letters(tokenizer: &Tokenizer, letters: &[char]) -> Result<Vec<u32>> {
    letters.iter().map(|letter| {
        let encoded = tokenizer.encode(letter.to_string(), false)?;
        match encoded.get_ids() {
            [id] => Ok(*id),
            ids => Err(SystemOneError::Tokenizer(format!(
                "letter {letter:?} must encode to exactly one token, got {}",
                ids.len()
            ))),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tokenizers::{
        models::bpe::{BPE, Vocab},
        pre_tokenizers::byte_level::ByteLevel,
    };

    use super::*;

    fn build_byte_tokenizer() -> Tokenizer {
        let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let model = BPE::builder()
            .vocab_and_merges(
                alphabet
                    .into_iter()
                    .enumerate()
                    .map(|(i, c)| (c.to_string(), i as u32))
                    .collect::<Vocab>(),
                vec![],
            )
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        app: String,
        task_family: String,
        ax_tree: String,
        goal: Option<String>,
        options: Vec<CuaS1Option>,
    }

    #[derive(Deserialize)]
    struct Fixtures {
        tokenizer_revision: String,
        cases: Vec<PromptFixture>,
    }

    #[derive(Deserialize)]
    struct PromptFixture {
        name: String,
        chat_text: String,
        input_ids: Vec<u32>,
        letters: Vec<char>,
        letter_ids: Vec<u32>,
    }

    fn fixtures() -> (Vec<Case>, Fixtures) {
        let cases: Vec<Case> = serde_json::from_str(include_str!(
            "../../../../research/cua-s1/cases.json"
        ))
        .unwrap();
        let fixtures: Fixtures = serde_json::from_str(include_str!(
            "../../../../research/cua-s1/prompts.json"
        ))
        .unwrap();
        assert_eq!(cases.len(), 6);
        assert_eq!(cases.len(), fixtures.cases.len());
        (cases, fixtures)
    }

    #[test]
    fn upstream_chat_text_and_letter_order() {
        let (cases, fixtures) = fixtures();
        for (case, expected) in cases.iter().zip(&fixtures.cases) {
            assert_eq!(case.name, expected.name);
            let (text, letters) = build_prompt(
                &case.options,
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
            )
            .unwrap();
            assert_eq!(text, expected.chat_text, "{}", case.name);
            assert_eq!(letters, expected.letters, "{}", case.name);
        }
    }

    fn assert_prompt_prefix_splits(tokenizer: &Tokenizer) {
        let (cases, _) = fixtures();
        for case in cases {
            let state = format!("{}\n\nOptions:\nSão Paulo 🙂", case.ax_tree);
            let goal = Some("Choose.\n\nOptions:\nThese are part of the goal.");
            let prefix = PromptPrefix::encode(
                tokenizer,
                &case.app,
                &case.task_family,
                &state,
                goal,
            )
            .unwrap();
            assert!(prefix.chat_text.ends_with("\n\nOptions:\n"));
            for count in [1, 2, 13, 14, 18, 19, 26] {
                let mut options: Vec<_> =
                    case.options.iter().cycle().take(count).cloned().collect();
                options[0].label.push_str("\nOptions:\ninside an option");
                let input = CuaS1Input::encode(
                    tokenizer,
                    &options,
                    &case.app,
                    &case.task_family,
                    &state,
                    goal,
                )
                .unwrap();
                let suffix_ids = prefix
                    .encode_suffix(tokenizer, &input, 0)
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    [&prefix.input_ids[..], &suffix_ids].concat(),
                    input.input_ids
                );
                let suffix = &input.chat_text[prefix.chat_text.len()..];
                assert!(suffix.starts_with("A. "));
                assert!(suffix.ends_with(
                    "\n\nAnswer with a single letter.<|im_end|>\n\
                     <|im_start|>assistant\n<think>\n"
                ));
                // Even identical text cannot bypass the truncation guard.
                assert!(
                    prefix
                        .encode_suffix(tokenizer, &input, 1)
                        .unwrap()
                        .is_none()
                );
                let (truncated, dropped) = CuaS1Input::encode_with_max_len(
                    tokenizer,
                    &options,
                    &case.app,
                    &case.task_family,
                    &state,
                    goal,
                    input.input_ids.len() - 1,
                )
                .unwrap();
                assert!(dropped > 0);
                assert!(
                    prefix
                        .encode_suffix(tokenizer, &truncated, dropped)
                        .unwrap()
                        .is_none()
                );
                assert!(
                    prefix
                        .encode_suffix(tokenizer, &truncated, 0)
                        .unwrap()
                        .is_none()
                );
                let mut mismatched = input.clone();
                mismatched.input_ids[0] += 1;
                assert!(
                    prefix
                        .encode_suffix(tokenizer, &mismatched, 0)
                        .unwrap()
                        .is_none()
                );
            }
        }
    }

    #[test]
    fn splits_only_identical_untruncated_prompt_prefixes() {
        assert_prompt_prefix_splits(&build_byte_tokenizer());
    }

    #[test]
    #[ignore = "requires the pinned local Qwen3.5 tokenizer in artifacts/cua-s1/base"]
    fn splits_prompt_prefixes_with_the_pinned_tokenizer() {
        let (cases, fixtures) = fixtures();
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/cua-s1/base")
            .join(&fixtures.tokenizer_revision)
            .join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(path).unwrap();
        assert_prompt_prefix_splits(&tokenizer);
        for (case, expected) in cases.iter().zip(&fixtures.cases) {
            let input = CuaS1Input::encode(
                &tokenizer,
                &case.options,
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
            )
            .unwrap();
            let prefix = PromptPrefix::encode(
                &tokenizer,
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
            )
            .unwrap();
            let suffix_ids = prefix
                .encode_suffix(&tokenizer, &input, 0)
                .unwrap()
                .unwrap();
            assert_eq!(
                [&prefix.input_ids[..], &suffix_ids].concat(),
                expected.input_ids
            );
            // Splitting inside a BPE token changes the concatenated token IDs.
            let chat_text = "<|im_start|>s".to_string();
            let prefix = PromptPrefix {
                input_ids: tokenizer
                    .encode(chat_text.as_str(), true)
                    .unwrap()
                    .get_ids()
                    .to_vec(),
                chat_text,
            };
            assert!(
                prefix
                    .encode_suffix(&tokenizer, &input, 0)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn leaves_short_prompts_untouched() {
        let tokenizer = build_byte_tokenizer();
        let (cases, _) = fixtures();
        for case in cases {
            let full = CuaS1Input::encode(
                &tokenizer,
                &case.options,
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
            )
            .unwrap();
            for max_len in [full.input_ids.len(), full.input_ids.len() + 100] {
                let (input, dropped) = CuaS1Input::encode_with_max_len(
                    &tokenizer,
                    &case.options,
                    &case.app,
                    &case.task_family,
                    &case.ax_tree,
                    case.goal.as_deref(),
                    max_len,
                )
                .unwrap();
                assert_eq!(
                    serde_json::to_value(input).unwrap(),
                    serde_json::to_value(&full).unwrap()
                );
                assert_eq!(dropped, 0);
            }
        }
    }

    #[test]
    fn truncates_only_the_state_to_fit_the_complete_prompt() {
        let tokenizer = build_byte_tokenizer();
        let (cases, _) = fixtures();
        let options = &cases[0].options;
        let state = "Saved message. ".repeat(200);
        let goal = Some("Save this message.");
        let (empty, letters) =
            build_prompt(options, "mail", "save", "", goal).unwrap();
        let empty_len = tokenizer.encode(empty.as_str(), true).unwrap().len();
        for kept in [0, 1, 37] {
            let max_len = empty_len + kept;
            let (input, dropped) = CuaS1Input::encode_with_max_len(
                &tokenizer, options, "mail", "save", &state, goal, max_len,
            )
            .unwrap();
            let (expected, _) =
                build_prompt(options, "mail", "save", &state[..kept], goal)
                    .unwrap();
            assert_eq!(input.input_ids.len(), max_len);
            assert_eq!(input.chat_text, expected);
            assert_eq!(input.letters, letters);
            assert_eq!(
                input.letter_ids,
                encode_letters(&tokenizer, &letters).unwrap()
            );
            assert_eq!(
                dropped,
                tokenizer.encode(state.as_str(), false).unwrap().len() - kept
            );
            assert_eq!(
                input.input_ids,
                tokenizer.encode(expected.as_str(), true).unwrap().get_ids()
            );
        }
    }

    #[test]
    fn rejects_limits_smaller_than_the_prompt_with_empty_state() {
        let tokenizer = build_byte_tokenizer();
        let (cases, _) = fixtures();
        let options = &cases[0].options;
        let (empty, _) =
            build_prompt(options, "mail", "save", "", Some("Save.")).unwrap();
        let empty_len = tokenizer.encode(empty.as_str(), true).unwrap().len();
        for max_len in [0, empty_len - 1] {
            let error = CuaS1Input::encode_with_max_len(
                &tokenizer,
                options,
                "mail",
                "save",
                "message",
                Some("Save."),
                max_len,
            )
            .unwrap_err();
            let SystemOneError::Question { id, reason } = error else {
                panic!("expected a per-question error");
            };
            assert_eq!(id, "save");
            assert!(reason.contains(&format!(
                "requires {empty_len} tokens even with an empty state"
            )));
            assert!(reason.contains(&format!("max_len ({max_len})")));
        }
    }

    #[test]
    fn truncates_at_unicode_boundaries() {
        let tokenizer = build_byte_tokenizer();
        let (cases, _) = fixtures();
        let options = &cases[0].options;
        let (empty, _) =
            build_prompt(options, "mail", "save", "", None).unwrap();
        let empty_len = tokenizer.encode(empty.as_str(), true).unwrap().len();
        for budget in [1, 3, 4, 5] {
            let (input, dropped) = CuaS1Input::encode_with_max_len(
                &tokenizer,
                options,
                "mail",
                "save",
                "🙂🙂",
                None,
                empty_len + budget,
            )
            .unwrap();
            let retained = if budget < 4 { "" } else { "🙂" };
            assert_eq!(
                input.chat_text,
                build_prompt(options, "mail", "save", retained, None)
                    .unwrap()
                    .0
            );
            assert!(input.input_ids.len() <= empty_len + budget);
            assert_eq!(dropped, 8 - retained.len());
        }
    }

    #[test]
    #[ignore = "requires the pinned local Qwen3.5 tokenizer in artifacts/cua-s1/base"]
    fn bounds_prompts_with_the_pinned_tokenizer() {
        let (cases, fixtures) = fixtures();
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/cua-s1/base")
            .join(&fixtures.tokenizer_revision)
            .join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(path).unwrap();
        for (case, expected) in cases.iter().zip(fixtures.cases) {
            let (short, dropped) = CuaS1Input::encode_with_max_len(
                &tokenizer,
                &case.options,
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
                expected.input_ids.len(),
            )
            .unwrap();
            assert_eq!(short.chat_text, expected.chat_text);
            assert_eq!(short.input_ids, expected.input_ids);
            assert_eq!(dropped, 0);
            let state = format!(
                "{} {}",
                case.ax_tree,
                "Long email. São Paulo 🙂 東京\n".repeat(1000)
            );
            let (empty, _) = build_prompt(
                &case.options,
                &case.app,
                &case.task_family,
                "",
                case.goal.as_deref(),
            )
            .unwrap();
            let empty_len =
                tokenizer.encode(empty.as_str(), true).unwrap().len();
            for max_len in [empty_len, empty_len + 1, empty_len + 37, 4096] {
                let (input, dropped) = CuaS1Input::encode_with_max_len(
                    &tokenizer,
                    &case.options,
                    &case.app,
                    &case.task_family,
                    &state,
                    case.goal.as_deref(),
                    max_len,
                )
                .unwrap();
                assert!(input.input_ids.len() <= max_len);
                assert!(dropped > 0);
                let retained = input
                    .chat_text
                    .split_once("Accessibility tree:\n")
                    .unwrap()
                    .1
                    .split_once("\n\nOptions:\n")
                    .unwrap()
                    .0;
                assert!(state.starts_with(retained));
                assert_eq!(
                    input.chat_text,
                    build_prompt(
                        &case.options,
                        &case.app,
                        &case.task_family,
                        retained,
                        case.goal.as_deref()
                    )
                    .unwrap()
                    .0
                );
                assert_eq!(
                    input.input_ids,
                    tokenizer
                        .encode(input.chat_text.as_str(), true)
                        .unwrap()
                        .get_ids()
                );
                assert_eq!(input.letters, expected.letters);
                assert_eq!(input.letter_ids, expected.letter_ids);
                if max_len == empty_len {
                    assert!(retained.is_empty());
                    assert_eq!(
                        dropped,
                        tokenizer.encode(state.as_str(), false).unwrap().len()
                    );
                }
            }
            assert!(matches!(
                CuaS1Input::encode_with_max_len(
                    &tokenizer,
                    &case.options,
                    &case.app,
                    &case.task_family,
                    &state,
                    case.goal.as_deref(),
                    empty_len - 1
                ),
                Err(SystemOneError::Question { .. })
            ));
        }
    }

    #[test]
    #[ignore = "requires the pinned local Qwen3.5 tokenizer in artifacts/cua-s1/base"]
    fn upstream_token_ids() {
        let (cases, fixtures) = fixtures();
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../artifacts/cua-s1/base")
            .join(&fixtures.tokenizer_revision)
            .join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(path).unwrap();
        for (case, expected) in cases.iter().zip(&fixtures.cases) {
            assert_eq!(case.name, expected.name);
            let input = CuaS1Input::encode(
                &tokenizer,
                &case.options,
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
            )
            .unwrap();
            assert_eq!(input.chat_text, expected.chat_text, "{}", case.name);
            assert_eq!(input.letters, expected.letters, "{}", case.name);
            assert_eq!(input.input_ids, expected.input_ids, "{}", case.name);
            assert_eq!(input.letter_ids, expected.letter_ids, "{}", case.name);
        }
    }

    #[test]
    fn option_limits_and_empty_tree() {
        let (cases, _) = fixtures();
        for count in [0, 1, 26, 27] {
            let options = vec![cases[0].options[0].clone(); count];
            let result = build_prompt(&options, "app", "task", "tree", None);
            if count == 0 || count == 27 {
                assert!(result.is_err());
            } else {
                let (text, letters) = result.unwrap();
                assert_eq!(letters.len(), count);
                assert_eq!(letters[0], 'A');
                assert_eq!(
                    letters[count - 1],
                    if count == 26 { 'Z' } else { 'A' }
                );
                assert!(text.contains(&format!(
                    "{}. button \"Save\" -> click",
                    letters[count - 1]
                )));
            }
        }
        assert!(
            CuaS1Input::encode(
                &build_byte_tokenizer(),
                &cases[0].options,
                "app",
                "task",
                "",
                None
            )
            .is_err()
        );
    }

    #[test]
    fn empty_goal_and_entity_fields() {
        let (cases, _) = fixtures();
        let options = &cases[0].options;
        assert_eq!(
            build_prompt(options, "app", "task", "tree", None).unwrap(),
            build_prompt(options, "app", "task", "tree", Some("")).unwrap(),
        );
        let mut option = cases[1].options[0].clone();
        for entity_id in [None, Some(String::new())] {
            option.entity_id = entity_id;
            assert_eq!(
                describe_option('A', &option),
                "A. textbox \"Email\" -> fill"
            );
        }
        option.action = "click".into();
        option.entity_id = Some("account_email".into());
        assert_eq!(
            describe_option('A', &option),
            "A. textbox \"Email\" -> click"
        );
    }

    #[test]
    fn rejects_letters_with_zero_or_multiple_tokens() {
        use tokenizers::{
            models::bpe::{BPE, Vocab},
            normalizers::replace::Replace,
        };

        let vocab: Vocab = [("x".to_string(), 0)].into_iter().collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        for (replacement, count) in [("", 0), ("xx", 2)] {
            tokenizer
                .with_normalizer(Some(Replace::new("A", replacement).unwrap()))
                .unwrap();
            let error = encode_letters(&tokenizer, &['A']).unwrap_err();
            assert!(error.to_string().contains(&format!("got {count}")));
        }
    }
}
