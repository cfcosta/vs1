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

impl CuaS1Input {
    /// Builds a text prompt using the Qwen3.5 `tokenizer.json` tokenizer.
    /// Requires 1..=26 options, a nonempty tree, and one token per letter.
    pub fn encode(
        tokenizer: &Tokenizer,
        options: &[CuaS1Option],
        app: &str,
        task_family: &str,
        ax_tree: &str,
        goal: Option<&str>,
    ) -> Result<Self> {
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
            "Cua-S1 requires between 1 and 26 options".into(),
        ));
    }
    if ax_tree.is_empty() {
        return Err(SystemOneError::Config(
            "Cua-S1 text modality requires ax_tree".into(),
        ));
    }
    let letters: Vec<char> = ('A'..='Z').take(options.len()).collect();
    let option_lines = letters
        .iter()
        .zip(options)
        .map(|(&letter, option)| describe_option(letter, option))
        .collect::<Vec<_>>()
        .join("\n");
    let mut user = String::new();
    if let Some(goal) = goal.filter(|s| !s.is_empty()) {
        user.push_str(&format!("Goal: {goal}\n\n"));
    }
    user.push_str(&format!(
        "App: {app}\nTask family: {task_family}\n\n\
         Accessibility tree:\n{ax_tree}\n\n\
         Options:\n{option_lines}\n\nAnswer with a single letter."
    ));
    Ok((
        format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n"
        ),
        letters,
    ))
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

    use super::*;

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
            build_prompt(&cases[0].options, "app", "task", "", None).is_err()
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
