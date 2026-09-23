use std::path::Path;

use candle_core::{DType, Device};
use serde::Deserialize;
use vs1::{
    CuaS1,
    CuaS1Builder,
    CuaS1Option,
    CuaS1OptionPrediction,
    DecisionModel,
    Question,
    SystemOneError,
    SystemOneRequest,
    cua_s1::{
        ADAPTER_REVISION,
        BASE_REPO_ID,
        BASE_REVISION,
        DEFAULT_REPO_ID,
        MODEL_NAME,
    },
};

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
struct Reference {
    base_model: String,
    base_revision: String,
    adapter_repo: String,
    adapter_revision: String,
    cases: Vec<ReferenceCase>,
}

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    options: Vec<ReferenceOption>,
}

#[derive(Deserialize)]
struct ReferenceOption {
    #[serde(flatten)]
    option: CuaS1Option,
    letter: char,
    probability: f32,
}

fn parse_reference() -> Reference {
    serde_json::from_str(include_str!(
        "../../../research/cua-s1/probabilities-f32.json"
    ))
    .unwrap()
}

fn parse_wikipedia_case() -> Case {
    #[derive(Deserialize)]
    struct BrowserDecision {
        app: String,
        task_family: String,
        ax_tree: String,
        goal: Option<String>,
        options: Vec<(CuaS1Option, serde::de::IgnoredAny)>,
    }

    let decision: BrowserDecision = serde_json::from_str(include_str!(
        "../../../research/cua-s1/wikipedia-56-options.json"
    ))
    .unwrap();
    Case {
        name: "wikipedia-56-options".into(),
        app: decision.app,
        task_family: decision.task_family,
        ax_tree: decision.ax_tree,
        goal: decision.goal,
        options: decision
            .options
            .into_iter()
            .map(|(option, _)| option)
            .collect(),
    }
}

#[test]
fn parses_wikipedia_browser_options() {
    let case = parse_wikipedia_case();
    assert_eq!(case.options.len(), 56);
    assert_eq!(case.task_family, "web_navigation");
    assert!(!case.ax_tree.is_empty());
    assert!(case.goal.is_some());
    assert_eq!(case.options[0].element_id, "e1");
    assert_eq!(case.options[0].action, "click");
    assert_eq!(case.options[55].element_id, "BLOCKED");
}

#[test]
fn pins_revisions_to_the_reference() {
    let reference = parse_reference();
    assert_eq!(BASE_REPO_ID, reference.base_model);
    assert_eq!(BASE_REVISION, reference.base_revision);
    assert_eq!(DEFAULT_REPO_ID, reference.adapter_repo);
    assert_eq!(ADAPTER_REVISION, reference.adapter_revision);
    let prompts: serde_json::Value = serde_json::from_str(include_str!(
        "../../../research/cua-s1/prompts.json"
    ))
    .unwrap();
    assert_eq!(BASE_REVISION, prompts["tokenizer_revision"]);
}

#[test]
fn rejects_unsupported_dtypes_before_loading() {
    for dtype in [DType::BF16, DType::F16, DType::F64, DType::U32] {
        let builder: CuaS1Builder = CuaS1::from("unused/repo")
            .with_device(Device::Cpu)
            .with_dtype(dtype);
        let Err(SystemOneError::Config(message)) = CuaS1::try_from(builder)
        else {
            panic!("expected a configuration error for {dtype:?}");
        };
        assert!(message.contains("only CPU with F32"), "{message}");
    }
}

#[test]
fn reports_missing_local_files_without_downloading() {
    let missing = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("Cargo.toml/not-a-directory");
    let result = CuaS1::try_from(
        CuaS1::from("unused/repo").with_local_directories(&missing, &missing),
    );
    assert!(matches!(result, Err(SystemOneError::Io(_))));
}

#[test]
#[ignore = "requires pinned artifacts/cua-s1 weights, about 20 GB RAM"]
fn answers_system_one_from_local_checkpoints_on_cpu() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/cua-s1");
    let model: CuaS1 = CuaS1::from(DEFAULT_REPO_ID)
        .with_device(Device::Cpu)
        .with_local_directories(
            root.join("base").join(BASE_REVISION),
            root.join("adapter").join(ADAPTER_REVISION).join("text"),
        )
        .try_into()
        .unwrap();
    assert!(model.device().is_cpu());
    assert_eq!(model.dtype(), DType::F32);
    let request = SystemOneRequest::new("The invoice is marked PAID.")
        .question(
            "status",
            Question::choice(
                "What is the invoice status?",
                [("paid", "paid invoice"), ("unpaid", "unpaid invoice")],
            ),
        );
    let response = model.system_one(&request).unwrap();
    assert_eq!(response.model, MODEL_NAME);
    assert_eq!(response.answers.len(), 1);
    assert!(response.usage.input_tokens > 0);
    let vs1::Answer::Choice(answer) = &response.answers["status"] else {
        panic!("choice")
    };
    assert!(answer.probabilities.contains_key(&answer.choice));
    assert!((answer.probabilities.values().sum::<f32>() - 1.0).abs() < 1e-6);
    assert!((0.0..=1.0).contains(&answer.confidence));
    assert!(answer.action.is_none());

    let model: DecisionModel = model.into();
    assert_eq!(model.model_name(), MODEL_NAME);
    assert!(model.local().is_none());
    assert!(model.system_one_batch(&[]).unwrap().is_empty());
    let empty = SystemOneRequest::new("No questions.");
    assert!(model.system_one(&empty).unwrap().answers.is_empty());
    let responses = model.system_one_batch(&[empty, request]).unwrap();
    assert_eq!(responses.len(), 2);
    assert!(responses[0].answers.is_empty());
    assert_eq!(responses[0].usage.input_tokens, 0);
    assert_eq!(responses[1], response);
}

#[test]
#[ignore = "requires pinned artifacts/cua-s1 weights, about 20 GB RAM"]
fn scores_all_reference_cases_from_local_checkpoints() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/cua-s1");
    let model: CuaS1 = CuaS1::from(DEFAULT_REPO_ID)
        .with_local_directories(
            root.join("base").join(BASE_REVISION),
            root.join("adapter").join(ADAPTER_REVISION).join("text"),
        )
        .try_into()
        .unwrap();
    assert_eq!(model.model_name(), MODEL_NAME);
    let cases: Vec<Case> = serde_json::from_str(include_str!(
        "../../../research/cua-s1/cases.json"
    ))
    .unwrap();
    let reference = parse_reference();
    assert_eq!(cases.len(), 6);
    assert_eq!(cases.len(), reference.cases.len());
    let mut max_probability_difference = 0f32;
    let mut has_matching_top_options = true;
    for (case, expected) in cases.iter().zip(&reference.cases) {
        assert_eq!(case.name, expected.name);
        let predictions: Vec<CuaS1OptionPrediction> = model
            .score_options(
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
                &case.options,
            )
            .unwrap();
        assert_eq!(predictions.len(), case.options.len());
        assert_eq!(predictions.len(), expected.options.len());
        let mut probability_difference = 0f32;
        for ((prediction, option), expected) in
            predictions.iter().zip(&case.options).zip(&expected.options)
        {
            assert_eq!(prediction.letter, expected.letter);
            assert_eq!(
                serde_json::to_value(&prediction.option).unwrap(),
                serde_json::to_value(option).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&prediction.option).unwrap(),
                serde_json::to_value(&expected.option).unwrap()
            );
            assert!(prediction.logit.is_finite());
            assert!((0.0..=1.0).contains(&prediction.probability));
            probability_difference = probability_difference
                .max((prediction.probability - expected.probability).abs());
        }
        let total: f32 = predictions.iter().map(|p| p.probability).sum();
        assert!((total - 1.0).abs() < 1e-6);
        max_probability_difference =
            max_probability_difference.max(probability_difference);
        let top = predictions
            .iter()
            .max_by(|a, b| a.probability.total_cmp(&b.probability))
            .unwrap()
            .letter;
        let expected_top = expected
            .options
            .iter()
            .max_by(|a, b| a.probability.total_cmp(&b.probability))
            .unwrap()
            .letter;
        has_matching_top_options &= top == expected_top;
        println!(
            "{}: top {top} (expected {expected_top}), max probability difference {probability_difference:e}",
            case.name
        );
    }
    println!(
        "all 6 cases: max probability difference {max_probability_difference:e}"
    );
    assert!(
        has_matching_top_options,
        "top option differs from reference"
    );
    assert!(
        max_probability_difference <= 1e-4,
        "probabilities exceed atol=1e-4"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA"]
fn scores_all_bf16_reference_cases_on_cuda() {
    let device = Device::new_cuda(0).unwrap();
    let result = CuaS1::try_from(
        CuaS1::from("unused/repo")
            .with_device(device.clone())
            .with_dtype(DType::F32),
    );
    let Err(SystemOneError::Config(message)) = result else {
        panic!("expected CUDA/F32 to be rejected before loading");
    };
    assert!(message.contains("12 GB"), "{message}");
    assert!(message.contains("use BF16"), "{message}");

    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/cua-s1");
    let model: CuaS1 = CuaS1::from(DEFAULT_REPO_ID)
        .with_device(device)
        .with_local_directories(
            root.join("base").join(BASE_REVISION),
            root.join("adapter").join(ADAPTER_REVISION).join("text"),
        )
        .try_into()
        .unwrap();
    let cases: Vec<Case> = serde_json::from_str(include_str!(
        "../../../research/cua-s1/cases.json"
    ))
    .unwrap();
    let reference: Reference = serde_json::from_str(include_str!(
        "../../../research/cua-s1/probabilities-bf16.json"
    ))
    .unwrap();
    assert_eq!(cases.len(), 6);
    assert_eq!(cases.len(), reference.cases.len());
    let mut max_probability_difference = 0f32;
    let mut has_matching_top_options = true;
    for (case, expected) in cases.iter().zip(&reference.cases) {
        assert_eq!(case.name, expected.name);
        let predictions = model
            .score_options(
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
                &case.options,
            )
            .unwrap();
        assert_eq!(predictions.len(), expected.options.len());
        let mut probability_difference = 0f32;
        for (prediction, expected) in predictions.iter().zip(&expected.options)
        {
            assert_eq!(prediction.letter, expected.letter);
            assert!(prediction.logit.is_finite());
            assert!((0.0..=1.0).contains(&prediction.probability));
            probability_difference = probability_difference
                .max((prediction.probability - expected.probability).abs());
        }
        let total: f32 = predictions.iter().map(|p| p.probability).sum();
        assert!((total - 1.0).abs() < 1e-6);
        max_probability_difference =
            max_probability_difference.max(probability_difference);
        let top = predictions
            .iter()
            .max_by(|a, b| a.probability.total_cmp(&b.probability))
            .unwrap()
            .letter;
        let expected_top = expected
            .options
            .iter()
            .max_by(|a, b| a.probability.total_cmp(&b.probability))
            .unwrap()
            .letter;
        has_matching_top_options &= top == expected_top;
        println!(
            "{}: top {top} (expected {expected_top}), max probability difference {probability_difference:e}",
            case.name
        );
    }
    println!(
        "all 6 cases: max probability difference {max_probability_difference:e}"
    );
    assert!(
        has_matching_top_options,
        "top option differs from BF16 reference"
    );
    assert!(
        max_probability_difference <= 0.03,
        "probabilities exceed atol=0.03"
    );
}

#[test]
#[ignore = "requires CUDA and the cua-s1 checkpoint"]
fn measures_wikipedia_scoring_on_cuda() {
    use std::{hint::black_box, time::Instant};

    let iterations = std::env::var_os("VS1_CUA_S1_BENCH_ITERATIONS")
        .map(|value| value.to_str().unwrap().parse::<usize>().unwrap())
        .unwrap_or(10);
    assert!(
        iterations > 0,
        "VS1_CUA_S1_BENCH_ITERATIONS must be positive"
    );
    let mut case = parse_wikipedia_case();
    if let Some(value) = std::env::var_os("VS1_CUA_S1_BENCH_OPTIONS") {
        let option_count = value.to_str().unwrap().parse::<usize>().unwrap();
        assert!(
            option_count > 0,
            "VS1_CUA_S1_BENCH_OPTIONS must be positive"
        );
        case.options.truncate(option_count);
    }

    let device = Device::new_cuda(0).unwrap();
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/cua-s1");
    let model: CuaS1 = CuaS1::from(DEFAULT_REPO_ID)
        .with_device(device.clone())
        .with_dtype(DType::BF16)
        .with_local_directories(
            root.join("base").join(BASE_REVISION),
            root.join("adapter").join(ADAPTER_REVISION).join("text"),
        )
        .try_into()
        .unwrap();
    let score = || {
        model
            .score_options(
                &case.app,
                &case.task_family,
                &case.ax_tree,
                case.goal.as_deref(),
                black_box(&case.options),
            )
            .unwrap()
    };
    let mut predictions = Vec::new();
    for _ in 0..2 {
        device.synchronize().unwrap();
        predictions = score();
        device.synchronize().unwrap();
    }
    let expected = serde_json::to_value(&predictions).unwrap();
    let mut latencies_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        device.synchronize().unwrap();
        let started = Instant::now();
        let actual = score();
        device.synchronize().unwrap();
        latencies_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(
            serde_json::to_value(&actual).unwrap(),
            expected,
            "Wikipedia scoring changed between calls"
        );
    }
    let mut sorted_latencies_ms = latencies_ms.clone();
    sorted_latencies_ms.sort_by(f64::total_cmp);
    let median_ms = (sorted_latencies_ms[(iterations - 1) / 2]
        + sorted_latencies_ms[iterations / 2])
        / 2.0;
    predictions.sort_by(|a, b| b.probability.total_cmp(&a.probability));
    let chosen = predictions.iter().find(|p| p.is_selected).unwrap();
    let report = serde_json::json!({
        "case": case.name,
        "device": "cuda:0",
        "dtype": "bf16",
        "base_revision": BASE_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "max_len": model.max_len(),
        "warmup_calls": 2,
        "iterations": iterations,
        "option_count": case.options.len(),
        "median_ms": median_ms,
        "latencies_ms": latencies_ms,
        "forward_passes": chosen.forward_passes,
        "chosen_option": chosen,
        "top_options": predictions.iter().take(5).collect::<Vec<_>>(),
    });
    let json = serde_json::to_string_pretty(&report).unwrap();
    println!("{json}");
    if let Some(path) = std::env::var_os("VS1_CUA_S1_BENCH_OUTPUT") {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, format!("{json}\n")).unwrap();
    }
}
