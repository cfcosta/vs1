use std::path::Path;

use candle_core::{DType, Device};
use serde::Deserialize;
use vs1::{
    CuaS1,
    CuaS1Builder,
    CuaS1Option,
    CuaS1OptionPrediction,
    SystemOneError,
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
