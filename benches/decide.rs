//! Latency of `system_one` at the shapes docbert cares about: one
//! question about one state, a handful of questions about one state,
//! and the retrieval shape of one relevance question about each of
//! thirty candidate passages.
//!
//! The bench loads `convaiinnovations/laya` from the Hugging Face
//! cache and is skipped unless `VS1_BENCH=1` is set,
//! so a plain `cargo bench` on a box without the model stays quick.
//! On CUDA builds the model runs on GPU 0, in BF16 unless
//! `VS1_BENCH_DTYPE=f32`; otherwise on the CPU in F32.
//! `VS1_BENCH_SUBFOLDER=multilingual` benches the
//! mmBERT-base checkpoint instead.
//!
//! The thirty-candidate case is also swept over batch sizes, since
//! that is the knob `SystemOneBuilder::with_batch_size` exposes.

use std::{hint::black_box, time::Duration};

use candle_core::{DType, Device};
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
use vs1::{Question, State, SystemOne, SystemOneRequest};

const PASSAGE: &str = "sync selects the changed files in indexing.rs. It finds \
    the files with walker::discover_files, calculates a new collection Merkle \
    snapshot from these files, compares this snapshot with the previous \
    snapshot in config.db, and divides the paths into three groups: new \
    files, changed files, and deleted files. It changes the deleted paths \
    into stable document IDs and indexes and embeds only the new and changed \
    files, removing the data for the deleted ones. The Merkle snapshot diff \
    controls which files docbert selects; mtime is kept as metadata for each \
    document but is not used to find the changed files.";

fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).expect("bench built with cuda needs GPU 0")
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn load_model(batch_size: Option<usize>) -> SystemOne {
    let mut builder =
        SystemOne::from(vs1::DEFAULT_REPO_ID).with_device(device());
    if let Ok(subfolder) = std::env::var("VS1_BENCH_SUBFOLDER") {
        builder = builder.with_subfolder(subfolder);
    }
    if let Some(batch_size) = batch_size {
        builder = builder.with_batch_size(batch_size);
    }
    match std::env::var("VS1_BENCH_DTYPE").as_deref() {
        Ok("f32") => builder = builder.with_dtype(DType::F32),
        Ok("bf16") => builder = builder.with_dtype(DType::BF16),
        Ok("f16") => builder = builder.with_dtype(DType::F16),
        _ => {}
    }
    builder
        .try_into()
        .expect("failed to load convaiinnovations/laya; is it in the HF cache?")
}

fn relevance_question() -> Question {
    Question::noul_with_criteria(
        "Does the passage contain the information the search query is looking for?",
        "the passage directly answers or discusses what the query asks about",
        "the passage is about something else, or only shares a few words with the query",
    )
}

fn candidate(i: usize) -> SystemOneRequest {
    let state = State::Json(serde_json::json!({
        "query": "how does docbert sync decide which files changed",
        "title": "Pipeline",
        "passage": format!("Candidate {i}. {PASSAGE}"),
    }));
    SystemOneRequest::new(state).question("relevant", relevance_question())
}

fn triage_request() -> SystemOneRequest {
    SystemOneRequest::new(PASSAGE)
        .question("relevant", relevance_question())
        .question(
            "topic",
            Question::choice(
                "What is the passage about?",
                [
                    ("indexing", "how files become searchable"),
                    ("search", "how queries are answered"),
                    ("storage", "where data lives on disk"),
                    ("other", "none of the above"),
                ],
            ),
        )
        .question(
            "detail",
            Question::score(
                "How detailed is the passage?",
                ["a one-line summary", "a paragraph", "a full specification"],
            ),
        )
        .question(
            "mentions_mtime",
            Question::noul("Does the passage mention mtime?"),
        )
        .question(
            "mentions_pdf",
            Question::noul("Does the passage mention PDF files?"),
        )
}

fn bench_decide(c: &mut Criterion) {
    if std::env::var_os("VS1_BENCH").is_none() {
        eprintln!("VS1_BENCH is not set; skipping model benchmarks");
        return;
    }
    let model = load_model(None);
    eprintln!(
        "benching {} on {:?} as {:?}, batch size {}",
        model.model_name(),
        model.device(),
        model.dtype(),
        model.batch_size()
    );

    let one = candidate(0);
    let five = triage_request();
    let thirty: Vec<SystemOneRequest> = (0..30).map(candidate).collect();

    // Warm up kernels and the tokenizer once.
    model.system_one_batch(&thirty).expect("warmup");

    let mut group = c.benchmark_group("system_one");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    group.throughput(Throughput::Elements(1));
    group.bench_with_input(BenchmarkId::new("questions", 1), &one, |b, req| {
        b.iter(|| black_box(model.system_one(black_box(req)).unwrap()));
    });

    group.throughput(Throughput::Elements(five.questions.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("questions", five.questions.len()),
        &five,
        |b, req| {
            b.iter(|| black_box(model.system_one(black_box(req)).unwrap()));
        },
    );

    group.throughput(Throughput::Elements(thirty.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("candidates", thirty.len()),
        &thirty,
        |b, reqs| {
            b.iter(|| {
                black_box(model.system_one_batch(black_box(reqs)).unwrap())
            });
        },
    );
    group.finish();
    drop(model);

    let mut group = c.benchmark_group("system_one_batch_size");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    group.throughput(Throughput::Elements(thirty.len() as u64));
    for batch_size in [8usize, 16, 32] {
        let model = load_model(Some(batch_size));
        model.system_one_batch(&thirty).expect("warmup");
        group.bench_with_input(
            BenchmarkId::new("candidates_30", batch_size),
            &thirty,
            |b, reqs| {
                b.iter(|| {
                    black_box(model.system_one_batch(black_box(reqs)).unwrap())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decide);
criterion_main!(benches);
