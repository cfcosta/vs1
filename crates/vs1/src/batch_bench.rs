//! Opt-in batch profiling and paired benchmarks. Run alone with one test thread.
use std::{collections::BTreeMap, time::Instant};

use candle_core::cuda_backend::cudarc::driver::{
    CudaEvent,
    sys::CUevent_flags,
};
use serde_json::{Value, json};

use super::*;

pub(super) fn cases() -> Vec<(String, Vec<SystemOneRequest>)> {
    let paragraph = "The indexing pipeline compares content hashes and updates changed documents. Unchanged files are skipped. ";
    let candidate = |i, repeats| {
        SystemOneRequest::new(format!(
            "Document {i}. {}",
            paragraph.repeat(repeats)
        ))
        .question(
            "relevant",
            Question::noul("Does this explain how changed files are selected?"),
        )
    };
    let mut cases = vec![];
    for n in [1, 8, 32, 64, 128] {
        cases.push((format!("{n}"), (0..n).map(|i| candidate(i, 5)).collect()));
    }
    cases.push((
        "mixed128".into(),
        (0..128)
            .map(|i| candidate(i, [1, 3, 9, 20][i % 4]))
            .collect(),
    ));
    let mut shared = SystemOneRequest::new(paragraph.repeat(5));
    for i in 0..128 {
        shared = shared.question(format!("q{i}"), Question::noul(format!("Question {i}: Does this explain how changed files are selected?")));
    }
    cases.push(("shared128".into(), vec![shared]));
    for (name, fixture) in [
        (
            "browser_call3",
            include_str!("../tests/fixtures/jev/call3_request.json"),
        ),
        (
            "browser_call5",
            include_str!("../tests/fixtures/jev/call5_request.json"),
        ),
    ] {
        cases.push((name.into(), vec![serde_json::from_str(fixture).unwrap()]));
    }
    cases
}

struct Span {
    name: &'static str,
    host_ms: f64,
    events: Option<(CudaEvent, CudaEvent)>,
}

#[derive(Default)]
struct Profile(Vec<Span>);
impl Profile {
    fn stage<T>(
        &mut self,
        name: &'static str,
        device: Option<&Device>,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let stream = device.map(|d| d.as_cuda_device().unwrap().cuda_stream());
        let start = stream.as_ref().map(|s| {
            s.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap()
        });
        let time = Instant::now();
        let output = f()?;
        let host_ms = time.elapsed().as_secs_f64() * 1000.0;
        let events = start.map(|start| {
            (
                start,
                stream
                    .unwrap()
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap(),
            )
        });
        self.0.push(Span {
            name,
            host_ms,
            events,
        });
        Ok(output)
    }
    fn finish(
        self,
        device: &Device,
    ) -> Result<BTreeMap<&'static str, [f64; 2]>> {
        device.synchronize()?;
        let mut totals = BTreeMap::new();
        for span in self.0 {
            let total = totals.entry(span.name).or_insert([0.0, 0.0]);
            total[0] += span.host_ms;
            if let Some((start, end)) = span.events {
                total[1] += start.elapsed_ms(&end).unwrap() as f64;
            }
        }
        Ok(totals)
    }
}

// Mirror the packed forward's stage boundaries without instrumentation in
// production. The test checks each resulting logit/action against forward_batch.
fn profile_batch(
    model: &SystemOne,
    items: &[&EncodedItem],
    p: &mut Profile,
) -> Result<Vec<ItemOutput>> {
    let dev = &model.device;
    let c = p.stage("collate_cpu", None, || {
        Ok(collate(items, model.special.pad))
    })?;
    let ids = p.stage("input_upload", Some(dev), || {
        Ok(Tensor::from_slice(&c.input_ids, (c.batch, c.seq_len), dev)?)
    })?;
    let hidden = p.stage("encoder", Some(dev), || {
        Ok(model.encoder.forward_varlen_packed(&ids, &c.lens)?)
    })?;
    let (total, hidden_size) = hidden.dims2()?;
    let (hidden, logits, offsets) =
        p.stage("decision_head", Some(dev), || {
            let kinds: Vec<u32> = c
                .kinds
                .iter()
                .zip(&c.lens)
                .flat_map(|(&k, &n)| std::iter::repeat_n(k, n))
                .collect();
            let kinds = Tensor::from_vec(kinds, total, dev)?;
            let mut hidden =
                (hidden + model.type_emb.index_select(&kinds, 0)?)?;
            let (seqlens, max_len) =
                crate::modernbert::cumulative_seqlens(&c.lens, dev)?;
            for layer in &model.head {
                hidden = layer.forward_packed(&hidden, &seqlens, max_len)?;
            }
            let mut offsets = Vec::with_capacity(c.batch);
            let mut start = 0u32;
            for &len in &c.lens {
                offsets.push(start);
                start += len as u32;
            }
            let markers: Vec<u32> = c
                .marker_pos
                .chunks(c.max_markers.max(1))
                .zip(&offsets)
                .flat_map(|(row, &offset)| row.iter().map(move |&m| offset + m))
                .collect();
            let markers =
                Tensor::from_vec(markers, c.batch * c.max_markers, dev)?;
            let marked = hidden.index_select(&markers, 0)?.reshape((
                c.batch,
                c.max_markers,
                hidden_size,
            ))?;
            Ok((hidden, model.scorer.forward(&marked)?, offsets))
        })?;
    let rows =
        p.stage(
            "logits_readback",
            Some(dev),
            || Ok(logits.to_vec2::<f32>()?),
        )?;
    let pooled = p.stage("pooled_gather", Some(dev), || {
        let indices = Tensor::from_vec(offsets, c.batch, dev)?;
        Ok(hidden
            .index_select(&indices, 0)?
            .to_dtype(DType::F32)?
            .contiguous()?)
    })?;
    let features: Vec<f32> = p.stage("confidence_cpu", None, || {
        Ok(rows
            .iter()
            .zip(&c.option_counts)
            .flat_map(|(row, &k)| action_features(row, k))
            .collect())
    })?;
    let act = p.stage("action_head_upload", Some(dev), || {
        let features = Tensor::from_vec(
            features,
            (c.batch, crate::head::ACTION_FEATURES),
            dev,
        )?;
        Ok(model.act_head.forward(&pooled, &features)?)
    })?;
    let act =
        p.stage("actions_readback", Some(dev), || Ok(act.to_vec2::<f32>()?))?;
    Ok(rows
        .into_iter()
        .zip(&c.option_counts)
        .zip(act)
        .map(|((row, &k), act)| ItemOutput {
            logits: row[..k].to_vec(),
            act_probability: act.first().copied().unwrap_or(0.0),
        })
        .collect())
}

fn model() -> anyhow::Result<SystemOne> {
    Ok(SystemOne::from(crate::DEFAULT_REPO_ID)
        .with_device(Device::new_cuda(0)?)
        .with_dtype(DType::BF16)
        .try_into()?)
}

fn snapshot(responses: &[SystemOneResponse]) -> Value {
    json!({"json": responses, "actions": responses.iter().map(|r|
        r.answers.values().map(|a| a.action().map(|a| a.act_probability)).collect::<Vec<_>>()
    ).collect::<Vec<_>>()})
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

fn run_paired(
    reference: &'static std::sync::atomic::AtomicBool,
) -> anyhow::Result<()> {
    use std::sync::atomic::Ordering;
    struct Reset(&'static std::sync::atomic::AtomicBool);
    impl Drop for Reset {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Relaxed);
        }
    }
    let _reset = Reset(reference);
    let model = model()?;
    let mut report = vec![];
    for (name, requests) in cases() {
        reference.store(true, Ordering::Relaxed);
        let expected = snapshot(&model.system_one_batch(&requests)?);
        for warmup in 0..6 {
            reference.store(warmup % 2 == 0, Ordering::Relaxed);
            assert_eq!(snapshot(&model.system_one_batch(&requests)?), expected);
        }
        let (mut baseline, mut candidate, mut ratios) =
            (vec![], vec![], vec![]);
        let mut faster = 0;
        for iteration in 0..40 {
            let mut pair = [0.0; 2];
            for index in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                reference.store(index == 0, Ordering::Relaxed);
                model.device.synchronize()?;
                let time = Instant::now();
                let response = model.system_one_batch(&requests)?;
                model.device.synchronize()?;
                pair[index] = time.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(
                    snapshot(&response),
                    expected,
                    "{name}, pair {iteration}, variant {index}"
                );
            }
            faster += usize::from(pair[1] < pair[0]);
            baseline.push(pair[0]);
            candidate.push(pair[1]);
            ratios.push(pair[1] / pair[0]);
        }
        let row = json!({"name":name, "baseline_p50_ms":median(&mut baseline),
            "candidate_p50_ms":median(&mut candidate), "paired_change_percent":100.0 * (median(&mut ratios) - 1.0),
            "faster_pairs":faster, "pairs":40, "outputs_exact":true});
        eprintln!("{row}");
        report.push(row);
    }
    eprintln!("PAIRED_REPORT={}", serde_json::to_string(&report)?);
    Ok(())
}

#[test]
#[ignore = "requires CUDA and checkpoint; run alone"]
fn paired_preparation_latency() -> anyhow::Result<()> {
    run_paired(&REFERENCE_PREP)
}

#[test]
#[ignore = "requires CUDA and checkpoint; run alone"]
fn parallel_preparation_preserves_sequences_and_errors() -> anyhow::Result<()> {
    let model = model()?;
    for (_, requests) in cases() {
        assert_eq!(
            model.prepare_items(&requests)?,
            model.prepare_items_serial(&requests)?
        );
    }
    assert!(PREP_POOL.get().is_some_and(Option::is_some));
    // Errors in different requests and in different questions of one request
    // must still select the first failure in the original iteration order.
    let mut many = SystemOneRequest::new("state");
    for i in 0..32 {
        many = many.question(
            format!("invalid-{i}"),
            Question::choice("Invalid", [("only", "single option")]),
        );
    }
    let separate: Vec<_> = (0..32)
        .map(|i| {
            SystemOneRequest::new("state").question(
                format!("invalid-{i}"),
                Question::choice("Invalid", [("only", "single option")]),
            )
        })
        .collect();
    for requests in [vec![many], separate] {
        assert_eq!(
            model.prepare_items(&requests).unwrap_err().to_string(),
            model
                .prepare_items_serial(&requests)
                .unwrap_err()
                .to_string()
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA and checkpoint; run alone"]
fn preparation_cpu_latency() -> anyhow::Result<()> {
    let model = model()?;
    let mut report = vec![];
    for (name, requests) in cases() {
        let expected = model.prepare_items_serial(&requests)?;
        for _ in 0..5 {
            assert_eq!(model.prepare_items(&requests)?, expected);
        }
        let (mut serial, mut parallel, mut ratios) = (vec![], vec![], vec![]);
        for i in 0..40 {
            let mut pair = [0.0; 2];
            for index in if i % 2 == 0 { [0, 1] } else { [1, 0] } {
                let time = Instant::now();
                let result = if index == 0 {
                    model.prepare_items_serial(&requests)?
                } else {
                    model.prepare_items(&requests)?
                };
                pair[index] = time.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(result, expected);
            }
            serial.push(pair[0]);
            parallel.push(pair[1]);
            ratios.push(pair[1] / pair[0]);
        }
        report.push(json!({"name":name, "serial_p50_ms":median(&mut serial),
            "parallel_p50_ms":median(&mut parallel), "paired_change_percent":100.0 * (median(&mut ratios) - 1.0)}));
    }
    eprintln!("PREPARATION_REPORT={}", serde_json::to_string(&report)?);
    Ok(())
}

#[test]
#[ignore = "requires CUDA and checkpoint; run alone"]
fn profile_batch_stages() -> anyhow::Result<()> {
    let model = model()?;
    let mut report = vec![];
    for (name, requests) in cases() {
        for _ in 0..5 {
            model.system_one_batch(&requests)?;
        }
        let mut samples = BTreeMap::<&str, Vec<[f64; 2]>>::new();
        for _ in 0..10 {
            let mut p = Profile::default();
            let (items, order) = p.stage("prepare_cpu", None, || {
                let mut items = vec![];
                for request in &requests {
                    let state = model.encode_state(&request.state)?;
                    for (id, q) in &request.questions {
                        items.push(model.build_sequence(&state, id, q)?);
                    }
                }
                let mut order: Vec<_> = (0..items.len()).collect();
                order.sort_by_key(|&i| std::cmp::Reverse(items[i].ids.len()));
                Ok((items, order))
            })?;
            for chunk in order.chunks(model.batch_size) {
                let batch: Vec<_> = chunk.iter().map(|&i| &items[i]).collect();
                let actual = profile_batch(&model, &batch, &mut p)?;
                let expected = model.forward_batch(&batch)?;
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.logits, b.logits);
                    assert_eq!(
                        a.act_probability.to_bits(),
                        b.act_probability.to_bits()
                    );
                }
            }
            for (stage, values) in p.finish(&model.device)? {
                samples.entry(stage).or_default().push(values);
            }
        }
        let stages: BTreeMap<_, _> = samples
            .into_iter()
            .map(|(stage, values)| {
                let mean = |i| {
                    values.iter().map(|v| v[i]).sum::<f64>()
                        / values.len() as f64
                };
                (stage, json!({"host_ms":mean(0), "cuda_stream_ms":mean(1)}))
            })
            .collect();
        let row = json!({"name":name, "stages":stages});
        eprintln!("{row}");
        report.push(row);
    }
    eprintln!("PROFILE_REPORT={}", serde_json::to_string(&report)?);
    Ok(())
}
