mod browser;
mod model;
mod policy;
mod scenario;
mod verify;

use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use browser::{Browser, Stale};
use clap::{Args, Parser};
use model::Backend;
use serde_json::{Value, json};

#[derive(Args, Debug)]
pub struct ModelArgs {
    #[arg(long, default_value="local", value_parser=["local","typesafe"])]
    backend: String,
    #[arg(long, env = "VS1_DEVICE", default_value = "cpu")]
    device: String,
    #[arg(long, default_value=vs1::DEFAULT_REPO_ID)]
    checkpoint: String,
    #[arg(long, default_value = "")]
    subfolder: String,
    #[arg(long)]
    max_len: Option<usize>,
    #[arg(long)]
    head_max_len: Option<usize>,
}

/// A browser agent: one natural-language goal, indexed actions, in-process vs1 decisions.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Run a JSON scenario containing a plan, setup scripts, and outcome checks.
    #[arg(long, conflicts_with_all=["url","goal","task","replay","check_browser","expect_url","expect_text"])]
    scenario: Option<PathBuf>,
    #[arg(long, default_value="model", value_parser=["model","lexical"], requires="scenario")]
    chooser: String,
    #[arg(long, default_value="overlap", value_parser=["none","overlap"], requires="scenario")]
    retrieval: String,
    /// Starting page. Chrome must expose a debugging endpoint.
    #[arg(long)]
    url: Option<String>,
    /// Natural-language goal. Repeated values are joined into one goal.
    #[arg(long)]
    goal: Vec<String>,
    /// Optional reproducible task for measurement.
    #[arg(long, value_parser=["hotel","flights","wikipedia"])]
    task: Option<String>,
    #[arg(long, env = "CDP_URL", default_value = "http://127.0.0.1:9222")]
    cdp: String,
    #[command(flatten)]
    model: ModelArgs,
    /// Original Jev prompt or a shorter prompt for local checkpoint context limits.
    #[arg(long, default_value="compact", value_parser=["compact","upstream"])]
    prompt: String,
    /// Fresh output directory; traces may contain page content and entered text.
    #[arg(long, default_value = "artifacts/browser-run")]
    output: PathBuf,
    #[arg(long, default_value_t = 60)]
    max_steps: usize,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    /// Record CDP JPEG frames with their original timestamps.
    #[arg(long)]
    record: bool,
    /// Save a screenshot after every observation.
    #[arg(long)]
    screenshots: bool,
    /// Independent final URL equality check for a custom goal.
    #[arg(long)]
    expect_url: Option<String>,
    /// Independent final visible-text checks, all required.
    #[arg(long)]
    expect_text: Vec<String>,
    /// Benchmark a captured SystemOne request without touching the browser.
    #[arg(long)]
    replay: Option<PathBuf>,
    /// Exercise real DOM guards and mutations locally without loading a model.
    #[arg(long)]
    check_browser: bool,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    ensure!(
        args.repeat > 0 && args.max_steps > 0,
        "repeat and max-steps must be positive"
    );
    if let Some(path) = &args.scenario {
        return scenario::run(&args, path);
    }
    if args.check_browser {
        return verify::check_browser(&args.cdp);
    }
    ensure!(
        !args.output.exists(),
        "output already exists; choose a fresh --output directory"
    );
    fs::create_dir_all(&args.output)?;
    let task_input = if args.replay.is_none() {
        Some(task(&args)?)
    } else {
        None
    };
    let mut backend = Backend::load(&args.model)?;
    backend.warmup()?;
    eprintln!("Model: {}", backend.metadata);
    if let Some(path) = &args.replay {
        return replay(&args, &backend, path);
    }
    let (url, goal) = task_input.expect("validated task input");
    let mut summaries = vec![];
    let mut failed = false;
    for run in 0..args.repeat {
        let folder = args.output.join(format!("run-{:02}", run + 1));
        fs::create_dir(&folder)?;
        let summary = run_agent(&args, &backend, &url, &goal, &folder)?;
        failed |= summary["error"].is_string()
            || summary["status"] != "done"
            || summary["verification"]["passed"] == false;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        summaries.push(summary);
    }
    let report = json!({"configuration":backend.metadata,"prompt":args.prompt,"runs":summaries,
        "timing":"First prediction after initial observation through terminal decision or failure; excludes model load, warmup, initial navigation and independent final verification.",
        "historical_baselines":{"jev_optimized_flights_ms":7092,"jev_original_flights_ms":9450,"jev_hotel_smoke_ms":1896,"jev_wikipedia_smoke_ms":2798}});
    fs::write(
        args.output.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    ensure!(
        !failed,
        "one or more runs failed or did not satisfy independent verification; see summary.json"
    );
    Ok(())
}

fn task(args: &Cli) -> Result<(String, String)> {
    if let Some(task) = args.task.as_deref() {
        ensure!(
            args.url.is_none() && args.goal.is_empty(),
            "use either --task or --url/--goal"
        );
        return Ok(match task {
            "hotel" => {
                let path=args.output.join("fixture.html");
                fs::write(&path,include_str!("../assets/fixture.html"))?;
                (url::Url::from_file_path(path.canonicalize()?).map_err(|_|anyhow::anyhow!("invalid fixture path"))?.to_string(),
                    "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, then open Casa Flora.".into())
            }
            "flights" => ("https://www.google.com/travel/flights?hl=en".into(),
                "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. Stop when matching flight options are visible. Do not select or book a flight.".into()),
            "wikipedia" => ("https://en.wikipedia.org/wiki/Main_Page".into(),
                "Find and open the Wikipedia article about Gödel’s incompleteness theorems.".into()),
            _ => unreachable!(),
        });
    }
    let url = args
        .url
        .clone()
        .context("supply --url and --goal (or --task)")?;
    let parsed = url::Url::parse(&url)?;
    ensure!(
        ["http", "https", "file", "data", "about"].contains(&parsed.scheme()),
        "unsupported starting URL scheme"
    );
    let goal = args.goal.join("\n").trim().to_owned();
    ensure!(!goal.is_empty(), "supply a nonempty --goal");
    Ok((url, goal))
}

fn replay(args: &Cli, backend: &Backend, path: &Path) -> Result<()> {
    let body: Value = serde_json::from_slice(&fs::read(path)?)?;
    let requests = if let Some(a) = body.as_array() {
        a.clone()
    } else {
        vec![body]
    };
    let mut measurements = vec![];
    for (index, request) in requests.iter().enumerate() {
        let inspection = backend.inspect(request)?;
        // Shape-specific warmup is excluded and explicitly recorded separately.
        let warm = Instant::now();
        backend.decide(request)?;
        let shape_warmup_ms = warm.elapsed().as_secs_f64() * 1000.0;
        let mut times = vec![];
        let mut response = Value::Null;
        for _ in 0..args.repeat {
            let started = Instant::now();
            response = backend.decide(request)?;
            times.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        let n = times.len();
        let median = (times[(n - 1) / 2] + times[n / 2]) / 2.0;
        measurements.push(json!({"request":index,"latencies_ms":times,"median_ms":median,
            "shape_warmup_ms":shape_warmup_ms,"inspection":inspection,"response":response}));
    }
    let result = json!({"configuration":backend.metadata,"input":path,"runs":measurements});
    fs::write(
        args.output.join("replay.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_agent(
    args: &Cli,
    backend: &Backend,
    url: &str,
    goal: &str,
    folder: &Path,
) -> Result<Value> {
    let setup = Instant::now();
    let mut browser = Browser::connect(&args.cdp, url)?;
    run_agent_page(args, backend, goal, folder, &mut browser, setup, None)
}

fn run_agent_page(
    args: &Cli,
    backend: &Backend,
    goal: &str,
    folder: &Path,
    browser: &mut Browser,
    setup: Instant,
    verifier: Option<&[String]>,
) -> Result<Value> {
    let mut page = browser.observe()?;
    let url = page["url"].clone();
    if args.record {
        browser.start_recording(folder.join("screencast"))?;
    }
    if args.screenshots {
        browser.screenshot(&folder.join("000000.jpg"))?;
    }
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    browser.counts.clear();
    let epoch = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
    let started = Instant::now();
    let mut history = vec![];
    let mut decisions = vec![];
    let mut model_attempts = vec![];
    let mut text_calls = vec![];
    let mut pending_text: Option<(Value, String, Value)> = None;
    let mut status = "ready";
    let mut error = None;
    let mut pending_action = Value::Null;
    let mut event_log = fs::File::create(folder.join("events.jsonl"))?;
    use std::io::Write;
    for _ in 0..args.max_steps.saturating_mul(2) {
        let tick = (|| -> Result<()> {
            if !browser.fresh(&page, None)? {
                page = browser.observe()?;
            }
            let (request, space) = policy::request(
                &page,
                goal,
                &history,
                args.prompt == "compact",
            )?;
            let inference = Instant::now();
            let result = backend.decide(&request);
            let latency = inference.elapsed().as_secs_f64() * 1000.0;
            model_attempts.push(json!({"latency_ms":latency,"error":result.as_ref().err().map(|e|format!("{e:#}"))}));
            let response = result?;
            let mut decision = policy::resolve(&request, &space, &response)?;
            decision["latency_ms"] = json!(latency);
            decision["elapsed_ms"] =
                json!(started.elapsed().as_secs_f64() * 1000.0);
            decision["request"] = request;
            decision["fingerprint"] = page["fingerprint"].clone();
            decisions.push(decision.clone());
            let operation = decision["operation"].as_str().unwrap();
            if ["DONE", "BLOCKED"].contains(&operation) {
                if !browser.fresh(&page, None)? {
                    return Err(
                        Stale("page changed before terminal decision").into()
                    );
                }
                status = if operation == "DONE" {
                    "done"
                } else {
                    "blocked"
                };
                return Ok(());
            }
            ensure!(history.len() < args.max_steps, "action budget exhausted");
            let action = &decision["action"];
            let mut text = None;
            let mut helper = Value::Null;
            if action["kind"] == "fill" {
                if !browser.fresh(&page, None)? {
                    return Err(
                        Stale("page changed before text generation").into()
                    );
                }
                let context =
                    model::field_context(goal, action, &page, &history);
                if let Some((_, cached, metadata)) = pending_text
                    .as_ref()
                    .filter(|(input, _, _)| input == &context)
                {
                    text = Some(cached.clone());
                    helper = metadata.clone();
                } else {
                    let helper_started = Instant::now();
                    let (generated, metadata) = match backend
                        .field_text(&context)
                    {
                        Ok(result) => result,
                        Err(error) => {
                            text_calls.push(json!({"field":action["label"],"error":format!("{error:#}"),
                                "metadata":{"latency_ms":helper_started.elapsed().as_secs_f64()*1000.0}}));
                            return Err(error);
                        }
                    };
                    text_calls.push(json!({"field":action["label"],"value":generated,"metadata":metadata}));
                    pending_text =
                        Some((context, generated.clone(), metadata.clone()));
                    text = Some(generated);
                    helper = metadata;
                }
            }
            pending_action =
                json!({"action":action,"text":text,"execution":"attempted"});
            writeln!(event_log, "{}", pending_action)?;
            event_log.flush()?;
            browser.act(action, &page, text.as_deref())?;
            pending_action = Value::Null;
            pending_text = None;
            let entry = json!({"step":history.len()+1,"action":action["label"],"kind":action["kind"],"choice":action["id"],
                "operation":operation,"target":decision["target"],"text":text,"text_helper":helper,
                "latency_ms":latency,"executed_ms":started.elapsed().as_secs_f64()*1000.0,"page_changed":null});
            // Persist successful execution before observation or settling can fail.
            writeln!(
                event_log,
                "{}",
                json!({"execution":"confirmed","entry":entry})
            )?;
            event_log.flush()?;
            history.push(entry);
            let previous = page["fingerprint"].clone();
            page = browser.observe()?;
            let last = history.last_mut().unwrap();
            last["page_changed"] = json!(page["fingerprint"] != previous);
            if args.screenshots {
                browser.screenshot(&folder.join(format!(
                    "{:06}.jpg",
                    started.elapsed().as_millis()
                )))?;
            }
            eprintln!(
                "{:>7.0} ms  {:>2} actions  {} {}",
                started.elapsed().as_secs_f64() * 1000.0,
                history.len(),
                operation,
                action["label"].as_str().unwrap_or("")
            );
            if history.len() >= 3
                && history
                    .iter()
                    .rev()
                    .take(3)
                    .all(|h| h["page_changed"] == false && h["kind"] != "wait")
            {
                status = "blocked";
            }
            Ok(())
        })();
        if let Err(e) = tick {
            if e.downcast_ref::<Stale>().is_some() {
                pending_action = Value::Null; // Only pre-mutation rejections have this error type.
                match browser.observe() {
                    Ok(fresh) => page = fresh,
                    Err(e) => {
                        error = Some(e.to_string());
                        break;
                    }
                }
                continue;
            }
            error = Some(format!("{e:#}"));
            break;
        }
        if status != "ready" {
            break;
        }
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if status == "ready" && error.is_none() {
        error = Some("model-call budget exhausted".into());
    }
    if error.is_some() {
        status = "error";
    }
    let counts = browser.counts.clone();
    let final_page = browser.observe();
    let verification = match &final_page {
        Ok(page) => match verifier {
            Some(lines) => scenario::verification(browser, lines, page),
            None => verify::outcome(
                page,
                args.task.as_deref(),
                args.expect_url.as_deref(),
                &args.expect_text,
            ),
        },
        Err(e) => json!({"passed":false,"error":e.to_string()}),
    };
    let recording_error = browser.stop_recording().err().map(|e| e.to_string());
    let screenshot_error = browser
        .screenshot(&folder.join("final.jpg"))
        .err()
        .map(|e| e.to_string());
    let mut latencies: Vec<f64> = model_attempts
        .iter()
        .filter_map(|d| d["latency_ms"].as_f64())
        .collect();
    latencies.sort_by(f64::total_cmp);
    let median = if latencies.is_empty() {
        0.0
    } else {
        (latencies[(latencies.len() - 1) / 2] + latencies[latencies.len() / 2])
            / 2.0
    };
    let summary = json!({"status":status,"elapsed_ms":elapsed_ms,"setup_ms":setup_ms,"error":error,"verification":verification,
        "actions":history.len(),"decisions":model_attempts.len(),"accepted_responses":decisions.len(),"decision_median_ms":median,"decision_total_ms":latencies.iter().sum::<f64>(),
        "text_calls":text_calls.len(),"text_total_ms":text_calls.iter().filter_map(|c|c["metadata"]["latency_ms"].as_f64()).sum::<f64>(),
        "cdp_calls":counts.values().sum::<usize>(),"browser":browser.version(),"recording_error":recording_error,"screenshot_error":screenshot_error});
    let trace = json!({"summary":summary,"configuration":backend.metadata,"prompt":args.prompt,"url":url,"goal":goal,
        "started_epoch":epoch,"history":history,"decisions":decisions,"model_attempts":model_attempts,"text_calls":text_calls,"pending_action":pending_action,
        "page":page,"final_page":final_page.ok(),"cdp":counts});
    fs::write(
        folder.join("trace.json"),
        serde_json::to_vec_pretty(&trace)?,
    )?;
    Ok(summary)
}

#[cfg(test)]
mod scenario_cli_tests {
    use super::*;
    #[test]
    fn scenario_mode_is_explicit_and_preserves_normal_task_parsing() {
        assert!(
            Cli::try_parse_from(["vs1-browser", "--task", "hotel"]).is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "vs1-browser",
                "--scenario",
                "examples/hotel.json",
                "--chooser",
                "lexical"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from(["vs1-browser", "--chooser", "lexical"])
                .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "vs1-browser",
                "--scenario",
                "x",
                "--task",
                "hotel"
            ])
            .is_err()
        );
    }
}
