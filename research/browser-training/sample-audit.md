# Downloaded dataset sample audit

Observed on 2026-09-19 (America/Sao_Paulo). These are direct inspections of pinned
training data, not conclusions inferred solely from dataset descriptions.

## Local files and selection

| Dataset         | Raw local sample                                                        | Selection                                            | Size/coverage                   |
| --------------- | ----------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------- |
| Mind2Web        | `../../artifacts/browser-training/samples/mind2web-sample.jsonl`        | First 2 complete tasks from train shards 0, 5 and 10 | 6 tasks, 49 steps, 3 websites   |
| Typed decisions | `../../artifacts/browser-training/samples/typed-decisions-sample.jsonl` | First/middle/last case in each of 4 workflows        | 12 cases, 60 questions          |
| WebWorldData    | `../../artifacts/browser-training/samples/webworld-sample.jsonl`        | First 4 complete rows at each of 3 byte offsets      | 12 trajectories, 42 transitions |

The complete typed-decisions training Parquet was downloaded because it is only
598,824 bytes; it contains 1,200 cases. No test data was downloaded. The scripts
are in this directory; revisions and hashes are in [download-manifest.json](download-manifest.json).
The local samples total approximately 26 MB. WebWorldData was accessed using HTTP
206 range responses rather than downloading its entire 52 GB source file.

These samples deliberately test multiple shards/workflows/file regions. They are
not random or representative. In particular, Mind2Web's sampled websites are
`exploretock`, `yelp` and `sports.yahoo`, with two tasks per site.

## Mind2Web: concrete adapter work

The source task object has `website`, `domain`, `subdomain`, `annotation_id`,
`confirmed_task`, `action_reprs` and `actions`. Each action carries raw/cleaned
HTML, an operation, and positive/negative candidate arrays. Candidate attributes
are a JSON string, not a nested object.

Measured findings:

- Normalized operation counts: 38 CLICK, 5 TYPE, 6 SELECT.
- Original operations: 36 CLICK, 5 TYPE, 6 SELECT, 2 HOVER. The two hovers must not
  become click training labels for our current CLI.
- 6/49 steps have an empty positive-candidate list. Their action descriptions
  contain useful text, but a description alone does not establish exact identity.
- All 43 listed positive node IDs are present in the respective raw HTML.
- Positive targets include 8 SVG nodes. Mapping an icon to the current runtime's
  observed control requires explicit handling; it is not always a direct role map.
- Negative candidate count is 56–684, median 213. A shortlist/retrieval stage is
  required; we have not yet measured recall for one.
- Snapshot attributes include `input_value`, `option_selected` (on options),
  `aria_label`, bounding boxes and `is_clickable`. Preserve their semantics rather
  than assuming ordinary live-DOM attribute names.
- Remove annotation instrumentation such as `data_pw_testid_buckeye*` and the
  positive/negative grouping from anything the model sees.

Select mapping is a real correctness issue, not cosmetic normalization. None of
the six annotations equals the underlying option value; four equal the visible
label after whitespace normalization. The two time labels require punctuation
normalization. The inspected 5 PM option has visible text `5:00 PM`, source label
`5 00 PM`, and actual value `17:00`. The 10 AM counterpart uses value `10:00`.
A simple alphanumeric-label comparison gives one match for each of the six
sample selects, but this is only an audit heuristic: production must validate
uniqueness and reject collisions instead of treating it as a universal parser.

## Typed decisions: a ready format with imperfect labels

The actual Parquet records contain `id`, `workflow`, `split`, `state`, `questions`,
`gold`, `factors`, `label_agreement` and `n_questions`. Parse the five JSON string
columns explicitly. They are not directly nested JSON as a naive wire adapter
might assume.

The sample has 18 choice, 18 noul and 24 score questions. All 60 probability
supports match their questions; the largest probability-sum rounding error is
approximately 1e-6. Normalize that small error and retain soft distributions.

The downloaded [source card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
identifies synthetic states and teacher-generated soft labels. This is useful
auxiliary supervision, not verified ground truth. `factors` contains latent
scenario information: feeding it to the encoder leaks information unavailable at
inference. `label_agreement` also belongs in audit metadata, not the observation.

The `all` config already combines the four workflows. Concatenating both `all`
and the individual workflow configs would duplicate the training cases.

## WebWorldData: transition records, not ready-made policy labels

The downloaded records all have exactly one top-level field: `conversations`.
Messages use `from` and `value`, with `human`/`gpt` roles. The card's
`messages`/`role`/`content` example is not the observed schema.

The generic instruction requests the next page after a supplied action. Extract
`Initial Page State`, `First Action`, subsequent `Action` turns and their returned
states. Some calls contain nested quotes; do not strip arbitrary quote characters
or execute the expressions. All 42 sampled action expressions parsed as AST
calls in the audit without executing them.

Observed action counts:

| Action           | Count |
| ---------------- | ----- |
| click            | 31    |
| goto             | 5     |
| fill             | 2     |
| noop             | 2     |
| focus            | 1     |
| send_msg_to_user | 1     |

One element action refers to an ID absent from its pre-state. That transition
must not become a target-choice example by guessing. `focus`, navigation and
message actions also exceed or differ from our current runtime semantics.

There are no explicit user-goal, success, origin-subset or quality-score fields
in these 12 records. The human messages describe world simulation, not a task
objective. Therefore none of these records qualifies for direct goal-conditioned
policy imitation as-is. This does not establish that goals are absent from every
record in the full dataset.

Transitions can support carefully defined auxiliary tasks. Their observed next
state may be a label source or an explicit input to a transition-verification
question. It must never leak into a pre-action browser-policy input. No-op,
changed-state, end-of-record and `send_msg_to_user` are not success labels.

## Measured source lengths

Token counts use the pinned English Laya tokenizer. These are untruncated source
texts, without question/options, not final proposed training sequence lengths.

| Source text                               |    Min |  Median |     Max |
| ----------------------------------------- | -----: | ------: | ------: |
| Mind2Web raw HTML, 49 states              | 11,798 |  38,309 | 212,072 |
| Mind2Web cleaned HTML, 49 states          |  3,746 |  13,203 |  81,590 |
| Typed-decision serialized state, 12 cases |     92 |     244 |     301 |
| WebWorld pre-state, 42 transitions        |     14 | 2,219.5 |  18,049 |

24/42 WebWorld pre-states exceed 2,048 tokens, and 3 exceed 8,192. Source-token
counts alone already invalidate a strategy of passing every record unchanged to
Laya's default 512-token configuration. Retrieval and budget-aware observation
rendering are required even for a longer-context training configuration.

## Resulting decisions

1. Use Mind2Web for operation/target labels, with rejection accounting and
   label-blind retrieval; establish conversion coverage before estimating epochs.
2. Use the documented typed-decisions subset as a small general-decision task,
   preserving soft labels and excluding latent factors.
3. Keep the sampled WebWorld transitions out of policy imitation until goals and
   action quality are established. Evaluate auxiliary use separately.
4. Collect additional verified completion and failure examples: none of these
   three inputs automatically supplies the terminal supervision we need.
5. Keep a matched baseline and avoid claiming gains caused merely by more updates,
   more context or different candidate filtering.

The full implementation contract and experiment gates are in the
[training specification](../../docs/browser-decision-training-spec.md).
