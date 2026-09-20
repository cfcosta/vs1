"""Audit downloaded samples without executing pages or dataset actions.

Requires tokenizers, pyarrow and lxml. Produces measurements, not training data.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

from lxml import html
from tokenizers import Tokenizer


def stats(values):
    values = sorted(values)
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "p95_nearest_rank": values[max(0, (95 * len(values) + 99) // 100 - 1)], "max": values[-1]}


def load(root, name):
    return [json.loads(line)["record"] for line in (root / f"{name}-sample.jsonl").open()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("artifacts/browser-training/samples"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(args.tokenizer))
    tok.no_truncation()
    tok.no_padding()
    ntok = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    report = {"scope": "Small convenience samples; no population-quality estimate.",
              "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
              "token_counts": "Source text only, without question/options; no truncation."}

    tasks = load(args.samples, "mind2web")
    steps = [a for task in tasks for a in task["actions"]]
    attrs, tags, pos_missing, raw_missing = Counter(), Counter(), 0, 0
    multi_positive = 0
    selection_matches = []
    for a in steps:
        tree = html.fromstring(a["raw_html"])
        raw_nodes = {n.get("backend_node_id"): n for n in tree.iter() if n.get("backend_node_id")}
        pos_missing += not a["pos_candidates"]
        multi_positive += len(a["pos_candidates"]) > 1
        for pos in a["pos_candidates"]:
            tags[pos['tag']] += 1
            attrs.update(json.loads(pos["attributes"]).keys())
            raw_missing += pos['backend_node_id'] not in raw_nodes
        if a["operation"]["op"] == "SELECT":
            value = a["operation"]["value"]
            options = [o for p in a["pos_candidates"] for o in raw_nodes[p['backend_node_id']].iter('option')]
            key = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
            normalized_matches = [o for o in options if key(o.text_content()) == key(value)]
            selection_matches.append({"annotation_value": value, "option_count": len(options),
                                      "matches_value": any(o.get('value') == value for o in options),
                                      "matches_text": any(' '.join(o.text_content().split()) == value for o in options),
                                      "alphanumeric_label_match_count": len(normalized_matches),
                                      "resolved_value_if_unique": normalized_matches[0].get('value') if len(normalized_matches) == 1 else None})
    report["mind2web"] = {
        "tasks": len(tasks), "websites": sorted({t['website'] for t in tasks}), "steps": len(steps),
        "task_keys": list(tasks[0]), "step_keys": list(steps[0]),
        "operations": dict(Counter(a['operation']['op'] for a in steps)),
        "original_operations": dict(Counter(a['operation']['original_op'] for a in steps)),
        "steps_without_positive_candidates": pos_missing, "steps_with_multiple_positive_candidates": multi_positive,
        "positive_candidates_missing_in_raw_html": raw_missing,
        "positive_tags": dict(tags), "positive_attribute_names": dict(attrs),
        "positive_counts": stats([len(a['pos_candidates']) for a in steps]),
        "negative_counts": stats([len(a['neg_candidates']) for a in steps]),
        "raw_html_tokens": stats([ntok(a['raw_html']) for a in steps]),
        "cleaned_html_tokens": stats([ntok(a['cleaned_html']) for a in steps]),
        "select_mapping_checks": selection_matches,
    }

    rows = load(args.samples, "typed-decisions")
    qtypes, totals, errors, states = Counter(), [], [], []
    for r in rows:
        state, questions, gold = (json.loads(r[k]) for k in ['state', 'questions', 'gold'])
        states.append(ntok(json.dumps(state, ensure_ascii=False)))
        assert set(questions) == set(gold)
        for key, q in questions.items():
            qtypes[q['type']] += 1
            g = gold[key]
            keys = list(q['criteria']) if q['type'] == 'choice' else ([str(i) for i in range(len(q['criteria']))] if q['type'] == 'score' else ['false', 'true'])
            assert set(keys) == set(g['probabilities'])
            p = [g['probabilities'][k] for k in keys]
            assert all(0 <= v <= 1 for v in p)
            totals.append(len(keys)); errors.append(abs(sum(p)-1))
    report['typed_decisions'] = {
        'sample_cases':len(rows), 'questions':sum(qtypes.values()), 'question_types':dict(qtypes),
        'workflows': dict(Counter(r['workflow'] for r in rows)), 'record_keys':list(rows[0]),
        'json_encoded_columns':[k for k in ['state','questions','gold','factors','label_agreement'] if isinstance(rows[0][k],str)],
        'state_tokens':stats(states), 'options_per_question':stats(totals),
        'maximum_probability_sum_error':max(errors),
    }

    trajectories = load(args.samples, 'webworld')
    operations, roles = Counter(), Counter()
    lengths, turn_counts, transitions = [], [], []
    missing_targets, parse_errors, same_states = 0, [], 0
    for row_index, r in enumerate(trajectories):
        conv = r['conversations'];turn_counts.append(len(conv))
        current = None
        for index in range(0,len(conv),2):
            user, response = conv[index:index+2]
            roles.update([user['from'],response['from']])
            assert user['from']=='human' and response['from']=='gpt'
            match = re.search(r"\n\n(?:First Action|Action): '(.*)'\n\nNext Page State:\s*$",user['value'], re.S)
            assert match is not None, (row_index,index)
            if index==0:
                current=user['value'].split('Initial Page State:\n',1)[1][:].split('\n\nFirst Action:',1)[0]
            command=match.group(1)
            try:
                node=ast.parse(command,mode='eval').body
                assert isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                operation=node.func.id
            except (SyntaxError, AssertionError) as e:
                parse_errors.append({'row':row_index,'turn':index,'error':str(e)})
                operation='UNPARSED'
                node=None
            operations[operation]+=1
            n=ntok(current);lengths.append(n)
            target_present=None
            if operation in ['click','fill','select_option','focus','hover']:
                first=node.args[0]
                assert isinstance(first,ast.Constant) and isinstance(first.value,(str,int))
                target_present=bool(re.search(r'^\s*\['+re.escape(str(first.value))+r'\]',current,re.M))
                missing_targets+=not target_present
            same=current==response['value'];same_states+=same
            transitions.append({'sample_row':row_index,'turn':index//2,'operation':operation,
                                'pre_state_tokens':n,'target_present':target_present,'identical_next_state':same})
            current=response['value']
    report['webworld']={
        'trajectories':len(trajectories), 'record_keys':list(trajectories[0]),
        'conversation_roles':dict(roles),'conversation_message_counts':stats(turn_counts),
        'transitions':len(transitions),'operations':dict(operations),'parse_errors':parse_errors,
        'element_actions_with_missing_target':missing_targets,'identical_state_transitions':same_states,
        'pre_state_tokens':stats(lengths),'pre_states_over_2048_tokens':sum(n>2048 for n in lengths),
        'pre_states_over_8192_tokens':sum(n>8192 for n in lengths),
        'explicit_goal_or_success_metadata':False,
        'metadata_observation':'Only conversations keys; user turns request state prediction, not a goal-directed policy.',
    }
    (args.samples/'webworld-transition-audit.json').write_text(json.dumps(transitions,indent=2)+'\n')
    (args.samples/'inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
