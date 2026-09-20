"""Summarize fixed-state prompt probes; raw observations remain ignored artifacts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument('--responses', default='responses.json')
    parser.add_argument('--output', default='analysis.json')
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    responses = json.loads((args.directory / args.responses).read_text())
    assert len(manifest) == len(responses)
    summary = {}
    rows = []
    for case, result in zip(manifest, responses):
        answers = result['response']['answers']
        decision = answers['operation']['choice']
        done = decision == 'DONE'
        row = {**case, 'decision': decision,
               'done_probability': answers['operation']['probabilities'].get('DONE', 0),
               'noul_probability': answers['complete']['noul'],
               'completion_choice': answers['completion_choice']['choice'],
               'operation_tokens': len(result['ids']['operation']['ids'])}
        rows.append(row)
        metrics = summary.setdefault(case['variant'], dict(unfinished=0, completed=0,
            premature_done=0, correct_done=0, noul_correct=0, completion_choice_correct=0,
            sequences_at_limit=0))
        metrics['completed' if case['complete'] else 'unfinished'] += 1
        metrics['premature_done'] += int(done and not case['complete'])
        metrics['correct_done'] += int(done and case['complete'])
        metrics['noul_correct'] += int((answers['complete']['noul'] >= .5) == case['complete'])
        metrics['completion_choice_correct'] += int((answers['completion_choice']['choice'] == 'complete') == case['complete'])
        metrics['sequences_at_limit'] += int(row['operation_tokens'] == args.max_len)
    output = {'scope': 'Controlled states, not rollouts. No target accuracy inferred from operation selection.',
              'summary': summary, 'rows': rows}
    (args.directory / args.output).write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
