"""Generate requirement-level questions from captured observations, not gold labels."""
import argparse
import json
from pathlib import Path

HOTEL = [
    'the destination search has been applied for Lisbon',
    'the current category filter is Design',
    'the Free cancellation filter is enabled',
    'the currently open detail page is Casa Flora, not a search result listing',
]
FORM = [
    'the Name field contains Ada or the confirmation says Submitted Name: Ada',
    'Terms are currently checked or the confirmation says Terms accepted',
    'the form has been submitted',
    'the current page is Confirmation',
]
# Evaluation-only truth table for the controlled setups in prompt_probe.rs.
GOLD = {
    'initial': [0, 0, 0, 0], 'typed': [0, 0, 0, 0],
    'searched': [1, 0, 0, 0], 'design': [1, 1, 0, 0],
    'filtered': [1, 1, 1, 0], 'complete': [1, 1, 1, 1],
    'wrong_filters': [0, 0, 0, 1], 'wrong_property': [1, 1, 1, 0],
    'form_empty': [0, 0, 0, 0], 'form_named': [1, 0, 0, 0],
    'form_ready': [1, 1, 0, 0], 'form_complete': [1, 1, 1, 1],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((args.capture / name).read_text())
    captures, requests, manifest = load('captures.json'), load('requests.json'), load('manifest.json')
    generated, metadata = [], []
    for capture in captures:
        name = capture['case']
        texts = FORM if name.startswith('form') else HOTEL
        for variant in ['compact', 'natural_controls']:
            original = next(r for m, r in zip(manifest, requests)
                            if m['case'] == name and m['variant'] == 'compact')
            state = original['state']
            if variant == 'natural_controls':
                controls, seen = [], set()
                for action in capture['page']['actions']:
                    if action.get('node') in seen:
                        continue
                    seen.add(action.get('node'))
                    if action['kind'] == 'fill':
                        controls.append(f"{action['label']}: text is {json.dumps(action['value'])}.")
                    elif action['kind'] == 'select':
                        controls.append(f"{action['label'].split(' → ')[0]}: currently selected {action['current_value']}.")
                    elif 'checked' in action:
                        checked = 'checked' if action['checked'] == 'true' else 'unchecked'
                        controls.append(f"{action['label']}: {checked}.")
                state = (f"Page title: {capture['page']['title']}\nCurrent form values:\n"
                         + '\n'.join(controls) + f"\nVisible page text:\n{capture['page']['text']}")
            questions = {f'fact{i}': {'type': 'noul', 'instructions':
                f'Is it true that {text}? Use current observed values, not available options.'}
                for i, text in enumerate(texts)}
            questions['complete'] = {'type': 'noul', 'instructions':
                f"Has the entire task been completed? {capture['goal']}"}
            generated.append({'state': state, 'questions': questions})
            metadata.append({'case': name, 'variant': variant, 'gold': GOLD[name]})
    for filename, value in [('requests.json', generated), ('manifest.json', metadata)]:
        (args.output / filename).write_text(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
