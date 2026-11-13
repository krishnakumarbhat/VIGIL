# Posthoc accounting: solutions are never exposed to the solver or decoder.
import json
import time
from pathlib import Path

def valid_grid(grid):
    """Enforce ARC's rectangular, nonempty integer color-grid contract."""
    return (isinstance(grid, list) and 1 <= len(grid) <= 30
            and isinstance(grid[0], list) and 1 <= len(grid[0]) <= 30
            and all(isinstance(row, list) and len(row) == len(grid[0]) for row in grid)
            and all(type(value) is int and 0 <= value <= 9 for row in grid for value in row))

def score_selected(predictions, solutions, challenges, selected_ids):
    """Compute exact match per test row and official task-normalized pass@2."""
    rows = []
    task_scores = []
    for task_id in selected_ids:
        task_hits = []
        assert len(predictions[task_id]) == len(challenges[task_id]['test'])
        assert len(solutions[task_id]) == len(challenges[task_id]['test'])
        for index, truth in enumerate(solutions[task_id]):
            attempts = predictions[task_id][index]
            first = attempts['attempt_1'] == truth
            second = attempts['attempt_2'] == truth
            hit = first or second
            task_hits.append(hit)
            rows.append({'task_id': task_id, 'test_index': index,
                         'pass1': first, 'pass2': hit})
        task_scores.append(sum(task_hits) / len(task_hits))
    return {'per_test_rows': rows, 'n_test_rows_selected': len(rows),
            'n_pass1': sum(row['pass1'] for row in rows),
            'n_pass2': sum(row['pass2'] for row in rows),
            'pass1_fraction': sum(row['pass1'] for row in rows) / len(rows),
            'pass2_fraction': sum(row['pass2'] for row in rows) / len(rows),
            'task_normalized_pass2': sum(task_scores) / len(task_scores)}

runtime = json.load(open('/kaggle/working/runtime_mode.json'))
report = {'mode': runtime['mode'], 'duration_s': round(time.time() - runtime['T0'], 1),
          'solver_budget_s': runtime['budget_s'], 'source_version': 1,
          'smoke_benchmark': True, 'not_unbiased_heldout': True,
          'selected_val_ids': VAL_IDS,
          'no_accuracy_claim': 'public model may have trained on evaluation tasks; rank not guaranteed'}
predictions = json.load(open('/kaggle/working/submission.json'))
assert set(predictions) == set(data.queries), 'Submission task IDs do not match inference inputs'
invalid = []
for task_id, entries in predictions.items():
    if len(entries) != len(data.queries[task_id]['test']):
        invalid.append(task_id)
        continue
    for entry in entries:
        if set(entry) != {'attempt_1', 'attempt_2'} or not all(valid_grid(entry[k]) for k in entry):
            invalid.append(task_id)
report['invalid_entries'] = len(invalid)
report['submission_task_count'] = len(predictions)
assert not invalid, f'Invalid submission tasks: {invalid}'
bundles = list(Path('/kaggle/inference_outputs').iterdir())
bundle_keys = {p.name.split('.')[0] for p in bundles}
report['n_prediction_bundles'] = len(bundles)
report['model_test_rows_with_bundles'] = sorted(bundle_keys)
if runtime['mode'] == 'validation':
    solutions = json.load(open('/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json'))
    report.update(score_selected(predictions, solutions, data.queries, VAL_IDS))
    report['tasks_with_all_test_row_bundles'] = [task_id for task_id in VAL_IDS
        if all(f'{task_id}_{i}' in bundle_keys for i in range(len(data.queries[task_id]['test'])))]
    report['n_tasks_with_all_test_row_bundles'] = len(report['tasks_with_all_test_row_bundles'])
    report['tasks_with_nonfallback_candidates'] = [task_id for task_id in VAL_IDS
        if all(decoder.decoded_results.get(f'{task_id}_{i}') for i in range(len(data.queries[task_id]['test'])))]
with open('/kaggle/working/validation_result.json', 'w') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report, indent=2))
