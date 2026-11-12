from pathlib import Path
import importlib.util
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'validation' / '.matplotlib'))
import joblib
import nbformat
import numpy as np
import pandas as pd
from PIL import Image

spec = importlib.util.spec_from_file_location('peakguard', ROOT / 'src' / 'peakguard.py')
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)
manifest = json.loads((ROOT / 'outputs' / 'run_manifest.json').read_text(encoding='utf-8'))
selection = json.loads((ROOT / 'outputs' / 'selection_before_test.json').read_text(encoding='utf-8'))
metrics = pd.read_csv(ROOT / 'outputs' / 'test_metrics.csv')
validation = pd.read_csv(ROOT / 'outputs' / 'validation_metrics.csv')
predictions = pd.read_csv(ROOT / 'outputs' / 'test_predictions.csv', parse_dates=['issue_time', 'target_time'])
raw, provenance = pg.load_data('local', str(ROOT / 'data' / 'PJME_hourly.csv'))
cleaned, quality = pg.clean_data(raw)
frame, columns = pg.make_features(cleaned)
parts, threshold, audit = pg.split_data(cleaned, frame, columns)
checks = []


def check(condition, name):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


check(provenance['sha256'] == manifest['provenance']['sha256'], 'Dataset SHA-256 matches manifest')
check(len(raw) == 145366 and list(raw.columns) == ['Datetime', 'PJME_MW'], 'Actual source shape/schema verified')
check(threshold == manifest['peak_threshold_mw'] == selection['peak_threshold_mw'], 'Train P90 equals frozen manifest/selection')
check(audit == manifest['split_audit'], 'Recomputed raw boundaries and class counts match manifest')
check(selection['selected_name'] == validation.iloc[0]['model'] == manifest['selected_on_validation'], 'Winner locked from validation, not final test')
check(len(predictions) == len(parts['test']), 'All models evaluated on the exact same test rows')
check((predictions['target_time'] - predictions['issue_time']).dt.total_seconds().eq(3600).all(), 'Every forecast targets exactly next source hour')
check(np.array_equal(predictions['actual_peak'], parts['test']['is_peak']), 'Exported labels match frozen P90 labels')
for _, row in metrics.iterrows():
    name = row['model']
    actual_metrics = pg.metric_row(name, predictions['actual_peak'], predictions[f'{name}_prediction'], predictions[f'{name}_score'])
    for metric in ('accuracy', 'precision', 'recall', 'f1', 'f2', 'average_precision'):
        check(math.isclose(row[metric], actual_metrics[metric], rel_tol=1e-12, abs_tol=1e-12), f'{name}: recomputed {metric}')
    check(row['decision_cutoff'] == selection['decision_cutoffs'][name], f'{name}: cutoff unchanged after selection')
artifact = joblib.load(ROOT / 'outputs' / 'peakguard_model.joblib')
scores, predicted = pg.predict_candidate(artifact['candidate'], parts['test'], artifact['feature_columns'])
check(np.array_equal(predicted, predictions['selected_prediction']), 'Trusted local model artifact reproduces test predictions')
check(np.allclose(scores, predictions[f'{artifact["selected_name"]}_score'], rtol=1e-12, atol=1e-12), 'Trusted local model artifact reproduces test scores')
alert = pg.simulate_alert(cleaned, artifact)
check(alert['actual_future_known'] is False and alert['simulation_only'] is True, 'Alert simulation does not claim known future')
check(alert == json.loads((ROOT / 'outputs' / 'simulated_alert.json').read_text()), 'Saved simulated alert matches model inference')
notebook = nbformat.read(ROOT / 'PeakGuard_AI_CPE312.ipynb', as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
check(all(cell.execution_count is not None for cell in code_cells), 'Every notebook code cell executed')
check(not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs), 'No notebook cell errors')
image_outputs = sum('image/png' in output.get('data', {}) for cell in code_cells for output in cell.outputs)
check(image_outputs >= 14, 'All fourteen plots embedded in executed notebook')
figures = list((ROOT / 'outputs' / 'figures').glob('*.png'))
check(len(figures) == 14, 'Fourteen PNG figures exported')
for figure in figures:
    with Image.open(figure) as image:
        dpi = image.info.get('dpi', (0, 0))
        check(min(dpi) >= 299, f'{figure.name}: 300-DPI metadata')
summary = (ROOT / 'POSTER_SUMMARY.md').read_text(encoding='utf-8')
check(all(f'R{number}' in summary for number in range(1, 8)), 'Poster includes R1-R7')
check(provenance['sha256'] in summary, 'Poster traces to actual CSV hash')
check('Pending Team Confirmation' in summary and 'validation/INDEPENDENT_REVIEW.md' in summary, 'Poster accurately retains unresolved team/review acceptance')
result = {'status': 'PASS', 'checks_passed': len(checks), 'checks': checks, 'notebook_code_cells': len(code_cells), 'embedded_png_outputs': image_outputs, 'figures': len(figures), 'scope': 'project deliverables only; no parent-directory user files', 'independent_review': 'See validation/INDEPENDENT_REVIEW.md; this script does not establish independent review', 'colab_live': 'NOT RUN'}
(ROOT / 'validation' / 'artifact_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
