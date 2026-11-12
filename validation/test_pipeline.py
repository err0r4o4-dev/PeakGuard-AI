import importlib.util
import json
from pathlib import Path
import sys
import unittest
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'src' / 'peakguard.py'


def load_module():
    spec = importlib.util.spec_from_file_location('peakguard', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineContract(unittest.TestCase):
    def test_implementation_exists(self):
        self.assertTrue(MODULE.exists(), 'Causal pipeline has not been implemented')

    def setUp(self):
        if not MODULE.exists() and self._testMethodName != 'test_implementation_exists':
            self.skipTest('Implementation pending')
        if MODULE.exists():
            self.pg = load_module()
            self.index = pd.date_range('2020-01-01', periods=1600, freq='h')
            self.raw = pd.DataFrame({'Datetime': self.index, 'PJME_MW': np.arange(1600) + 100.0})

    def test_schema_rejects_missing_demand(self):
        with self.assertRaisesRegex(ValueError, 'PJME_MW'):
            self.pg.clean_data(self.raw.drop(columns='PJME_MW'))

    def test_conflicting_duplicate_is_unknown_not_last_value(self):
        duplicate = self.raw.iloc[[500]].copy()
        duplicate['PJME_MW'] = 99999
        clean, report = self.pg.clean_data(pd.concat([self.raw, duplicate]))
        self.assertTrue(pd.isna(clean.loc[self.index[500], 'demand_mw']))
        self.assertEqual(report['conflicting_duplicate_hours'], 1)

    def test_identical_duplicate_collapses(self):
        clean, report = self.pg.clean_data(pd.concat([self.raw, self.raw.iloc[[500]]]))
        self.assertEqual(len(clean), len(self.raw))
        self.assertEqual(clean.loc[self.index[500], 'demand_mw'], 600)

    def test_invalid_demand_and_gap_stay_missing(self):
        raw = self.raw.drop(index=500).copy()
        raw.loc[501, 'PJME_MW'] = -10
        clean, report = self.pg.clean_data(raw)
        self.assertTrue(clean.loc[self.index[500:502], 'demand_mw'].isna().all())
        self.assertEqual(report['missing_grid_hours'], 1)

    def test_target_does_not_bridge_missing_hour(self):
        clean, _ = self.pg.clean_data(self.raw.drop(index=500))
        frame, columns = self.pg.make_features(clean)
        self.assertTrue(pd.isna(frame.loc[self.index[499], 'target_mw']))
        self.assertEqual(frame.loc[self.index[498], 'target_mw'], 599)
        self.assertNotIn('target_mw', columns)
        self.assertNotIn('target_time', columns)

    def test_features_are_causal_under_future_perturbation(self):
        clean, _ = self.pg.clean_data(self.raw)
        changed = clean.copy()
        changed.loc[self.index[801]:, 'demand_mw'] *= 100
        original, columns = self.pg.make_features(clean)
        altered, _ = self.pg.make_features(changed)
        pd.testing.assert_frame_equal(original.loc[:self.index[800], columns], altered.loc[:self.index[800], columns])
        self.assertEqual(original.loc[self.index[800], 'lag_24h_mw'], clean.loc[self.index[776], 'demand_mw'])
        self.assertEqual(original.loc[self.index[800], 'rolling_mean_24h_mw'], clean.loc[self.index[777:801], 'demand_mw'].mean())

    def test_split_purges_boundary_labels_and_freezes_p90(self):
        clean, _ = self.pg.clean_data(self.raw)
        frame, columns = self.pg.make_features(clean)
        parts, threshold, audit = self.pg.split_data(clean, frame, columns)
        self.assertAlmostEqual(threshold, clean.iloc[:1120]['demand_mw'].quantile(.9))
        self.assertLess(parts['train']['target_time'].max(), parts['validation'].index.min())
        self.assertLess(parts['validation']['target_time'].max(), parts['test'].index.min())
        changed = clean.copy()
        changed.iloc[1120:, 0] *= 100
        new_frame, _ = self.pg.make_features(changed)
        new_parts, new_threshold, _ = self.pg.split_data(changed, new_frame, columns)
        self.assertEqual(threshold, new_threshold)
        pd.testing.assert_frame_equal(parts['train'], new_parts['train'])

    def test_day_ago_baseline_uses_target_hour_yesterday(self):
        clean, _ = self.pg.clean_data(self.raw)
        frame, _ = self.pg.make_features(clean)
        self.assertEqual(frame.loc[self.index[800], 'day_ago_target_mw'], clean.loc[self.index[777], 'demand_mw'])

    def test_metrics_handle_one_class_without_inventing_ap(self):
        metrics = self.pg.metric_row('normal', [0, 0], [0, 0], [.1, .2])
        self.assertEqual(metrics['accuracy'], 1)
        self.assertIsNone(metrics['average_precision'])

    def test_threshold_search_is_deterministic(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([.1, .2, .7, .8])
        chosen = self.pg.choose_cutoff(labels, scores)
        self.assertEqual(chosen, self.pg.choose_cutoff(labels, scores))
        self.assertEqual(self.pg.metric_row('chosen', labels, scores >= chosen, scores)['f2'], 1)

    def test_causal_time_operations_emit_no_deprecation_warnings(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            clean, _ = self.pg.clean_data(self.raw)
            frame, columns = self.pg.make_features(clean)
            self.pg.split_data(clean, frame, columns)

    def test_training_rejects_single_class_validation(self):
        clean, _ = self.pg.clean_data(self.raw)
        frame, columns = self.pg.make_features(clean)
        parts, threshold, _ = self.pg.split_data(clean, frame, columns)
        parts['validation']['is_peak'] = 0
        with self.assertRaisesRegex(ValueError, 'validation has one class'):
            self.pg.train_candidates(parts, columns, threshold)

    def test_kaggle_download_replaces_stale_csv_from_current_archive(self):
        import tempfile
        import zipfile
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "PJME_hourly.csv"
            destination.write_text("Datetime,PJME_MW\n2020-01-01 00:00:00,1\n", encoding="utf-8")
            (Path(folder) / "kaggle.json").write_text("{}", encoding="utf-8")
            def download(command, **kwargs):
                with zipfile.ZipFile(Path(command[-1]) / "PJME_hourly.csv.zip", "w") as archive:
                    archive.writestr("PJME_hourly.csv", "Datetime,PJME_MW\n2020-01-01 00:00:00,2\n")
                return SimpleNamespace(returncode=0)
            with patch("shutil.which", return_value="kaggle"), patch.dict(os.environ, {"KAGGLE_CONFIG_DIR": folder}), patch.object(self.pg.subprocess, "run", side_effect=download):
                raw, provenance = self.pg.load_data("kaggle", str(destination))
            self.assertEqual(raw["PJME_MW"].tolist(), [2])
            self.assertEqual(provenance["raw_rows"], 1)

    def test_local_loader_reports_missing_file_and_hashes_real_file(self):
        import hashlib
        with self.assertRaises(FileNotFoundError):
            self.pg.load_data('local', str(ROOT / 'missing-data.csv'))
        path = ROOT / 'data' / 'PJME_hourly.csv'
        if path.exists():
            raw, evidence = self.pg.load_data('local', str(path))
            self.assertEqual(evidence['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(evidence['raw_rows'], len(raw))

    def test_invalid_timestamp_is_counted_and_off_hour_rejected(self):
        raw = self.raw.copy()
        raw["Datetime"] = raw["Datetime"].astype(object)
        raw.loc[500, 'Datetime'] = 'not-a-date'
        clean, report = self.pg.clean_data(raw)
        self.assertEqual(report['invalid_datetime_rows'], 1)
        self.assertTrue(pd.isna(clean.loc[self.index[500], 'demand_mw']))
        raw = self.raw.copy()
        raw.loc[500, 'Datetime'] = self.index[500] + np.timedelta64(15, "m")
        with self.assertRaisesRegex(ValueError, 'Non-hour-aligned'):
            self.pg.clean_data(raw)

    def test_notebook_embedded_source_matches_module(self):
        import nbformat
        path = ROOT / 'PeakGuard_AI_CPE312.ipynb'
        if not path.exists():
            self.skipTest('Notebook generation pending')
        notebook = nbformat.read(path, as_version=4)
        source = MODULE.read_text(encoding='utf-8')
        for block in source.split('\n\ndef '):
            expected = block.strip() if block == source.split('\n\ndef ')[0] else ('def ' + block).strip()
            self.assertTrue(any(cell.cell_type == 'code' and cell.source.strip() == expected for cell in notebook.cells))

    def test_artifact_verifier_checks_project_without_parent_notebook(self):
        import subprocess
        result = subprocess.run([sys.executable, str(ROOT / "validation" / "verify_artifacts.py")], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["scope"], "project deliverables only; no parent-directory user files")

    def test_notebook_validates_and_compiles(self):
        path = ROOT / 'PeakGuard_AI_CPE312.ipynb'
        if not path.exists():
            self.skipTest('Notebook generation pending')
        import nbformat
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for position, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'cell-{position}', 'exec')
        text = '\n'.join(cell.source for cell in notebook.cells)
        for section in range(1, 8):
            self.assertIn(f'R{section}', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
