import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import nbformat

from test_pipeline import ROOT, load_module


class LoaderPortability(unittest.TestCase):
    def setUp(self):
        self.pg = load_module()
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'validation')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / 'PJME_hourly.csv'
        self.path.write_text('Datetime,PJME_MW\n2020-01-01 00:00:00,12345\n', encoding='utf-8')

    def colab_modules(self, **services):
        google = ModuleType('google')
        google.__path__ = []
        colab = ModuleType('google.colab')
        for name, service in services.items():
            setattr(colab, name, service)
        google.colab = colab
        return patch.dict(sys.modules, {'google': google, 'google.colab': colab})

    def kaggle_environment(self):
        (self.directory / 'kaggle.json').write_text('{}', encoding='utf-8')
        return patch.dict(os.environ, {'KAGGLE_CONFIG_DIR': str(self.directory), 'USERPROFILE': str(self.directory), 'HOME': str(self.directory)}, clear=True)

    def test_upload_reads_the_selected_csv(self):
        files = SimpleNamespace(upload=Mock(return_value={str(self.path): self.path.read_bytes()}))
        with self.colab_modules(files=files):
            raw, provenance = self.pg.load_data('upload', str(self.directory / 'not-selected.csv'))
        self.assertEqual(raw['PJME_MW'].tolist(), [12345])
        self.assertEqual(provenance['file'], 'PJME_hourly.csv')

    def test_upload_rejects_absent_or_ambiguous_dataset(self):
        for uploaded in ({}, {'wrong.csv': b''}, {'first/PJME_hourly.csv': b'', 'second/PJME_hourly.csv': b''}):
            with self.subTest(names=list(uploaded)), self.colab_modules(files=SimpleNamespace(upload=Mock(return_value=uploaded))):
                with self.assertRaisesRegex(ValueError, 'exactly one file'):
                    self.pg.load_data('upload')

    def test_upload_outside_colab_has_actionable_error(self):
        with patch.dict(sys.modules, {'google.colab': None}):
            with self.assertRaisesRegex(RuntimeError, 'Outside Colab use local mode'):
                self.pg.load_data('upload')

    def test_drive_reads_configured_path_after_mount(self):
        drive = SimpleNamespace(mount=Mock())
        with self.colab_modules(drive=drive):
            raw, provenance = self.pg.load_data('drive', drive_path=str(self.path))
        self.assertEqual(raw['PJME_MW'].tolist(), [12345])
        self.assertEqual(provenance['raw_rows'], 1)
        drive.mount.assert_called_once_with('/content/drive')

    def test_drive_requires_explicit_path(self):
        with self.assertRaisesRegex(ValueError, 'Set DRIVE_PATH'):
            self.pg.load_data('drive')

    def test_drive_outside_colab_has_actionable_error(self):
        with patch.dict(sys.modules, {'google.colab': None}):
            with self.assertRaisesRegex(RuntimeError, 'Use local mode elsewhere'):
                self.pg.load_data('drive', drive_path=str(self.path))

    def test_kaggle_reads_current_direct_csv_not_previous_destination(self):
        download_locations = []
        def download(command, **kwargs):
            self.assertEqual(command[:7], ['kaggle', 'datasets', 'download', '-d', 'robikscube/hourly-energy-consumption', '-f', 'PJME_hourly.csv'])
            self.assertEqual(command[-2], '-p')
            location = Path(command[-1])
            download_locations.append(location)
            (location / 'PJME_hourly.csv').write_text('Datetime,PJME_MW\n2020-01-01 00:00:00,23456\n', encoding='utf-8')
            return SimpleNamespace(returncode=0)
        with self.kaggle_environment(), patch('shutil.which', return_value='kaggle'), patch.object(self.pg.subprocess, 'run', side_effect=download):
            raw, _ = self.pg.load_data('kaggle', str(self.path))
        self.assertEqual(raw['PJME_MW'].tolist(), [23456])
        self.assertFalse(download_locations[0].exists())

    def test_kaggle_failure_preserves_csv_and_suppresses_command_output(self):
        before = self.path.read_bytes()
        response = SimpleNamespace(returncode=1, stdout='private-output-marker', stderr='private-error-marker')
        with self.kaggle_environment(), patch('shutil.which', return_value='kaggle'), patch.object(self.pg.subprocess, 'run', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'Kaggle download failed') as caught:
                self.pg.load_data('kaggle', str(self.path))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertNotIn('private-output-marker', str(caught.exception))
        self.assertNotIn('private-error-marker', str(caught.exception))

    def test_kaggle_missing_download_output_does_not_reuse_existing_csv(self):
        before = self.path.read_bytes()
        with self.kaggle_environment(), patch('shutil.which', return_value='kaggle'), patch.object(self.pg.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(FileNotFoundError, 'did not produce'):
                self.pg.load_data('kaggle', str(self.path))
        self.assertEqual(self.path.read_bytes(), before)

    def test_kaggle_requires_cli(self):
        with patch('shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Install the optional kaggle package'):
                self.pg.load_data('kaggle', str(self.path))

    def test_kaggle_requires_private_configuration(self):
        with patch.dict(os.environ, {'KAGGLE_CONFIG_DIR': str(self.directory), 'USERPROFILE': str(self.directory), 'HOME': str(self.directory)}, clear=True), patch('shutil.which', return_value='kaggle'):
            with self.assertRaisesRegex(RuntimeError, 'Configure Kaggle credentials privately'):
                self.pg.load_data('kaggle', str(self.path))

    def test_kaggle_archive_cannot_extract_unexpected_members(self):
        before = self.path.read_bytes()
        def download(command, **kwargs):
            with zipfile.ZipFile(Path(command[-1]) / 'PJME_hourly.csv.zip', 'w') as archive:
                archive.writestr('../unexpected.csv', 'unexpected')
            return SimpleNamespace(returncode=0)
        with self.kaggle_environment(), patch('shutil.which', return_value='kaggle'), patch.object(self.pg.subprocess, 'run', side_effect=download):
            with self.assertRaises(KeyError):
                self.pg.load_data('kaggle', str(self.path))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse((self.directory / 'unexpected.csv').exists())

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'MODE must be'):
            self.pg.load_data('unsupported', str(self.path))


class NotebookRegeneration(unittest.TestCase):
    def test_regeneration_preserves_user_section_titles(self):
        original = ROOT / 'PeakGuard_AI_CPE312.ipynb'
        before = original.read_bytes()
        current = nbformat.read(original, as_version=4)
        with tempfile.TemporaryDirectory(dir=ROOT / 'validation') as folder:
            project = Path(folder)
            (project / 'validation').mkdir()
            (project / 'src').mkdir()
            script = project / 'validation' / 'build_notebook.py'
            shutil.copyfile(ROOT / 'validation' / 'build_notebook.py', script)
            shutil.copyfile(ROOT / 'src' / 'peakguard.py', project / 'src' / 'peakguard.py')
            result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            generated = nbformat.read(project / 'PeakGuard_AI_CPE312.ipynb', as_version=4)
        current_titles = [cell.source.splitlines()[0] for cell in current.cells if cell.cell_type == 'markdown' and cell.source.startswith('## R')]
        generated_titles = [cell.source.splitlines()[0] for cell in generated.cells if cell.cell_type == 'markdown' and cell.source.startswith('## R')]
        self.assertEqual(generated_titles, current_titles)
        self.assertEqual(len(generated_titles), 7)
        self.assertEqual(original.read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
