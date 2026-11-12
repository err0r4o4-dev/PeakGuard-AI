from pathlib import Path
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / 'validation' / '.jupyter'
kernel = runtime / 'kernels' / 'peakguard'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'PeakGuard local validation', 'language': 'python'}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(runtime)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime / 'runtime')
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
os.environ['PYTHONUTF8'] = '1'
path = ROOT / 'PeakGuard_AI_CPE312.ipynb'
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=900, kernel_name='peakguard', resources={'metadata': {'path': str(ROOT)}})
started = time.monotonic()
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
errors = [(position, output.get('ename'), output.get('evalue')) for position, cell in enumerate(notebook.cells) for output in cell.get('outputs', []) if output.output_type == 'error']
executed = sum(cell.cell_type == 'code' and cell.execution_count is not None for cell in notebook.cells)
result = {'status': 'PASS' if not errors else 'FAIL', 'code_cells_executed': executed, 'errors': errors, 'elapsed_seconds': round(time.monotonic() - started, 2), 'environment': 'local Python/Jupyter CPU; not Google Colab', 'notebook': path.name}
(ROOT / 'validation' / 'execution_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
if errors:
    raise SystemExit(1)
