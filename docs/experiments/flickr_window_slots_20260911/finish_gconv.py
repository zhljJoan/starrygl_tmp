"""Archive, plot and verify the already-running GConvGRU queue on completion."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
LIVE = Path('/mnt/data/zlj/starrygl-experiments/flickr_window_slots_20260911')
names = [f'gconv_gru_{arm}_s42' for arm in ('exact', 'cache', 'comp')]
while not all((LIVE / 'results' / name / 'result.json').exists() for name in names):
    for name in names:
        record = LIVE / 'logs' / f'{name}_process.json'
        if record.exists() and json.loads(record.read_text()).get('returncode', 0):
            raise RuntimeError(f'{name} failed; inspect {record}')
    if not Path('/proc/2481096/cmdline').exists():
        raise RuntimeError('GConvGRU queue exited without all three result files')
    time.sleep(30)
for src in (LIVE / 'results').iterdir():
    if src.is_dir():
        shutil.copytree(src, OUT / 'raw' / src.name, dirs_exist_ok=True)
for src in (LIVE / 'logs').glob('*_process.json'):
    shutil.copy2(src, OUT / src.name)
subprocess.run([sys.executable, str(OUT / 'analyze.py')], check=True)
subprocess.run([sys.executable, str(OUT / 'verify.py'), '--model', 'gconv_gru'], check=True)
(OUT / 'gconv_completion.json').write_text(json.dumps({
    'completed_unix': time.time(), 'runs': names, 'plots_updated': True,
    'verification': 'verification_gconv_gru.json', 'dcrnn_cache_comp': 'deferred',
}, indent=2) + '\n')
print('GConvGRU completed, archived, plotted and verified.', flush=True)
