"""Thin sequential launcher around starrygl.cli.coupled_ablation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time

ARCHIVE = Path(__file__).resolve().parent
ROOT = Path('/home/zlj/starrygl-undate/.experiment_artifacts/flickr_dcrnn_smoothing_w3_20260912')
PACKAGE = Path('/home/zlj/starrygl-undate/.worktrees/snapshot_smoothing/starrygl-open')
PYTHON = '/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python'
OLD = Path('/tmp/starrygl_open_flickr_ablation_20260911')
GPUS = sys.argv[1] if len(sys.argv) > 1 else '0,1'
ARMS = sys.argv[2:] or ['exact', 'cache', 'smooth']
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS, NCCL_IB_DISABLE='1', OMP_NUM_THREADS='4',
           MKL_NUM_THREADS='4', PYTHONPATH=str(PACKAGE / 'src'))
EXPECTED = json.loads((ARCHIVE / 'verification.json').read_text())['source_hashes']


def archive(name, source):
    destination = ARCHIVE / 'raw' / name
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ('manifest.json', 'plan.txt', 'epochs.jsonl', 'result.json'):
        path = source / filename
        if path.exists():
            shutil.copy2(path, destination / filename)
    subprocess.run([PYTHON, str(ARCHIVE / 'analyze.py')], check=True)


def run(arm, epochs, pilot=False):
    actual = {str(p.relative_to(PACKAGE / 'src/starrygl')): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (PACKAGE / 'src/starrygl').rglob('*.py')}
    assert actual == EXPECTED, 'source changed after verification; refuse mixed-source experiments'
    name = 'pilot_dcrnn_smooth_w3' if pilot else f'dcrnn_{arm}_s42'
    output = ROOT / 'results' / name
    artifact = OLD / 'prepared_dcrnn_2rank' if arm == 'exact' else Path('/mnt/data/zlj/starrygl-experiments/flickr_window_slots_20260911/prepared_dcrnn_hot')
    command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        '-m', 'starrygl.cli.coupled_ablation', '--data', str(OLD / 'data'), '--artifact-root', str(artifact),
        '--output', str(output), '--model', 'dcrnn', '--policy', 'exact' if arm == 'exact' else 'bounded_stale',
        '--num-full-snapshots', '3', '--access-pipeline', '--epochs', str(epochs), '--eval-every', '5']
    if arm == 'smooth':
        command.append('--compensate')
    record = dict(name=name, command=command, started_at_unix=time.time(), gpus=GPUS, status='running')
    with (ROOT / 'logs' / f'{name}.log').open('x') as log:
        process = subprocess.Popen(command, cwd=PACKAGE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = process.pid
        status_path = ROOT / 'logs' / f'{name}_process.json'
        status_path.write_text(json.dumps(record, indent=2)+'\n')
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            if not pilot:
                archive(name, output)
        record.update(returncode=process.returncode, ended_at_unix=time.time(), status='complete' if process.returncode == 0 else 'failed')
        status_path.write_text(json.dumps(record, indent=2)+'\n')
    archive(name, output)
    if process.returncode:
        raise RuntimeError(f'{name} failed; see {ROOT / "logs" / (name + ".log")}')
    result = json.loads((output / 'result.json').read_text())
    if pilot:
        row = json.loads((output / 'epochs.jsonl').read_text().splitlines()[-1])
        assert row['gamma_gradient_abs_sum_rank0'] > 0
        assert row['gamma_effective'] != 0.622459352016449
        for report in (row['state_reads'], row['val_state_reads'], result['test_state_reads']):
            assert report['future_rows'] == report['owner_stale_rows'] == 0
            assert all(slot['future_rows'] == 0 for plane in ('cold_history_inputs', 'hot_shared_predictions') for slot in report[plane])
        (ARCHIVE / 'pilot_check.json').write_text(json.dumps(dict(passed=True, row=row, result=result), indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        run(arm, 10)
    (ARCHIVE / ('completion_' + GPUS.replace(',', '_') + '.json')).write_text(json.dumps(dict(completed_at_unix=time.time(), arms=ARMS, epochs_each=10), indent=2)+'\n')
