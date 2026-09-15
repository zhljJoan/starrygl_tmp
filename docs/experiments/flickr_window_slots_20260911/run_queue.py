"""Thin experiment launcher: invokes the existing package CLI sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

model, gpus = sys.argv[1:]
root = Path('/mnt/data/zlj/starrygl-experiments/flickr_window_slots_20260911')
package = Path('/home/zlj/starrygl-undate/starrygl-open')
python = '/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python'
old = Path('/tmp/starrygl_open_flickr_ablation_20260911')
env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, NCCL_IB_DISABLE='1',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONPATH=str(package / 'src'))
for arm in ('exact', 'cache', 'comp'):
    name = f'{model}_{arm}_s42'
    artifact = old / f'prepared_{model}_2rank' if arm == 'exact' else root / f'prepared_{model}_hot'
    command = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               '-m', 'starrygl.cli.coupled_ablation', '--data', str(old / 'data'),
               '--artifact-root', str(artifact), '--output', str(root / 'results' / name),
               '--model', model, '--policy', 'exact' if arm == 'exact' else 'bounded_stale',
               '--epochs', '100', '--eval-every', '5']
    if arm == 'comp':
        command.append('--compensate')
    record = dict(name=name, command=command, gpus=gpus, started_at_unix=time.time())
    with (root / 'logs' / f'{name}.log').open('x') as log:
        process = subprocess.Popen(command, cwd=package, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = process.pid
        (root / 'logs' / f'{name}_process.json').write_text(json.dumps(record, indent=2))
        code = process.wait()
    record.update(returncode=code, ended_at_unix=time.time())
    record['wall_seconds'] = record['ended_at_unix'] - record['started_at_unix']
    (root / 'logs' / f'{name}_process.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    if code:
        raise SystemExit(code)
