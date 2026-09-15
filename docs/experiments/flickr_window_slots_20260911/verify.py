"""Validate completed experiment artifacts; no training or data mutation."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import zipfile

import torch

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=('dcrnn', 'gconv_gru'))
args = parser.parse_args()
checks = {}
source_hashes = None
with zipfile.ZipFile(OUT / 'source_snapshot.zip') as archive:
    archived_hashes = {name.removeprefix('src/starrygl/'): hashlib.sha256(archive.read(name)).hexdigest()
                       for name in archive.namelist() if name.startswith('src/starrygl/') and name.endswith('.py')}
for model in ((args.model,) if args.model else ('dcrnn', 'gconv_gru')):
    for arm in ('exact', 'cache', 'comp'):
        name = f'{model}_{arm}_s42'
        folder = OUT / 'raw' / name
        manifest = json.loads((folder / 'manifest.json').read_text())
        result = json.loads((folder / 'result.json').read_text())
        rows = [json.loads(s) for s in (folder / 'epochs.jsonl').read_text().splitlines()]
        assert manifest['source_hashes'] == archived_hashes, name
        assert len(rows) == 100 and [r['epoch'] for r in rows] == list(range(1, 101)), name
        val = [r for r in rows if 'val_mse' in r]
        assert len(val) == 21 and all(r['train_steps'] == 27 for r in rows), name
        assert all(r['val_steps'] == 13 for r in val) and result['test_steps'] == 27, name
        assert all(math.isfinite(r['train_mse']) for r in rows), name
        assert all(math.isfinite(r['val_mse']) for r in val), name
        assert math.isfinite(result['test_mse_operational']) and math.isfinite(result['test_mse_exact']), name
        best = min(val, key=lambda r: r['val_mse'])
        assert result['best_epoch'] == best['epoch'] and result['best_val_mse'] == best['val_mse'], name
        audits = [r['state_reads'] for r in rows] + [r['val_state_reads'] for r in val]
        audits += [result['test_state_reads'], result['exact_test_state_reads']]
        assert all(r['future_rows'] == 0 and r['owner_stale_rows'] == 0 for r in audits), name
        left = torch.load(folder / 'best_rank0.pt', map_location='cpu', weights_only=False)
        right = torch.load(folder / 'best_rank1.pt', map_location='cpu', weights_only=False)
        assert left['epoch'] == right['epoch'] == result['best_epoch'], name
        unequal = [k for k in left['model'] if not torch.equal(left['model'][k], right['model'][k])]
        # Increment buffers reflect rank-local observations, not synchronized parameters.
        parameters_unequal = [k for k in unequal if not k.startswith('increment.')]
        assert not parameters_unequal, (name, parameters_unequal)
        checks[name] = dict(epochs=100, validations=len(val), test_steps=result['test_steps'],
            source_matches_archive=True, finite_metrics=True, future_rows=0, owner_stale_rows=0,
            synchronized_model_parameters=True, rank_local_buffers_differ=unequal,
            train_stale_nonzero_increment_rows=sum(r['state_reads']['stale_remote_nonzero_increment_rows'] for r in rows),
            final_gamma=rows[-1].get('gamma'), final_gamma_effective=rows[-1].get('gamma_effective'))
(OUT / ('verification_' + args.model + '.json' if args.model else 'verification.json')).write_text(json.dumps(checks, indent=2)+'\n')
print(json.dumps(checks, indent=2))
