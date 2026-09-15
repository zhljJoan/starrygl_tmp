"""Compare observed first-ten-epoch MSE, without mixing final 100-epoch checkpoints."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOTS = {
    'DCRNN': Path('/home/zlj/starrygl-undate/.experiment_artifacts/flickr_dcrnn_smoothing_w3_20260912/results'),
    'GConvGRU': Path('/mnt/data/zlj/starrygl-experiments/flickr_smoothing_w3_20260911/results'),
}
PREFIXES = {'DCRNN': 'dcrnn', 'GConvGRU': 'gconv_gru'}
ARMS = {'exact': 'Exact', 'cache': 'Cache + cold extrapolation', 'smooth': 'Cache + cold extrapolation + hot fusion'}
COLORS = {'DCRNN': '#2458a6', 'GConvGRU': '#d85b26'}


def main():
    runs, manifests, records, summaries = {}, [], [], []
    for model, root in ROOTS.items():
        for arm in ARMS:
            path = root / f'{PREFIXES[model]}_{arm}_s42'
            if not (path / 'manifest.json').exists():
                continue
            manifests.append(json.loads((path / 'manifest.json').read_text()))
            rows = []
            if (path / 'epochs.jsonl').exists():
                for line in (path / 'epochs.jsonl').read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if row['epoch'] <= 10:
                        rows.append(row)
            runs[model, arm] = rows
            for row in rows:
                records.append(dict(model=model, arm=arm, epoch=row['epoch'], train_mse=row['train_mse'],
                                    val_mse=row.get('val_mse')))
            last = rows[-1] if rows else {}
            summaries.append(dict(model=model, arm=arm, epochs=len(rows), epoch10_train_mse=last.get('train_mse') if len(rows)==10 else None,
                epoch10_val_mse=last.get('val_mse') if len(rows)==10 else None))
    for manifest in manifests:
        assert manifest['source_hashes'] == manifests[0]['source_hashes']
        for key in ('data', 'num_full_snapshots', 'access_pipeline', 'seed', 'hidden_dim', 'lr', 'eval_every', 'hot_ratio', 'max_staleness'):
            assert manifest['arguments'][key] == manifests[0]['arguments'][key], key
    complete = len(runs) == 6 and all(len(rows)==10 and 'val_mse' in rows[-1] for rows in runs.values())
    (OUT / 'summary.json').write_text(json.dumps(dict(complete=complete, runs=summaries), indent=2)+'\n')
    with (OUT / 'mse_curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model','arm','epoch','train_mse','val_mse'])
        writer.writeheader(); writer.writerows(records)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey='row', constrained_layout=True)
    for col, (arm, title) in enumerate(ARMS.items()):
        for row_id, metric in enumerate(('train_mse', 'val_mse')):
            ax = axes[row_id, col]
            for model in ROOTS:
                points = [row for row in runs.get((model, arm), []) if metric in row]
                if points:
                    ax.plot([p['epoch'] for p in points], [p[metric] for p in points],
                        color=COLORS[model], label=model, marker='o', markersize=4,
                        linestyle='-' if model=='DCRNN' else '--')
            ax.set(title=title if row_id==0 else '', xlabel='Epoch', ylabel='Training MSE' if row_id==0 else 'Validation MSE',
                   xlim=(.7,10.3), xticks=[1,2,3,4,5,6,7,8,9,10], yscale='log')
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8)
    fig.suptitle('Flickr node regression | W=3 | 2 ranks/run | seed 42 | first 10 epochs' + ('' if complete else ' | IN PROGRESS'))
    for suffix in ('png','pdf'):
        fig.savefig(OUT / f'mse_comparison.{suffix}', dpi=180)
    plt.close(fig)
    lines = ['# DCRNN 与 GConvGRU：前 10 轮 MSE 对照', '',
        '**状态：'+('六组前 10 轮训练与验证已齐备。' if complete else 'DCRNN 短跑进行中，图中缺失部分尚未完成。')+'**', '',
        'Flickr 节点回归、W=3 滑动快照、每组双卡、seed=42、hidden=8、Adam lr=0.001、access_pipeline 开启。'
        '三种策略分别比较两种模型。训练曲线逐轮记录，验证仅在第 1、5、10 轮测量；连线不表示中间轮有验证观测。', '',
        'GConvGRU 使用已完成同配置长跑的前 10 轮；优化器无按总轮数变化的学习率计划。'
        'DCRNN 按新要求短跑 10 轮。没有混入 GConvGRU 的 100 轮最终测试指标。源码 hashes 和公共配置已校验一致。', '',
        '![MSE 对比](mse_comparison.png)', '',
        '| 模型 | 策略 | 已记录轮数 | 第10轮训练 MSE | 第10轮验证 MSE |', '|---|---|---:|---:|---:|']
    for entry in summaries:
        train = '待完成' if entry['epoch10_train_mse'] is None else f"{entry['epoch10_train_mse']:.8f}"
        val = '待完成' if entry['epoch10_val_mse'] is None else f"{entry['epoch10_val_mse']:.8f}"
        lines.append(f"| {entry['model']} | {entry['arm']} | {entry['epochs']} | {train} | {val} |")
    lines += ['', '这是单 seed 的早期学习曲线，不代表最终收敛精度。DCRNN 各组并行使用不同 GPU 对，本图不据此给出跨模型严格耗时结论。', '',
              '数据：[mse_curves.csv](mse_curves.csv)，矢量图：[mse_comparison.pdf](mse_comparison.pdf)。']
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
