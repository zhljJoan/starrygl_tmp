"""Plot archived metrics from the package CLI; no model/runtime implementation."""
from pathlib import Path
import csv
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
LABELS = {'exact': 'Exact', 'cache': 'Cache + cold extrapolation', 'smooth': 'Cache + cold extrapolation + hot fusion'}
COLORS = {'exact': '#2458a6', 'cache': '#d85b26', 'smooth': '#268849'}


def report():
    runs, summary, curves, targets = {}, [], [], []
    for arm in LABELS:
        root = OUT / 'raw' / f'gconv_gru_{arm}_s42'
        if not (root / 'epochs.jsonl').exists():
            continue
        rows = []
        for line in (root / 'epochs.jsonl').read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # live copy can end during a JSONL append
        if not rows:
            continue
        runs[arm] = rows
        result = json.loads((root / 'result.json').read_text()) if (root / 'result.json').exists() else {}
        last = rows[-1]
        summary.append(dict(arm=arm, epochs=len(rows), complete=bool(result), best_val_mse=last['best_val_mse'],
            best_epoch=last['best_epoch'], test_mse=result.get('test_mse_operational'),
            median_epoch_seconds=statistics.median(r['train_seconds'] for r in rows[1:] or rows),
            train_minutes=last['train_seconds_cumulative']/60, wall_minutes=last['run_seconds_cumulative']/60,
            peak_allocated_gib=result.get('peak_allocated_bytes_rankmax', 0)/2**30,
            gamma_effective=last.get('gamma_effective'),
            gamma_gradient_abs_sum=sum(r.get('gamma_gradient_abs_sum_rank0', 0) for r in rows)))
        for threshold in (.1, .05):
            hit = next((row for row in rows if row.get('val_mse', float('inf')) <= threshold), None)
            targets.append(dict(arm=arm, threshold=threshold, epoch=hit['epoch'] if hit else None,
                training_minutes=hit['train_seconds_cumulative']/60 if hit else None,
                wall_minutes=hit['run_seconds_cumulative']/60 if hit else None))
        for row in rows:
            curves.append(dict(arm=arm, epoch=row['epoch'], train_mse=row['train_mse'], val_mse=row.get('val_mse'),
                train_seconds=row['train_seconds'], train_seconds_cumulative=row['train_seconds_cumulative'],
                wall_seconds=row['run_seconds_cumulative'], gamma_effective=row.get('gamma_effective')))
    complete = len(summary) == 3 and all(r['complete'] for r in summary)
    (OUT / 'summary.json').write_text(json.dumps(dict(complete=complete, runs=summary), indent=2)+'\n')
    if not runs:
        (OUT / 'report.md').write_text('# Flickr W=3 GConvGRU\n\n短程检查运行中，正式三组结果尚未产生。配置见 [protocol.md](protocol.md)。\n')
        return
    with (OUT / 'curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    with (OUT / 'time_to_target.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(targets[0])); writer.writeheader(); writer.writerows(targets)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (x, y, title) in zip(axes.flat, (
        ('epoch', 'train_mse', 'Training MSE'), ('epoch', 'val_mse', 'Validation MSE'),
        ('train_seconds_cumulative', 'val_mse', 'Validation vs training time'),
        ('run_seconds_cumulative', 'val_mse', 'Validation vs CLI wall time'))):
        for arm, rows in runs.items():
            points = [r for r in rows if y in r]
            divisor = 1 if x == 'epoch' else 60
            ax.plot([r[x]/divisor for r in points], [r[y] for r in points], marker='o', markersize=3,
                linewidth=1.5, color=COLORS[arm], label=LABELS[arm])
        ax.set(title=title, xlabel='Epoch' if x == 'epoch' else 'Minutes', ylabel='MSE', yscale='log')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Flickr | GConvGRU | W=3 | seed 42' + ('' if complete else ' | IN PROGRESS'))
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'convergence.{suffix}', dpi=170)
    plt.close(fig)
    lines = ['# Flickr W=3 GConvGRU', '', '**状态：'+('三组已完成。' if complete else '运行中，当前曲线不是最终收敛结论。')+'**', '',
        '三组采用相同窗口、节点监督、seed 和优化器。测试列为各组实际返回状态的 MSE。', '',
        '| 组别 | 轮数 | 最优验证 MSE | 选中轮 | 实际测试 MSE | 训练秒/轮中位数（第2轮起） |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        test = '待完成' if s['test_mse'] is None else f"{s['test_mse']:.8f}"
        lines.append(f"| {LABELS[s['arm']]} | {s['epochs']} | {s['best_val_mse']:.8f} | {s['best_epoch']} | {test} | {s['median_epoch_seconds']:.2f} |")
    lines += ['', '![收敛与时间对照](convergence.png)', '',
        '时间含读取审计。旧实验独占 GPU 2/3，新实验独占 0/1，但共享 CPU/存储。单 seed 尚不能说明统计显著性。', '',
        '## 最后训练轮的逐槽位读取', '', '| 组别 | 状态 | 槽位（旧→新） | 平均滞后 | 有非零外推量的行数 |', '|---|---|---:|---:|---:|']
    for arm, rows in runs.items():
        if arm == 'exact':
            continue
        for plane in ('cold_history_inputs', 'hot_shared_predictions'):
            for slot in rows[-1]['state_reads'][plane]:
                lines.append(f"| {arm} | {plane} | {slot['slot']} | {slot['mean_lag']:.4f} | {slot['nonzero_extrapolation_rows']} |")
    lines += ['', '热点预测目标为当前快照输出；冷节点输入目标为前一快照输出，两者滞后不能混算。', '',
        '配置、过滤、公式和计时范围见 [protocol.md](protocol.md)。原始指标见 [raw](raw)，达到固定验证阈值的时间见 [time_to_target.csv](time_to_target.csv)（仅按验证观测点统计）。']
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    report()
