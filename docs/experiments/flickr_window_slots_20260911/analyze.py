"""Regenerate this experiment's tables/plots from archived JSONL and result JSON."""
from pathlib import Path
import csv
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
COLORS = {'exact': '#2458a6', 'cache': '#d85b26', 'comp': '#268849'}
LABELS = {'exact': 'exact', 'cache': 'window cache', 'comp': 'window cache + increment'}


def write_csv(name, rows):
    if rows:
        with (OUT / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def report():
    runs = []
    for p in sorted((OUT / 'raw').glob('*/manifest.json')):
        if not (p.parent / 'epochs.jsonl').exists():
            continue
        manifest = json.loads(p.read_text())
        args = manifest['arguments']
        arm = 'exact' if args['policy'] == 'exact' else ('comp' if args['compensate'] else 'cache')
        rows = [json.loads(line) for line in (p.parent / 'epochs.jsonl').read_text().splitlines()]
        result_path = p.parent / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        runs.append(dict(name=p.parent.name, model=args['model'], arm=arm, rows=rows,
                         result=result, manifest=manifest))
    if not runs:
        return
    runs.sort(key=lambda r: (r['model'], ('exact', 'cache', 'comp').index(r['arm'])))
    assert all(r['manifest']['source_hashes'] == runs[0]['manifest']['source_hashes'] for r in runs)
    complete = len(runs) == 6 and all(r['result'] is not None and len(r['rows']) == 100 for r in runs)
    gconv_complete = sum(r['model'] == 'gconv_gru' and r['result'] is not None
                         and len(r['rows']) == 100 for r in runs) == 3
    curves, timing, targets, summary = [], [], [], []
    for run in runs:
        rows, result = run['rows'], run['result'] or {}
        if not rows:
            continue
        last = rows[-1]
        assert math.isclose(last['train_seconds_cumulative'], sum(p['train_seconds'] for p in rows), rel_tol=1e-8)
        for p in rows:
            audit = p['state_reads']
            curves.append(dict(model=run['model'], arm=run['arm'], epoch=p['epoch'],
                train_mse=p['train_mse'], val_mse=p.get('val_mse'), train_seconds=p['train_seconds'],
                train_seconds_cumulative=p['train_seconds_cumulative'], eval_seconds_cumulative=p['eval_seconds_cumulative'],
                wall_seconds=p['run_seconds_cumulative'], gamma=p.get('gamma'), gamma_effective=p.get('gamma_effective'),
                hot_mean_lag=audit['history_hot_mean_lag'], cold_mean_lag=audit['history_cold_mean_lag'],
                stale_nonzero_increment_rows=audit['stale_remote_nonzero_increment_rows']))
        record = dict(model=run['model'], arm=run['arm'], epochs=len(rows),
            mean_epoch_seconds=statistics.mean(p['train_seconds'] for p in rows),
            median_epoch_seconds_excluding_first=statistics.median(p['train_seconds'] for p in rows[1:] or rows),
            train_minutes=last['train_seconds_cumulative']/60,
            train_validation_minutes=(last['train_seconds_cumulative']+last['eval_seconds_cumulative'])/60,
            wall_minutes_to_last_epoch=last['run_seconds_cumulative']/60,
            peak_allocated_gib=result.get('peak_allocated_bytes_rankmax', 0)/2**30,
            peak_reserved_gib=result.get('peak_reserved_bytes_rankmax', 0)/2**30)
        timing.append(record)
        for threshold in ((.05, .02) if run['model'] == 'dcrnn' else (.1, .05)):
            hit = next((p for p in rows if p.get('val_mse', math.inf) <= threshold), None)
            targets.append(dict(model=run['model'], arm=run['arm'], threshold=threshold,
                epoch=hit['epoch'] if hit else None,
                train_minutes=hit['train_seconds_cumulative']/60 if hit else None,
                train_validation_minutes=(hit['train_seconds_cumulative']+hit['eval_seconds_cumulative'])/60 if hit else None,
                wall_minutes=hit['run_seconds_cumulative']/60 if hit else None))
        summary.append(dict(name=run['name'], **record, best_epoch=last['best_epoch'],
            best_val_mse=last['best_val_mse'], test_mse=result.get('test_mse_operational'),
            test_mse_exact=result.get('test_mse_exact'), test_rmse=math.sqrt(result['test_mse_operational']) if result else None,
            last_gamma_effective=last.get('gamma_effective'),
            stale_nonzero_increment_rows=sum(p['state_reads']['stale_remote_nonzero_increment_rows'] for p in rows)))
    for name, records in [('curves.csv', curves), ('timing.csv', timing), ('time_to_target.csv', targets)]:
        write_csv(name, records)
    (OUT / 'summary.json').write_text(json.dumps(dict(complete=complete, runs=summary), indent=2)+'\n')
    for name, xaxes, yaxes in [
        ('convergence', ['epoch', 'epoch'], ['train_mse', 'val_mse']),
        ('convergence_time', ['train_seconds_cumulative', 'run_seconds_cumulative'], ['val_mse', 'val_mse'])]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for i, model in enumerate(('dcrnn', 'gconv_gru')):
            for j, (xkey, ykey) in enumerate(zip(xaxes, yaxes)):
                ax = axes[i, j]
                for run in runs:
                    if run['model'] != model:
                        continue
                    points = [p for p in run['rows'] if ykey in p]
                    divisor = 1 if xkey == 'epoch' else 60
                    ax.plot([p[xkey]/divisor for p in points], [p[ykey] for p in points],
                            label=LABELS[run['arm']], color=COLORS[run['arm']], linewidth=1.5,
                            linestyle={'exact':'-', 'cache':'--', 'comp':':'}[run['arm']],
                            marker='o' if ykey == 'val_mse' else None, markersize=3)
                ax.set(title=f'{model.upper()} | {ykey}', ylabel='MSE',
                       xlabel={'epoch':'Epoch', 'train_seconds_cumulative':'Training time (min)',
                               'run_seconds_cumulative':'CLI wall time (min)'}[xkey])
                ax.set_yscale('log')
                ax.grid(alpha=.2)
                if ax.lines:
                    ax.legend(fontsize=8)
        fig.suptitle('Flickr | window cache | seed 42' + ('' if complete else ' | IN PROGRESS'))
        fig.savefig(OUT / f'{name}.png', dpi=170)
        fig.savefig(OUT / f'{name}.pdf')
        plt.close(fig)
    gaps = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, model in zip(axes, ('dcrnn', 'gconv_gru')):
        baseline = next((r for r in runs if r['model'] == model and r['arm'] == 'exact'), None)
        reference = {p['epoch']:p['val_mse'] for p in baseline['rows'] if 'val_mse' in p} if baseline else {}
        for run in runs:
            if run['model'] != model or run['arm'] == 'exact':
                continue
            points = [dict(model=model, arm=run['arm'], epoch=p['epoch'],
                           relative_val_mse_gap_percent=100*(p['val_mse']/reference[p['epoch']]-1))
                      for p in run['rows'] if 'val_mse' in p and p['epoch'] in reference]
            gaps.extend(points)
            ax.plot([p['epoch'] for p in points], [p['relative_val_mse_gap_percent'] for p in points],
                    label=LABELS[run['arm']], color=COLORS[run['arm']],
                    linestyle='--' if run['arm'] == 'cache' else ':', marker='o', markersize=3)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set(title=model.upper(), xlabel='Epoch', ylabel='Validation MSE gap vs exact (%)')
        ax.grid(alpha=.2)
        if len(ax.lines) > 1:
            ax.legend(fontsize=8)
    fig.savefig(OUT / 'validation_gap.png', dpi=170)
    fig.savefig(OUT / 'validation_gap.pdf')
    plt.close(fig)
    write_csv('validation_gap.csv', gaps)
    fmt = lambda x, digits=6: '待完成' if x is None else f'{x:.{digits}f}'
    lines = ['# Flickr 滑动窗口缓存：收敛与耗时', '',
        '**状态：'+('六组各 100 轮及测试均完成。' if complete else
                    'GConvGRU 三组完成；DCRNN 缓存/补偿暂缓。' if gconv_complete else
                    'GConvGRU 运行中；DCRNN 缓存/补偿暂缓。')+'**', '',
        '**语义范围：本轮 W=1，comp 组使用 γ 缩放 increment 的原公式。'
        '它不验证随后明确的“本地 UPDATE 输出与共享预测状态加权”的多槽位平滑聚合。**', '',
        '执行顺序已按用户要求改为 GConvGRU 优先。DCRNN exact 已完成；'
        '其缓存组在首轮记录前停止，补偿组未启动，两组暂缓。'
        + ('GConvGRU 三组均已完成。' if gconv_complete else 'GConvGRU 缓存及补偿组按顺序继续。'), '',
        'DCRNN / GConvGRU，各做 exact、缓存不补偿、缓存加累计 increment 与可学习 γ。'
        'Flickr 节点下一快照 log-in-degree 回归，seed=42、hidden=8、lr=0.001、两卡、10% 热点。'
        '沿用每 batch 一个快照，W=1 加前驱槽；checkpoint 按实际策略验证 MSE 选择。', '',
        '| 模型 | 组别 | 轮数 | 最优验证 MSE | 选中轮 | 实际 test MSE | 同 checkpoint exact test MSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['model']} | {s['arm']} | {s['epochs']} | {fmt(s['best_val_mse'])} | {s['best_epoch']} | {fmt(s['test_mse'])} | {fmt(s['test_mse_exact'])} |")
    lines += ['', '![收敛](convergence.png)', '', '![相同轮次验证误差差异](validation_gap.png)', '',
        '| 模型 | 组别 | 平均训练秒/轮 | 第 2 轮起中位秒/轮 | 训练分钟 | 训练+验证分钟 | 至末轮墙钟分钟 | 峰值 allocated GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for t in timing:
        lines.append(f"| {t['model']} | {t['arm']} | {t['mean_epoch_seconds']:.2f} | {t['median_epoch_seconds_excluding_first']:.2f} | {t['train_minutes']:.2f} | {t['train_validation_minutes']:.2f} | {t['wall_minutes_to_last_epoch']:.2f} | {t['peak_allocated_gib']:.2f} |")
    lines += ['', '![误差随耗时变化](convergence_time.png)', '',
        '每个模型固定使用同一对 A40，三组按顺序执行；两个模型不共享 GPU，但共享 CPU 和磁盘。'
        '训练计时含读取审计，不含轮前重置和轮末审计汇总；验证含预热重放。'
        '墙钟从 CLI 入口计时，含启动加载、状态重置和检查点，末轮墙钟不含最终测试。'
        'Prepare 不在上述计时中；首轮可能与 Prepare 收尾重叠，另列第 2 轮起中位数。'
        '峰值是整个运行中两 rank 最大的 PyTorch allocated 值，非整卡使用量；缓存组还建立 exact 重放状态管理器。'
        '这些是带审计的实测运行时间，未分解通信、计算、热点复制和缓存开销。', '',
        '| 模型 | 组别 | 验证 MSE 阈值 | 首次观测轮 | 训练分钟 | 训练+验证分钟 |',
        '|---|---|---:|---:|---:|---:|']
    for t in targets:
        lines.append(f"| {t['model']} | {t['arm']} | {t['threshold']} | {t['epoch'] or '未达到'} | {fmt(t['train_minutes'],2)} | {fmt(t['train_validation_minutes'],2)} |")
    lines += ['', '验证每五轮一次，达到阈值的时间只按观测点判断。', '',
        '| 模型 / 组别 | 最后训练轮热点历史平均滞后 | 冷邻居平均滞后 | 全程滞后且 increment 非零行数 | 最终 sigmoid(γ) |',
        '|---|---:|---:|---:|---:|']
    for run, s in zip(runs, summary):
        audit = run['rows'][-1]['state_reads']
        lines.append(f"| {s['model']} / {s['arm']} | {audit['history_hot_mean_lag']:.4f} | {audit['history_cold_mean_lag']:.4f} | {s['stale_nonzero_increment_rows']} | {fmt(s['last_gamma_effective'],4)} |")
    lines += ['', '滞后审计读取模型补偿前的实际缓存；increment 非零且滞后才可能产生非零补偿。'
        'γ 列为生效系数 sigmoid(γ)，原始参数也保留在 JSONL。'
        '新路径的 shared mask 包括所有非 owner 历史行（热点副本与冷邻居），不能直接与上次 hot-only shared 比例比较。', '',
        '这是单 seed、固定 100 轮预算实验；是否达到收敛平台需检查末段趋势。'
        'Flickr persistence 基线：验证 MSE=0.00562756，测试 MSE=0.01507541（同一数据与划分）。'
        'DCRNN 仍有逐快照 reset-gate 交换；max_staleness 控制发布跳过次数，不是实际快照年龄上界。', '',
        '配置和限制见 [protocol.md](protocol.md)，原始指标见 [raw](raw)，'
        '汇总见 [summary.json](summary.json)，时间明细见 [timing.csv](timing.csv) 与 '
        '[time_to_target.csv](time_to_target.csv)。运行 `python analyze.py` 可重新生成图表。']
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    report()
