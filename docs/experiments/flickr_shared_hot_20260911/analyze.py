from pathlib import Path
import csv
import json
import math
import shutil
import statistics
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT / 'raw'


def report():
    runs = []
    for manifest_path in sorted(ROOT.glob('*/manifest.json')):
        folder = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        rows = []
        metrics = folder / 'epochs.jsonl'
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        result_path = folder / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        diagnostic_path = folder / 'diagnostic.json'
        diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
        args = manifest['arguments']
        arm = 'exact' if args['policy'] == 'exact' else ('shared_hot+comp' if args['compensate'] else 'shared_hot')
        runs.append(dict(name=folder.name, model=args['model'], arm=arm, manifest=manifest, rows=rows, result=result, diagnostic=diagnostic))
        raw = OUT / 'raw' / folder.name
        raw.mkdir(parents=True, exist_ok=True)
        for name in ('manifest.json', 'epochs.jsonl', 'plan.txt', 'result.json', 'diagnostic.json'):
            path = folder / name
            if path.exists() and path.resolve() != (raw / name).resolve():
                shutil.copy2(path, raw / name)
    if not runs:
        return False
    assert all(r['manifest']['source_hashes'] == runs[0]['manifest']['source_hashes'] for r in runs)
    final = len(runs) == 6 and all(r['result'] is not None or r['diagnostic'] is not None for r in runs)
    status = [{k:r[k] for k in ('name','model','arm')} | {'epochs_completed':len(r['rows']), 'result':r['result'], 'diagnostic':r['diagnostic']} for r in runs]
    (OUT / 'summary.json').write_text(json.dumps({'complete': final, 'runs':status},indent=2)+'\n')
    with (OUT / 'curves.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','arm','epoch','train_mse','val_mse','shared_fraction','shared_mean_lag','compensation_updates_rank0','train_seconds','train_seconds_cumulative','eval_seconds_cumulative'])
        for run in runs:
            for row in run['rows']:
                read = row['state_reads']
                writer.writerow([run['model'],run['arm'],row['epoch'],row['train_mse'],row.get('val_mse',''),read['shared_fraction'],read['shared_mean_lag'],row.get('compensation_updates_rank0',''),row['train_seconds'],row['train_seconds_cumulative'],row['eval_seconds_cumulative']])
    fig, axes = plt.subplots(2,2,figsize=(10,7), constrained_layout=True)
    colors = {'exact':'#2458a6','shared_hot':'#d85b26','shared_hot+comp':'#268849'}
    styles = {'exact':'-','shared_hot':'--','shared_hot+comp':':'}
    for i, model in enumerate(('dcrnn','gconv_gru')):
        for j, metric in enumerate(('train_mse','val_mse')):
            ax = axes[i,j]
            for run in runs:
                points = [row for row in run['rows'] if metric in row]
                if run['model'] != model or not points or run['diagnostic']:
                    continue
                arm = run['arm']
                ax.plot([p['epoch'] for p in points], [p[metric] for p in points], label=arm, color=colors[arm], linestyle=styles[arm], linewidth=1.7)
            ax.set(title=f'{model.upper()} — {metric}', xlabel='Epoch', ylabel='MSE')
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8)
    fig.suptitle('Flickr node regression | actual shared-hot reads | seed 42' + ('' if final else ' | RUNNING'))
    fig.savefig(OUT/'convergence.png',dpi=180)
    fig.savefig(OUT/'convergence.pdf')
    plt.close(fig)
    gaps = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, model in zip(axes, ('dcrnn', 'gconv_gru')):
        pair = [r for r in runs if r['model'] == model and not r['diagnostic']]
        exact = next((r for r in pair if r['arm'] == 'exact'), None)
        shared = next((r for r in pair if r['arm'] != 'exact'), None)
        if exact is None or shared is None:
            continue
        reference = {p['epoch']:p['val_mse'] for p in exact['rows'] if 'val_mse' in p}
        values = [dict(model=model, epoch=p['epoch'], exact_val_mse=reference[p['epoch']],
                       shared_val_mse=p['val_mse'], relative_gap_percent=100*(p['val_mse']/reference[p['epoch']]-1))
                  for p in shared['rows'] if 'val_mse' in p and p['epoch'] in reference]
        gaps.extend(values)
        ax.plot([p['epoch'] for p in values], [p['relative_gap_percent'] for p in values], 'o-', color='#d85b26', markersize=3)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set(title=model.upper(), xlabel='Epoch', ylabel='Validation MSE difference (%)')
        ax.grid(alpha=.2)
    fig.suptitle('Paired shared / exact - 1 | positive = higher shared MSE')
    fig.savefig(OUT/'validation_gap.png', dpi=180)
    fig.savefig(OUT/'validation_gap.pdf')
    plt.close(fig)
    if gaps:
        with (OUT/'validation_gap.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(gaps[0]))
            writer.writeheader()
            writer.writerows(gaps)
    timing, targets = [], []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for i, model in enumerate(('dcrnn', 'gconv_gru')):
        pair = [r for r in runs if r['model'] == model and not r['diagnostic']]
        if len(pair) != 2 or not all(r['rows'] for r in pair):
            continue
        common_epochs = min(len(r['rows']) for r in pair)
        for r in pair:
            rows = r['rows'][:common_epochs]
            last = rows[-1]
            train = last['train_seconds_cumulative']
            evaluation = last['eval_seconds_cumulative']
            assert math.isclose(train, sum(p['train_seconds'] for p in rows), rel_tol=1e-9)
            timing.append(dict(model=model, arm=r['arm'], paired_epochs=common_epochs,
                               train_seconds=train, validation_seconds=evaluation,
                               train_plus_validation_seconds=train+evaluation,
                               mean_epoch_seconds=train/common_epochs,
                               median_epoch_seconds=statistics.median(p['train_seconds'] for p in rows)))
            thresholds = (.1, .05, .04, .03, .02) if model == 'dcrnn' else (.3, .2, .15, .12, .1, .08, .06, .05)
            for threshold in thresholds:
                hit = next((p for p in rows if p.get('val_mse', math.inf) <= threshold), None)
                targets.append(dict(model=model, arm=r['arm'], paired_epochs=common_epochs,
                                    val_mse_threshold=threshold, epoch=hit['epoch'] if hit else None,
                                    train_seconds=hit['train_seconds_cumulative'] if hit else None,
                                    train_plus_validation_seconds=(hit['train_seconds_cumulative']+hit['eval_seconds_cumulative']) if hit else None))
            points = [p for p in rows if 'val_mse' in p]
            for j in range(2):
                minutes = [(p['train_seconds_cumulative'] + (p['eval_seconds_cumulative'] if j else 0))/60 for p in points]
                axes[i,j].plot(minutes, [p['val_mse'] for p in points], label=r['arm'], color=colors[r['arm']], linestyle=styles[r['arm']], linewidth=1.7)
        for j in range(2):
            axes[i,j].set(title=model.upper(), xlabel=('Training + validation' if j else 'Training')+' time (min)', ylabel='Validation MSE')
            axes[i,j].grid(alpha=.2)
            axes[i,j].legend(fontsize=8)
    fig.suptitle('Flickr | matched epoch budgets | concurrent GPU jobs')
    fig.savefig(OUT/'convergence_time.png', dpi=180)
    fig.savefig(OUT/'convergence_time.pdf')
    plt.close(fig)
    for filename, records in (('timing.csv', timing), ('time_to_target.csv', targets)):
        if records:
            with (OUT/filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    lines = ['# Flickr 实际 shared_hot 读取消融', '', '**状态：'+('4 组主对照已完成；2 组补偿开关有效性检查已结束。' if final else '训练仍在运行，以下为阶段性结果，不代表最终收敛精度。')+'**', '',
        '本地当前 starrygl-open；Flickr，节点下一快照 log-in-degree 回归；2 个 rank，10% hot 节点，hidden=8，Adam lr=0.001，seed=42，最多 100 轮，每 5 轮验证。训练/验证/测试按快照时间划分为 28/14/28 个窗口；各 split 最后一个无标签窗口推进状态但不计入损失。', '',
        '`bounded_stale` 直接使用当前 `AsyncMemoryCommitter/materialize_bounded` 返回的状态，未注入固定延迟。max_staleness=1 使用默认余弦刷新阈值 0.3；本实验 access_pipeline=False、wait_policy=block。checkpoint 按实际策略验证 MSE 选择，测试分别重放实际策略和 exact 状态。', '',
        '| 模型 | 组别 | 已完成轮数 | 最优验证 MSE | 选择轮数 | 实际读取 test MSE | exact 重放 test MSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in runs:
        if r['diagnostic']:
            continue
        last = r['rows'][-1] if r['rows'] else {}
        result = r['result'] or {}
        fmt = lambda x: ('有效性检查' if r['diagnostic'] else '待完成') if x is None else f'{x:.8f}'
        lines.append(f"| {r['model']} | {r['arm']} | {len(r['rows'])} | {fmt(last.get('best_val_mse'))} | {last.get('best_epoch','—')} | {fmt(result.get('test_mse_operational'))} | {fmt(result.get('test_mse_exact'))} |")
    lines += ['', '![收敛曲线](convergence.png)', '', '同轮验证误差差异（正值表示 shared_hot 的误差较大；只比较两组共同验证的轮次）：', '', '![同轮验证误差差异](validation_gap.png)', '', '时间对比（每个模型按两组共同完成的轮数统计）：', '', '| 模型 | 组别 | 配对轮数 | 平均每轮训练 / 秒 | 累计训练 / 分钟 | 训练＋验证 / 分钟 |', '|---|---|---:|---:|---:|---:|']
    for t in timing:
        lines.append(f"| {t['model']} | {t['arm']} | {t['paired_epochs']} | {t['mean_epoch_seconds']:.2f} | {t['train_seconds']/60:.2f} | {t['train_plus_validation_seconds']/60:.2f} |")
    lines += ['', '达到相同验证误差的时间（首次观测到达的验证轮次）：', '', '| 模型 | 组别 | 验证 MSE 阈值 | 首次轮数 | 训练 / 分钟 | 训练＋验证 / 分钟 |', '|---|---|---:|---:|---:|---:|']
    for t in targets:
        if t['val_mse_threshold'] not in ((.05, .02) if t['model']=='dcrnn' else (.1, .05)):
            continue
        train = f"{t['train_seconds']/60:.2f}" if t['epoch'] else '未达到'
        both = f"{t['train_plus_validation_seconds']/60:.2f}" if t['epoch'] else '未达到'
        lines.append(f"| {t['model']} | {t['arm']} | {t['val_mse_threshold']} | {t['epoch'] or '—'} | {train} | {both} |")
    lines += ['', '![时间与验证误差](convergence_time.png)', '',
        '计时为 rank 0 的同步 CUDA 实测值；训练包括逐批读取审计，但不包括每轮开始前的状态重置及结束后的审计汇总。验证包括状态重置、前序训练窗口预热和验证计算。“训练＋验证”是两段计时之和，未计入 Prepare、启动加载、训练轮间状态重置、检查点写盘和最终测试，不能当作端到端墙钟时间。验证每 5 轮执行，首次到达只在观测点判定；阈值用于事后时间比较。', '',
        '四组任务共享四张 A40，每个模型使用两个 rank；exact 在 GPU 0/1，shared_hot 在 GPU 2/3，同一策略下两个模型并行。access_pipeline=False 且存在审计开销，因此只能描述本次实验耗时，不能归因为策略的独占吞吐差异。最终测试中 shared_hot 组额外重放 exact，测试总耗时不能直接与 exact 组相除。详细时间见 [timing.csv](timing.csv) 和 [time_to_target.csv](time_to_target.csv)。', '',
        'GConvGRU shared 组保留补偿开关；即使增量为零，查表、掩码和额外梯度处理仍执行，因此该组时间差也包含补偿路径开销。初期两组短补偿检查也曾并行运行。当前代码路径仍需 cold-node owner 拉取，并额外执行 hot 候选过滤和发布；GCN 计算量不随缓存命中减少。上述路径事实说明 shared_hot 不保证加速，但各部分对本次耗时的贡献尚未 profiling，不能由总时间差直接归因。', '',
        '实际读取审计（四组主实验最后一个已完成训练轮次）：', '', '| 模型 / 组别 | 训练 shared 读取比例 | shared 平均滞后 | owner 滞后行 | 未来状态行 |', '|---|---:|---:|---:|---:|']
    for r in runs:
        if not r['rows'] or r['diagnostic']:
            continue
        s = r['rows'][-1]['state_reads']
        lines.append(f"| {r['model']} / {r['arm']} | {s['shared_fraction']:.4%} | {s['shared_mean_lag']:.4f} | {s['owner_stale_rows']} | {s['future_rows']} |")
    lines += ['', 'shared 比例按实际请求的节点状态行计算，不是消息边的权重比例。状态由快照 s 产生时记录版本 s+1；快照 t 期望版本 t，实际滞后为 t−返回版本。max_staleness 控制跳过发布刷新次数，结果中的滞后来自真实时间戳审计。', '',
        '当前补偿在本次 owner-only 的 full-snapshot 分区中没有非零更新：shared 行属于远端节点，当前增量估计器只更新本地计算的节点。小图 with/without compensation 训练、验证及测试数值相同；完整运行记录计数和 gamma，不能把零更新解释为补偿带来的提升。', '',
        '两组重复的补偿开关长跑在前 4 轮训练 MSE 逐轮完全相同、增量更新为零后停止；保留 DCRNN exact/shared_hot 和 GConvGRU exact/shared_hot+comp 主对照。GConvGRU 的 comp 标志保留当前默认开关，但本布局下补偿没有非零更新。\n\n单个 seed 的成对试验用于初步评估，不能说明多次运行的统计显著性。训练计时包含审计，多个任务共享设备，不用于吞吐性能结论。']
    baseline = OUT/'task_baselines.json'
    if baseline.exists():
        data = json.loads(baseline.read_text())
        if baseline.resolve() != (OUT/baseline.name).resolve():
            shutil.copy2(baseline,OUT/baseline.name)
        lines += ['', f"任务参照：直接沿用当前快照的 log-in-degree，验证 MSE={data['val']['persistence_mse']:.8f}，测试 MSE={data['test']['persistence_mse']:.8f}。这有助于判断绝对预测精度，不能仅因 exact/stale 接近就认定模型已经充分收敛。"]
    if final:
        lines += ['', '完整主对照的测试误差变化（不同训练轨迹、分别按验证选择检查点）：', '']
        for model in ('dcrnn', 'gconv_gru'):
            pair = [r for r in runs if r['model']==model and not r['diagnostic']]
            exact_run = next(r for r in pair if r['arm']=='exact')
            shared_run = next(r for r in pair if r['arm']!='exact')
            a, b = exact_run['result'], shared_run['result']
            gap = 100*(b['test_mse_operational']/a['test_mse_operational']-1)
            lines.append(f"- {model}: shared 主对照相对 exact 主对照的 test MSE 为 {gap:+.3f}%；检查点分别为第 {b['best_epoch']} / {a['best_epoch']} 轮。此差异包含训练轨迹与检查点选择差异，不能当作单次 stale 读取的收益。")
        lines += ['', '实际策略测试重放的状态读取审计：', '', '| 模型 / shared 组 | shared 读取比例 | shared 平均滞后 | owner 滞后行 | 未来状态行 |', '|---|---:|---:|---:|---:|']
        for r in runs:
            if r['diagnostic'] or r['arm']=='exact':
                continue
            s = r['result']['test_state_reads']
            lines.append(f"| {r['model']} | {s['shared_fraction']:.4%} | {s['shared_mean_lag']:.4f} | {s['owner_stale_rows']} | {s['future_rows']} |")
        starts_path = OUT/'process_starts.json'
        if starts_path.exists():
            starts = json.loads(starts_path.read_text())
            wall = {}
            lines += ['', '进程启动到最后一轮训练及验证记录写完的墙钟时间：', '', '| 模型 | 组别 | 墙钟 / 分钟 |', '|---|---|---:|']
            for r in runs:
                if r['diagnostic'] or r['name'] not in starts:
                    continue
                end = (ROOT/r['name']/'epochs.jsonl').stat().st_mtime
                elapsed = end-starts[r['name']]['process_start_unix']
                timed = r['result']['train_seconds']+r['result']['validation_seconds']
                assert elapsed >= timed-1
                wall[r['name']] = dict(startup_to_training_log_seconds=elapsed, logged_intervals_seconds=timed, log_mtime_unix=end)
                lines.append(f"| {r['model']} | {r['arm']} | {elapsed/60:.2f} |")
            (OUT/'wall_time.json').write_text(json.dumps(wall, indent=2)+'\n')
            lines += ['', '墙钟按 Linux 进程启动时间与最后一轮 epochs.jsonl 修改时间重建，精度约秒级；包含启动加载、训练、验证、轮间状态重置和训练阶段检查点写入，不含数据转换、Prepare 和最终测试。原始起点见 process_starts.json，终点随 raw/ 的 copy2 保留。']
        lines += ['', '同一检查点切换读取方式的影响：', '']
        for r in runs:
            if r['arm']=='exact' or r['result'] is None:
                continue
            d=r['result']; gap=d['test_mse_operational']-d['test_mse_exact']
            lines.append(f"- {r['model']} / {r['arm']}: ΔMSE={gap:+.8g} ({100*gap/max(d['test_mse_exact'],1e-20):+.5f}%)，实际读取 RMSE={math.sqrt(d['test_mse_operational']):.8f}。")
        lines += ['', '固定预算末段趋势（80 → 100 轮的验证 MSE；下降仍不等于达到平台）：', '']
        for r in runs:
            if r['diagnostic']:
                continue
            by_epoch = {p['epoch']:p for p in r['rows']}
            if 80 in by_epoch and 100 in by_epoch:
                a, b = by_epoch[80]['val_mse'], by_epoch[100]['val_mse']
                lines.append(f"- {r['model']} / {r['arm']}: {a:.8f} → {b:.8f} ({100*(b/a-1):+.2f}%)。")
    if final:
        lines += ['', '四组最后 20 轮的验证误差仍下降约 14%–21%，因此这里只报告固定 100 轮预算的精度，不能宣称已经充分收敛。所有模型的测试 MSE 也高于上述 persistence 基线。', '', '最终核验见 [verification.json](verification.json)：每组 100 个训练记录、21 次验证、27 个测试计分窗口；两个 rank 的模型检查点逐张量一致；所有已记录训练、验证和测试读取均无 owner 滞后或未来状态。实现测试记录为 272 passed / 15 skipped，另通过 DCRNN 单/双 rank 完整训练与梯度对齐检查。']
    lines += ['', '原始配置、源码 SHA256、逐轮指标和结果见 [raw](raw)，汇总数据见 [summary.json](summary.json) 与 [curves.csv](curves.csv)。复现实验见 [reproduce.md](reproduce.md)，实际算法见 [algorithm.md](algorithm.md)。运行 `python analyze.py` 可用本目录归档重新生成报告和图表。所有数字来自当前运行；不混入历史 TGM 结果。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':final,'epochs':{r['name']:len(r['rows']) for r in runs}}),flush=True)
    return final


if __name__ == '__main__':
    while True:
        if report() or '--watch' not in sys.argv:
            break
        time.sleep(45)
