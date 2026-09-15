# Task 契约

状态：Event、采样 Snapshot 和完整 Snapshot 已共用 prepared task slice 与
`TargetRoute` 构造。

## 任务输入

Prepare 已按 owner 生成平铺任务表。运行时窗口 `k` 只切一次：

```text
begin = task_ptr[k]
end   = task_ptr[k + 1]
target = payload[begin:end]
```

目标字段只有：

```text
节点任务：node_ids, label，可选 Event cutoff_ts
边任务：  src, dst, edge_ids, edge_rows, label，可选 Event cutoff_ts
```

不再保留含义重复的 `target_ids`、`target_snapshot_id` 和目标时间副本。任务表
已经按 owner 分片，所以 Batch 中也不需要每行重复保存 owner。

## 一次任务如何执行

```text
读取正样本
  -> 需要时生成负样本
  -> 把正负端点作为采样 roots
  -> 图访问和节点压缩
  -> 记录端点在 embedding 表中的行号
  -> 模型计算
  -> 必要时把远程端点 embedding 收集到 edge_master
  -> loss 和指标
```

Task 不选择窗口策略，不构造图，不直接发起 collective，也不更新时序状态。

## Snapshot 标签如何对齐

如果任务用 Snapshot `k` 的 embedding 预测 `k+1`，Prepare 把 `k+1` 的标签
直接放在任务行 `k`。最后一个没有标签的窗口得到空切片，运行时不再写
`if edge_prediction: drop last`。

## 负采样

当前只要求随机负采样：

```text
mode="dst"      保留正样本 src，只生成 neg_dst
mode="src_dst"  同时生成 neg_src 和 neg_dst
```

训练负样本在 sampler 所在设备向量化/native 在线生成，并在邻居采样前加入
roots。首版 CPU native sampler 直接生成 CPU roots，避免 GPU -> CPU 同步；以后
只有整体切换到 GPU sampler 时才一起迁移。验证和测试使用固定随机种子，可在
当前运行中缓存。

Local/global 候选域混合仍是同一个 `random` policy，不新增策略框架。
`neg_loss_weight` 只表示 UDF 明确返回的 loss 权重，不能自动设置成采样概率的
倒数；没有目标分布和校正公式时不声称 importance correction。

## 端点 Route

图压缩后，`TargetRoute` 记录正负端点对应的 embedding 行。如果 edge owner
没有本地计算某个端点，必须通过 runtime 的 Route 通信精确取回。热点缓存不能改变
edge task 的 loss 所有者。

## 当前未对齐

- `TaskTarget` 仍包含 `target_ids`、`target_ts` 和逐窗口负样本池对象。
