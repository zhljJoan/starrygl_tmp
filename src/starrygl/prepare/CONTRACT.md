# Prepare 契约

状态：owner-local `task_ptr/payload` 已写入 label shard；其余未对齐项见文末。

## 统一输入

Prepare 只排序一次，得到按时间排列的边表：

```text
src[E], dst[E]       全局节点编号
ts[E]                事件时间或 Snapshot 序号
edge_ids[E]          数据集中的稳定边编号
edge row             该边在排序后表中的位置 0..E-1
time_ptr_2[K, 2]     第 k 个窗口对应的半开边区间 [begin, end)
```

`edge_ids` 和 `edge_rows` 不能混用。例如某条边原始编号是 42，排序后可能位于
第 7 行。结果对齐使用 42，读取时间、特征和 owner 使用 7。

训练、验证和测试只限制哪些窗口产生监督信号，不截断更早的历史图数据。

## 处理顺序

```text
加载并按时间排序
  -> 生成 PartitionPlan
  -> 生成 node_dist_index / edge_dist_index
  -> 生成 node chunk
  -> 按 owner 生成任务表
  -> 按 ExecutionPlan 生成图视图
  -> 划分特征和标签
  -> 写入压缩产物
```

必须先划分再建视图，因为 Snapshot 的本地目标节点和所有通信 Route 都依赖
真实 owner。

## 任务表

每个 rank 只保存自己负责的任务。所有窗口共用一个前缀指针：

```text
task_ptr[K + 1]

节点任务：node_ids, label，可选 Event cutoff_ts
边任务：  src, dst, edge_ids, edge_rows, label，可选 Event cutoff_ts
```

窗口 `k` 只执行：

```python
begin, end = task_ptr[k:k + 2]
target = payload[begin:end]
```

空切片表示该 rank 在这个窗口没有任务，不需要按任务名称删除最后一个窗口。

如果用 Snapshot `k` 的 embedding 预测 Snapshot `k+1` 的边，Prepare 直接把
`k+1` 的标签放在任务行 `k`。运行时不再保存 `target_snapshot_id`。

训练负样本在线生成。验证和测试使用固定随机种子；只有数据集明确提供时才
把负样本写入产物。

## 需要生成哪些图视图

| 执行方式 | Prepare 输出 |
| --- | --- |
| Event 邻居采样 | EventView + T-CSR |
| Snapshot 邻居采样 | T-CSR |
| Snapshot 完整计算 | Snapshot-CSC |
| Snapshot chunk decay | Snapshot-CSC + node chunk |

邻居采样不能和 `chunk_decay` 同时启用。

### EventView

每个 rank 按时间保存自己 `edge_master` 的边：

```text
edge_rows, edge_ids, src, dst, ts
本地 time_ptr_2, state_write_mask, state_write_routes
```

它用于 Event 输入和状态更新；监督数据仍从任务表读取。

### T-CSR

每个节点的邻边按时间排序，主要字段是：

```text
indptr, indices, ts, edge_rows, edge_ids
node_dist_index, edge_dist_index
node_to_chunk, edge_chunk, node_is_hot
```

默认构造双向邻接。训练时直接初始化一次 C++ sampler，不允许每个 epoch
重新从 COO 排序和建图。

### Snapshot-CSC

每个 rank、每个 Snapshot 保存：

```text
src_nodes, dst_nodes, indptr, indices
edge_rows, edge_ids, ts
node_chunk, 归一化系数, 源节点放置位置, Route
```

任务标签不放进 Snapshot-CSC，统一从任务表读取。

## 特征如何存放

节点特征按 `node_master` 划分，并可附带热点或明确需要的一跳副本。边特征按
`edge_master` 划分：

```text
node_ids, node_row_map, node_feat
edge_rows, edge_row_map, edge_feat
```

静态节点特征形状为 `[N,F]`，多时间版本为 `[K,N,F]`。只有 `[K,N,F]`
可以选择按 Snapshot 源节点顺序嵌入 Snapshot-CSC。边特征始终单独存放，
不能复制到每个 Snapshot。

## 文件布局

```text
prepare.pt       划分索引、chunk、全局窗口表和视图清单
temporal_csr.pt  一份只读 T-CSR
graph_RRR.pt     该 rank 的 EventView / Snapshot-CSC / Route
feature_RRR.pt   该 rank 的特征和行映射
label_RRR.pt     该 rank 的标签、task_ptr 和任务 payload
```

Snapshot 的大字段统一保存为一维 `data` 加 `ptr`，不能保存大量 Python 对象。
同一节点上的进程应共享或 mmap 只读 T-CSR，避免重复内存。

## 当前未对齐

- Event 和 Snapshot 对 `edge_ids`、物理边行使用了不同名字，需要统一为
  `edge_ids` 与 `edge_rows`。
- T-CSR 初始化前仍会在 Python 中重新排序拓扑。
