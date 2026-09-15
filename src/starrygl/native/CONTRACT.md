# C++ 邻居采样契约

状态：Event 采样和大部分压缩输出已经存在；Snapshot 行区间采样尚未完成。

## 只初始化一个 sampler

训练任务开始时，用 Prepare 生成的 T-CSR 初始化一次 sampler：

```text
T-CSR 邻接和时间
每个邻接项的 edge_ids / edge_rows
fanouts、GNN 层数、采样策略
local_rank
node_part、edge_part、node_is_hot
node_dist_index、edge_dist_index
```

`node_part` 和 `edge_part` 用于判断本地、远程和边界节点。packed index 用于
生成特征与状态的读取 Route。双向 T-CSR 可能有 `A=2E` 个邻接项，因此
`edge_part[A]` 应由 `edge_dist_index[edge_rows]` 得到。

sampler 的模式在初始化时绑定。训练热路径只调用：

```text
sample_neighbors(roots, scope)
```

不再提供 `sample_event`，也不为每个 Snapshot 创建 sampler。

## Event 如何采样

```text
roots：需要计算的全局节点编号
scope：每个 root 的截止时间
```

第一层只采样 `edge_ts < root_ts` 的邻边，没有最早时间限制。下一层使用刚采到
的边时间作为新的截止时间，从而保持因果顺序。

## Snapshot 如何采样

对于窗口 `k`，`scope` 是最近 `H` 个 Snapshot 的边行区间：

```text
time_ptr_2[max(0, k-H+1):k+1]   # 形状 [H,2]
```

`H=snaps_count`，包含当前 Snapshot。历史范围可以跨越 train/val/test 边界，
因为 split 只限制监督信号。

C++ 对每个区间独立做 uniform 邻居采样，并在每个 GNN 层使用同一组区间。
一次 C++ 调用完成所有历史窗口和层的循环，Python 不能逐 Snapshot 调用。

Snapshot 邻居采样不能与 `chunk_decay` 组合。

## 返回什么

```text
mfgs[history][layer]       每层的 CSC/MFG
node_ids / node_ptr        本批涉及的节点及分段
edge_ids / edge_rows       逻辑边编号和物理边行
root_rows                  roots 在结果节点表中的位置
node_read_index / ptr      节点读取分组
edge_read_index / ptr      边读取分组
scatter rows               特征放回图结构的位置
```

采样、去重、分组和 scatter 映射在 C++ 或 Torch 中完成。CPU sampler 输出紧凑
或 pinned tensor，Stage B 再异步搬到 GPU。将来可以替换为 GPU sampler，但
调用和输出契约不能变化。

## 当前未对齐

- C++ 仍暴露 `neighbor_sample_from_nodes` 和 `sample_dtdg_uniform` 两套入口。
- `sample_dtdg_uniform` 按整数时间比较，尚不能直接接收边行区间。
- Python 没有调用 `set_edge_read_dist_index`。
- C++ 用 `>> 50` 解析 owner，Prepare 使用的是 `>> 48`。
- Python 仍会重建、排序拓扑，并执行部分 MFG 去重。

