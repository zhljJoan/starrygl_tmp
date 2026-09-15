# 图划分契约

状态：主要字段已经存在，但原生采样器仍有一个索引位数错误。

## 负责什么

图划分只决定两件事：数据由哪个 rank 负责，以及数据在该 rank 的哪一行。
它不负责生成每个批次的通信 Route。

固定划分器输出：

```text
node_master[N]    每个节点的权威所有者
edge_master[E]    每条边及其任务的权威所有者
shared_nodes      可在各 rank 缓存的热点节点
node_replicas     本地计算需要的节点副本
edge_replicas     可选的边副本
node_to_chunk[N]  节点所属的空间 chunk
edge_chunk[E]     边所属的 chunk
chunk_table       chunk 的 packed node/time 物理范围
route_table       owner 读取和静态 Snapshot boundary 的通信 metadata
```

当前不开公开的多策略对象。具体划分算法可以替换，但上面的输出含义不能变。

## 如何定位数据

Prepare 把 owner 和 owner 内部行号压进一个 64 位整数：

```text
dist_index = (owner_rank << 48) | owner_local_row
```

`node_dist_index` 按全局节点编号索引，`edge_dist_index` 按排序后的全局边行
索引。Python 和 C++ 必须统一使用 48 位保存本地行号。

热点节点即使在多个 rank 上有缓存，`dist_index` 仍然指向唯一的权威副本。

## 所有权规则

- 节点任务的输出、loss 和指标由 `node_master` 负责。
- 边任务的输出、loss 和指标由 `edge_master` 负责。
- Snapshot-CSC 的目标节点位于本地 `node_master`，源节点可以来自其他 rank。
- 热点缓存不能负责 checkpoint，也不能改变任务归属。

## Route 在哪里生成

- Snapshot-CSC 的图结构固定，Prepare 直接预计算 Route。
- T-CSR 的采样结果每批不同，采样结束后生成 Route。
- 状态写回使用单独的写 Route。

`route_table` 只保存已有的静态 tensor/mapping，不新增 RouteTable 类。动态 sampled
Route 仍由每批 requester rows 生成，不能塞进 `PartitionPlan`。

## 当前未对齐

- C++ 的边位置解析仍使用 `>> 50`，正确值是 `>> 48`。
- 从旧产物恢复 `PartitionPlan` 时，部分 replica 信息还没有还原。
