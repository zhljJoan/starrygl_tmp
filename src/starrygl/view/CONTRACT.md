# View 与 GraphBlock 契约

状态：`GraphBlock` 继续作为唯一的模型图输入，不新增平行的图容器。

## GraphBlock 表示什么

一个 `GraphBlock` 是模型本层计算需要的图张量：

```text
src_nodes             源节点的全局编号
dst_nodes             目标节点的全局编号
edge_ids              边在数据集中的稳定编号
edata["edge_rows"]    边在排序后全局边表中的物理行
format                csc / csr / coo / native
indptr, indices       压缩邻接
srcdata/dstdata/edata 与节点或边同顺序的数据
route                 跨 rank 数据放置方式
```

`edge_ids` 用于结果和任务对齐，`edge_rows` 用于读取边特征和时间。二者顺序
必须和图中的边完全一致。

Event MFG 的 node row 还必须在 `srcdata/dstdata` 正式保存 `cutoff_ts`；同一
`node_id` 的不同 cutoff 是不同 row。Snapshot 的 scatter/read row 必须能恢复
对应 `snapshot_id`，保证 `[K,N,F]` 按 `(snapshot_id,node_id)` 读取。这些字段
不能只靠 tuple 位置或 Python cache 猜测。

`Batch` 不提供 `meta`。`GraphBlock.cache` 只保存迁移期运行时/调试值；影响计算
正确性的字段不能只藏在 cache 中。

## 图访问返回什么

Event/T-CSR 和 Snapshot-CSC 最后都返回同一个短 tuple：

```text
(blocks, node_ids, edge_rows)
```

- `blocks` 是模型使用的图结构。
- `node_ids` 是本批 read table；Event 可在采样后按静态 feature 语义合并，
  Snapshot 时间版本特征只能按 `(snapshot_id,node_id)` 合并。
- `edge_rows` 是本批需要读取边特征的去重物理行。

采样模式的 `blocks` 形状是 `[历史窗口][GNN 层]`。完整 Snapshot 每个历史
窗口只保存一个 CSC，模型按 `num_layers` 重用，不能复制同一份 CSC。

不为这个 tuple 新建只有一个实现的包装类。

## Route 是什么

Route 只回答通信的三个问题：

1. 本 rank 要给每个 peer 发送多少行。
2. 从本地 tensor 的哪些行打包。
3. 收到后写入源节点表的哪些位置。

`GraphBlock.route` 只保存该 block 的 prepared layerwise boundary Route。
Snapshot 图固定，因此 Prepare 可预计算它。Sample-dependent feature/state read
Route 是 Stage B 的短生命周期 tensor：先交换 request rows，再交换 payload，
不能复制到每个 block 或藏在 cache。两种 Route 复用相同的 sizes/index/scatter
语义和薄通信上下文，不是同一个生命周期对象。

## 不允许的第二套接口

EventView、T-CSR 和 Snapshot-CSC 是存储方式，不是模型 API。模型不能接收
`EventBatch`、`SnapshotBatch`、`PartitionData` 或后端字典。DGL block 可以
按需生成，但不能成为每批重复构建的主路径。

## 当前未对齐

- 采样边的物理行仍藏在 `cache["edge_feature_ids"]`，应移动到
  `edata["edge_rows"]`。
- 一些去重和 `col` 构造仍在 Python 热路径，需要保留实际使用路径并把热点
  工作移到 Torch 或 C++。
