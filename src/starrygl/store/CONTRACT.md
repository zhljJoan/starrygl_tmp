# Store 契约

状态：任务表已落入 `LabelStore`；特征行名和统一依赖读取仍待收敛。

## 负责什么

运行时只加载一个 `StoreBundle`：

```text
GraphStore       窗口表、划分索引和图视图
FeatureManager  节点/边特征及本地行映射
LabelStore      标签、task_ptr 和任务 payload
StateManager    训练开始后创建的时序状态
```

模型和任务不能直接打开 `prepare.pt` 或 `graph_RRR.pt`。

## GraphStore

所有视图使用相同的全局窗口编号。`GraphStore` 直接提供：

```text
time_ptr_2
split 对应的全局窗口范围
node_dist_index / edge_dist_index
node_to_chunk / edge_chunk
ExecutionPlan 需要的图视图
```

它不再把 split 内编号转换成另一套 Snapshot 编号，也不搜索重复的窗口表。

## FeatureManager

节点读取输入全局 `node_ids`，边读取输入物理 `edge_rows`：

```text
read_nodes(node_ids, window_id)
read_edges(edge_rows)
```

只有 `[K,N,F]` 特征使用 `window_id`。静态特征忽略它。嵌入 Snapshot-CSC
只是另一种存放方式，不能产生第二套读取接口。

本地行映射命中时直接读取；缺失行根据 packed index 从 owner 获取。逻辑
`edge_ids` 不能当作特征行使用。

## StateManager

权威状态由 `node_master` 保存。本地缓存和热点缓存只提供更快的读取路径，
不负责 checkpoint。状态类型和允许的新鲜度来自 `ExecutionPlan`。

`compile` 只在 plan 中登记 `state_dependencies`，不分配逐批状态。trainer 的
`fit`、`evaluate` 和 `predict` 进入 epoch 前，根据这些依赖调用
`build_state_managers`，一次分配
owner values、可选 timestamps/mailbox 和显式 stale-cache tensor；模型只返回
`StateDelta`，runtime 在 backward 后校验并提交。Snapshot Batch-local recurrent
tensor 不注册到长期 manager，只有 plan 允许的跨 Batch state 才持久化。

## 压缩读取

Snapshot-CSC、Route 和嵌入特征都按 `data + ptr` 保存。读取窗口 `k` 时直接
切 tensor，不得重建 CSC 或反序列化一组 Python 对象。

Stage B 持有尚未完成的读取句柄，完成后只把 tensor 放入 `Batch`。

## 当前未对齐

- 边特征调用方仍混用 `ids` 和 `edge_feature_ids`，目标接口只接收
  `edge_rows`。
- 独立特征和 Snapshot 内嵌特征仍走不同的运行时 finish 函数。
