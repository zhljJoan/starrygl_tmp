# Batch 契约

状态：保留现有 `Batch`，删除只为不同执行模式服务的额外包装。

## Batch 是什么

`Batch` 是 Stage B 完成可预取读取后交给 Stage A 的唯一输入容器。Stage A 补齐
exact state 后，它才是交给模型的完整输入：

```text
features    已取得的节点和边特征
state       按状态类型组织的时序状态
targets     本批监督目标，以及可选的 Event 状态更新输入
blocks      采样 MFG 或多个完整 Snapshot 图
graph       单个完整图的快捷入口
num_layers  完整图需要重复执行的 GNN 层数
```

rank、epoch、step、split 等 loader 控制信息不随 Batch 传递。通信句柄、
ExecutionPlan、sampler 配置和状态版本也不能进入 Batch。

## blocks 如何组织

```text
Event 采样：      blocks[0][layer]
Snapshot 采样：   blocks[history][layer]
Snapshot 完整图： blocks[history][0]，按 num_layers 重用
单个完整图：      graph，按 num_layers 重用
```

每层 block 自己包含该层所需的源节点。Stage B 只读取一次去重后的特征和
状态，再按预先生成的行号放回各层位置。

## 任务和状态

`targets["task"]` 在采样前生成，之后一直复用到 loss 和指标计算。`EventRows`
只保存本窗口按原顺序排列的 `src/dst/edge_ids/ts/state_write_mask`，供 Event
特征读取和 memory/mailbox 更新使用；它不负责去重，也不是第二份标签。

Event 根节点按 `(node_id, cutoff_ts)` 去重；静态节点特征再按 `node_id` 去重并在
读取后恢复 sampler 布局；memory 更新按 node 选择本窗口最后一次有效写入。
这三种去重语义不同，不能塞进 `EventRows` 的构造函数。

`state` 的 key 来自计划，例如 `node_memory` 或 `neighbor_recurrent`。读取策略由
计划声明为 exact 或 stale-cache 近似，Batch 不保存 cache filter/control 字段。
一个 Snapshot Batch 含多个 temporal unit 时，runtime scan 仍按各自 predecessor
满足 exact dependency；stale-cache 则显式读取最近收到的 detached 副本，不能把
Batch 开头的一次 read 当作全部历史的 exact 状态。

## 当前未对齐

- 薄通信上下文仍有部分路径通过 `GraphBlock.cache` 取得；目标是由 runtime scan
  直接持有，而不是改放到 `Batch`。
