# Snapshot-CSC 执行契约

状态：Snapshot-CSC 是公共流程中的一个 graph accessor；layerwise 和 coupled
顺序由 Stage A 的唯一 runtime scan 执行，不是独立运行时。

## 一个 Snapshot slice 包含什么

```text
src_nodes, dst_nodes
indptr, indices
edge_ids, edge_rows, ts
dst chunk ptr、归一化系数、源节点放置位置
完整 Snapshot 的预计算 Route
可选的时间版本节点特征
```

`dst_nodes` 由本 rank 的 `node_master` 负责。`src_nodes` 先放本地目标节点，再
追加远程边界源节点，Route 的 recv placement 与该顺序一致。大字段使用 packed
`data + ptr`；任务标签始终从公共任务表读取。

## graph accessor

目标窗口 `k` 选择历史 Snapshot，并返回：

```text
(blocks, node_ids, edge_rows)
```

每个历史 Snapshot 只恢复一个 CSC，模型按 `num_layers` 重用，不能为每层复制
图。`[K,N,F]` 特征按 `(snapshot_id,node_id)` 读取；静态特征才允许跨 history
按 node id 合并。split 只决定 target，不截断历史图。

## Full 和 chunk decay

`full_snapshot` 使用完整 CSC 和预计算 Route。Paper-v1 的 `chunk_decay` 直接
复用被评测的 Flare 物理语义：

1. 每个 epoch、每个 rank 生成一次可复现 chunk permutation；本 epoch 的所有
   Batch 共用它。
2. 同一 Batch 的旧 Snapshot 使用该 permutation 的 nested prefix，最近 `F`
   个 Snapshot 保持完整。
3. prefix Snapshot 是 local induced CSC，显式使用空 Route；最近完整 Snapshot
   才执行跨 rank boundary exchange。prefix 在所有 rank 都是 local induced 图时
   不发起 collective。

这里的 boundary exchange 是同一 Batch 内的 layerwise 通信。跨 Batch 不保留这些
Snapshot 的全部中间 embedding；只从现有 state commit/cache-refresh 路径发送
最后一个 Snapshot 的 detached embedding/state，作为下一 Batch 的 boundary carry。

这避免在训练 step 热路径重建 peer Route，也与当前 Flare reference data flow
一致。它是明确的近似，不得伪装成 exact full graph。若未来要让 decayed prefix
也通信，Prepare 必须先提供按 `(dst_rank,dst_chunk)` 可对称切片的 Route；不能
只在 receiver 本地裁剪。

Chunk 选择、CSC prefix 和空/完整 Route 都必须由 prepared ptr 加 tensor slice
完成，不能扫描、重排图或构建 Python blob list。

## 初始输入和 layerwise 调度

Stage B 精确 materialize 每个 Snapshot 的初始 `H^0`，包括完整 Snapshot 所需的
远程 boundary feature。之后 Stage A 的 runtime-owned Snapshot scan 对每层执行：

```text
按 recent -> old 的全局固定 history 顺序
  -> 对 ready Snapshot 做本层 local message passing
  -> 非最终层：提交本层输出的 boundary exchange，供下一层使用
  -> await 时运行其他 ready Snapshot
  -> 所有 Snapshot 完成本层后进入下一层
最终层后按 oldest -> recent 执行 temporal update
```

history/layer 顺序由同一个 runtime scan 固定，不再生成 collective slot 表。
某个 history/layer 的 prepared Route 需要 collective 时，所有 rank 都进入该调用点，
无本地 Route 行的 rank 使用空 payload；全局均为 local prefix 时直接跳过通信。
Autograd reverse 由前向 Route 的 autograd graph 触发，不能根据本 rank 的 Route
大小重排或跳过 peer 仍需要的 collective。

公共模型接口仍是 `StarryModel.encode(Batch)`。StarryGL 提供的共享 Snapshot
图算子在训练开始前由 runtime 绑定，它拥有 layer loop、Route、Await 和薄通信
上下文；模型层只提供 tensor math。通信上下文/handle 不进入 `Batch`、
`GraphBlock.cache`，用户模型不能自行调用 distributed collective。

## Coupled 顺序

Coupled 模型按 temporal unit 正序执行：

- `neighbor_recurrent + exact` 在每个 GCN 前等待 producer temporal unit 的 owner
  watermark，并从固定 state dependency 调用点取得 remote neighbor state；不能只在整个
  Batch 开头读取一次。
- `neighbor_recurrent + stale_cache` 可以读取最近收到的 detached historical
  value；它不提供最大陈旧 temporal-unit 数保证。cache miss 在首版回 owner 并
  block，refresh filter 连续跳过达到 `max_skip` 后强制发送。
- EvolveGCN 的 `model_recurrent` 在各 rank 复制。runtime 先在固定 reduction 调用点
  得到一致 context，再按时间更新 weight；它不使用 node historical compensation。

重叠 Snapshot 训练的 live recurrent tensor 仍只在当前 Batch 内保留，但
`chunk_decay` 允许一个受限的跨 Batch boundary carry。以
`[1,2,3] -> [2,3,4]` 为例：

1. 第一个 Batch 完成 backward 后，仅 detach、保存并广播最后 Snapshot 的 `E3`；
2. 第二个 Batch 用 `E3` 初始化 window entry，或补齐 decayed prefix 中没有计算的
   node/chunk state，再重算 2、3 并计算 4；
3. 只对最后 Snapshot 4 计算 loss/metric/对外输出；2、3 的重复计算和内部非严格
   因果中间值属于 chunk-decay 近似，不单独提交；
4. 完成后只用 `E4` 替换 boundary carry，不广播当前 Batch 的其他中间 embedding。

只要 carry 的 producer time 小于最终 target 4，就没有 target-time future leakage。
这不等价于 exact per-Snapshot recurrence，plan 和 benchmark 必须明确标记。如果
任务监督或暴露中间 Snapshot，compile 必须拒绝该 carry。对
`[1,2,3] -> [4,5,6]`，`E3` 是 Snapshot 4 的直接 predecessor，可以 exact carry。

这条规则与 split 边界正交：split 只改变 target/loss，不截断图历史，也不触发
temporal state reset。同一 epoch 的 train、validation、test 沿时间线继续；进入
新 epoch 重放时间线前才 drain 并 reset temporal state、stale cache、skip counter
和 watermark，模型参数与 optimizer 保留，最后 Snapshot 的 boundary carry 也只在
epoch reset 时清空。checkpoint save 只 drain 并保存完整 resume state（包括该
carry），不 reset；从同一 cursor load 时恢复并继续，从新 epoch load 时执行 epoch
reset。缺少 runtime state 时不能从 epoch 中间直接恢复。

这些执行顺序仍在同一个 Stage A 内，不能创建第二套 loader、queue、target
constructor 或 outer training loop。

## 与 Snapshot neighbor 的关系

Snapshot neighbor 使用 T-CSR，不读取 Snapshot-CSC。两者只在 graph accessor
内部不同，返回后立即进入同一个 Stage B。不存在 `chunk_decay + neighbor`。

## 当前未对齐

- materialize 仍使用较多 row/blob 包装和 Python cache 管理。
- 当前 plan 为 Snapshot neighbor 同时要求 T-CSR 与 Snapshot-CSC。
- layerwise scheduler 仍可按本地 Route 到达顺序 launch，scheduler 还藏在 block
  cache/模型调用链中。
- Chunk prefix 的本地图、空 Route 和 epoch-static permutation 尚未有与 Flare 的
  output/gradient/throughput parity test；paper 正文“每 training iteration 重排”
  也必须在发布前改成这里被评测的 epoch cadence，或重新跑指标。
- Coupled exact 尚未覆盖 Batch 内逐 temporal-unit remote state dependency；
  stale-cache 近似和 EvolveGCN 也缺少多 rank 对照测试。
