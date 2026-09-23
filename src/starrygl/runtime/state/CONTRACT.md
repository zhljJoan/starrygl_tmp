# 状态读取与提交契约

状态：复用现有 StateManager/historical storage，但只保留 exact 和选择性刷新的
stale-cache 近似；不新增平行 state stack，也不声明 `bounded(K)`。

## Coupled Snapshot 的当前实现覆盖

2026-09-11 用户确认的滑动窗口槽位缓存替代下文旧的 last-boundary-only 草案：
只分配 W 个输出槽和一个前驱槽，W 为实际输入窗口长度，槽内版本用于因果和
时间差计算。窗口前进时复用输出槽，前驱保留过滤后的旧观察及累计统计。
本地计算历史与接收的 shared-hot 历史分开保存，consumer 只查询合法 predecessor。
cold 输入使用 cache + age * cumulative increment；可学习 sigmoid(gamma) 用于
混合各槽 local UPDATE 和 shared prediction，无共享观察时保留 local UPDATE。
batch 结束仅发布最终 boundary；启用 smoothing 的 hot 走既有过滤 all_gather，
cold 走一次绑定的 owner push Route。GC/DC 无 smoothing 时统一省去无消费者的
hot 发布。近似路径不在 cache miss 时逐快照拉取 owner；零初始版本是显式初始条件。
exact W>1 保留因果前驱，并在每个内层快照交换 live previous state；DCRNN 的
reset gate 交换仍然存在。Event exact/memory/mailbox 契约不变。
见 `docs/design/current/snapshot_boundary_history.md` 与 `store/snapshot_history.py`。

## 状态种类、owner 和生命周期

| kind | 权威位置 | 默认生命周期 | cache / compensation |
| --- | --- | --- | --- |
| `node_memory` | `node_master` | Event 跨 Batch | exact 或 shared-hot stale-cache；可补偿 |
| `mailbox` | `node_master` | Event 跨 Batch | 与 memory 同一原子 read/refresh group；不单独降级 |
| `node_recurrent` | `node_master` | Snapshot Batch 内；chunk boundary 或非重叠可 carry | local/exact 或 detached boundary seed |
| `neighbor_recurrent` | `node_master` | 当前 Snapshot scan；最后 Snapshot 可 boundary carry | exact 或 shared-hot stale-cache；可补偿 |
| `model_recurrent` | 每个模型 rank 复制，rank 0 checkpoint | Snapshot 按时间 | exact；不使用 node cache/补偿 |

是否 coupled 取决于当前 GCN 是否读取前一 temporal unit 产生的状态，不取决于
模型中有没有 RNN。`node_memory + mailbox` 是一个 dependency group：同一次 read
必须来自同一原子 payload、fallback 和 commit 操作，不能把 stale memory 与 exact
mailbox 混在一起。

重叠 Snapshot 的 live recurrent tensor 不跨 Batch 保留，但 paper-v1 的
`chunk_decay` 可以在 Batch 完成 backward 后，将最后 Snapshot 的 embedding/state
detach，作为唯一 window-boundary delta 交给现有 `StateManager`/cache-refresh
链路。下一 Batch 只把它当作 window-entry/缺失 chunk 的 boundary seed；当前 Batch
产生的其他中间 state 不提交。

以 `[1,2,3] -> [2,3,4]` 为例，第二个 Batch 可以从 `E3` 开始并重算 2、3；这会
改变严格的 per-Snapshot recurrence，但只要 loss/metric/对外输出仅属于最后的 4，
且 carry producer time `< 4`，就没有 target-time future leakage。若 2 或 3 也受
监督、对外输出或提交状态，compile 必须禁用该 carry。`[1,2,3] -> [4,5,6]` 中的
`E3` 则是 exact predecessor。Event memory/mailbox 始终按 Batch 持久化。

## Consistency、cache 和 wait

三个设置保持独立：

```text
cache         none / local / shared_hot
consistency   exact / stale_cache
wait          block / reschedule
```

`stale_cache` 是显式近似，只表示允许读取最近一次收到的 detached cache value；
它不检查 generation 或 producer temporal-unit age，也不承诺最大陈旧窗口数。
`max_skip` 只限制 producer 的 refresh filter 可以连续丢弃多少次候选更新。
Chunk-decay boundary carry 复用同一 detached state Route/update 路径，不增加 consistency
枚举；它由 `chunk_decay + latest-target-only` 在 compile 时静态推导。

`exact` read 先等待 `committed_through >= required_predecessor`，再从 owner 或其
已确认的 exact local copy 读取。`stale_cache` 在命中时直接读取；cache miss 在
首版发起 exact owner fetch 并 `block`。`reschedule` 尚未实现，不能让 rank 独立
跳过 collective。

一个 Snapshot Batch 含多个 temporal unit 时，exact `neighbor_recurrent` 必须在
每个 consumer GCN 前取得对应 predecessor 的 live state；不能只等待“上一个
Batch commit”，也不能把 Batch 内 live tensor detach 后经 StateManager 绕一圈。

v1 的 refresh 只能从 StateManager 的固定提交路径执行；后台线程不能绕过该路径
自行发起 free-form 或乱序 refresh。
生命周期不能把 split、epoch、reset 和 checkpoint 混为同一种边界：

- split boundary：drain 当前边界要求完成的 collective 与 pending commit，只切换
  target/loss 范围；保留 owner temporal state、`model_recurrent`、stale cache、
  `committed_through`、refresh skip counter 和最后 Snapshot boundary carry。同一
  epoch 跨 split 不 reset。
- epoch boundary：先 drain，再调用 temporal `reset()`；owner state 和
  `model_recurrent` 恢复 model-defined initial value，清空 stale cache/skip counter，
  watermark 恢复初始值，并清空 boundary carry。只重置时间运行态，不重置模型参数
  或 optimizer。
- 显式 `reset()`：语义与 epoch reset 相同，会切断当前时间线；runtime 不能在 split
  切换时隐式调用它。
- checkpoint save：先 drain，保存 model/optimizer、epoch、next temporal-unit cursor
  和可恢复的 owner state、watermark、cache/filter/boundary-carry state；保存本身
  不 reset。
- checkpoint load：同一 cursor 续跑时恢复整组运行态，不 reset；只加载权重并开始
  新 epoch 时按 epoch 规则 reset。缺少运行态的 checkpoint 不能从 epoch 中间直接
  续跑，必须从 epoch 起点 replay 或使用完整边界 checkpoint。
- 异常退出：停止新通信并 drain/cancel 已提交工作后销毁运行态，不把退出清理定义
  成一次可继续执行的 reset。

这个固定生命周期顺序和完整边界恢复替代 generation/version 分支；未来若允许乱序消息，
必须重新引入版本协议后才能开放。

## Historical cache 更新

`node_master` 保存权威值，local/shared-hot 只是 detached read replica。更新为：

```text
模型产生 StateDelta
  -> runtime 按 canonical node/time 合并重复写
  -> owner commit 并推进 committed_through
  -> vectorized change filter
  -> 变化足够大或 skip_count 达到 max_skip 时发送
  -> StateManager 按提交顺序更新 cache
```

`max_skip` 是连续跳过 refresh 的次数上限，不是 freshness bound。它不使缓存成为
exact，也不能证明“最多旧 K 个 temporal unit”。只要一次 refresh 需要 collective，
每个 rank 都参加并让空更新发送空 tensor；后台线程不能自行发 free-form collective。

可选补偿只能使用已缓存的 state/history tensor 和模型定义的增量估计。它不能用
不存在的 producer-age 保证包装成有界一致性。`model_recurrent` 不使用该补偿。
Filter、historical value、skip counter 和 Route tensor 与被过滤 state 位于同一
GPU；热路径不能用 CUDA row 索引 CPU cache，也不能 `.cpu()`/`.item()` 触发同步。

## 状态提交

```text
model.state_update(batch, output)
  -> runtime 检查 kind、node/model state 和 owner
  -> 同一 node 的重复 delta 按 canonical (event_ts, edge_row) 或模型规定 reduction
     用 Torch/native 合并
  -> 对重叠 chunk-decay，只选择最后 Snapshot 的 detached boundary delta
  -> StateManager 通过共享通信上下文发起 owner commit/cache refresh
  -> 保留 detached pending commit，与后续计算重叠
```

下一个 exact consumer 必须等待所需 watermark。Pending buffer 在对应 CUDA event
完成前不得复用；epoch/split/reset/checkpoint/异常退出边界必须按上述规则 drain
权威 commit，但只有 epoch 或显式 `reset()` 清空运行态。Cache refresh 永远不能
代替 owner commit 或推进 exact watermark。

`model_recurrent` 不按 node 分片。Runtime 在固定 context-reduction 调用点生成所有
rank 一致的 model context，各 replica 按同一 temporal order 更新；只有 rank 0 写
checkpoint，载入后广播同一 value。

### Event 状态通信的物理边界

Event、Snapshot、node 和 edge 执行共享的最大链路仍是：prepared row -> Batch ->
dependency read -> model/task -> `StateDelta` -> runtime commit。不可避免的专化只在
state kind 的 payload 形状和本地 apply：Event 的 `node_memory + mailbox` 必须原子
读取并分别应用，Snapshot recurrent state 没有 mailbox。owner 路由、collective
顺序和 detached commit 都继续复用 `StateManager`/`CommScheduler`。

2026-09-23 的 WIKI/TGN trace 表明差距在 state collective 次数，而不是 model
kernel。尝试用 Torch `cat` 将 owner push 的同 dtype 字段分桶；两 rank 正确性通过，
但四卡 train-only 中位数从 0.6481 s 退化到 0.6941 s，因此撤回。DGL 没有适合这种
变长 owner payload 的算子；自定义 C++/CUDA kernel 不引入。后续只能减少不必要的
协议阶段或复用 Prepare 的静态 Route，不能靠复制 buffer 来换 collective 数量。

随后测试的原子 owner commit 用一个 union node route 携带 memory、memory
timestamp、mail 和 mail timestamp，并用 presence mask 覆盖不对齐 node 集。它
消除了第二次动态 route/count 和重复 node payload，但构造 474 维 packet 的成本抵消
了 collective 减少：四卡 train-only 中位数 0.6761 s，未优于 0.6699 s 基线，故
撤回。Snapshot/recurrent 无 mailbox，始终继续走原 StateManager 路径。

## 与 layerwise 的最小统一

Layerwise embedding 和 historical state 只共用现有的 `Route`、
`launch -> wait_on_stream -> scatter` 和 profile counter：

- layer embedding：exact、cache=none、带 autograd，仅当前 forward/backward 存活；
- temporal state：exact 或 stale-cache，可 local/shared_hot、detached，并按
  `max_skip` 选择性刷新。

不创建通用 CacheManager、provider hierarchy、第二个 Batch 或 model-local state
manager。

## 当前未对齐

- 当前配置和 plan 仍使用 `bounded_stale/max_staleness` 名称；目标应收敛为
  `stale_cache` 和 `filter.max_skip`，不能继续宣称有界 freshness。
- split drain-and-preserve、epoch reset、checkpoint 完整恢复和异常退出尚未形成
  生命周期测试矩阵。
- 一些 state/hydration correctness 字段仍藏在 Batch metadata 和多层 helper 中。
- GPU filter/historical buffer 仍可能建在 CPU，Stage B 也没有 stream/event 契约。
- `reschedule` 尚未实现，首版只能安全声明 `block`。
- Coupled Snapshot 缺少 Batch 内逐 temporal-unit exact/stale-cache 多 rank 对照；
  memory/mailbox 原子 group 和 EvolveGCN replicated model state 也缺少测试。
