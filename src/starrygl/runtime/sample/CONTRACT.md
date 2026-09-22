# T-CSR 采样契约

状态：本模块只绑定一次 native sampler，并把紧凑输出转成公共图输入；不再有
Event/Snapshot 两套入口。

## 初始化

任务开始时，用 T-CSR、packed partition index、fanout、层数、基础采样策略和
可选 boundary retention 参数创建一次 sampler。Event 与 Snapshot neighbor
共用：

```text
sample_neighbors(roots, scope) -> (blocks, node_ids, edge_rows)
```

首版 sampler 在 CPU/native 执行。负样本在同一设备向量化生成，native 调用
释放 GIL，输出紧凑 pinned tensor；不能先在 GPU 生成 roots 再同步回 CPU。
GraphBlock 默认保留 sampler 返回的边重数，和 MemShare block 语义一致；静态
edge feature read 仍按物理行去重。只有明确接受改变消息重数时才设置
`deduplicate_edges=True`。

## Event 的因果 root

Event 采样行的身份是 `(node_id, cutoff_ts)`，不是只有 `node_id`：

- 正边的 cutoff 唯一由 canonical `ts[edge_rows]` 得到；
- `neg_src`/`neg_dst` 继承对应正边的 cutoff；
- 节点任务使用该全局窗口的结束边界；
- memory/mailbox 需要更新的 Event roots 与 task roots 取并集。

同一节点在两个 cutoff 下必须保留为两个 MFG row，不能在采样前按 node id
合并。静态 feature read 可以在采样后按 node id 进一步去重，再由 native scatter
rows 放回所有时序行。下一层 root 使用刚采到边的时间作为 cutoff，保持因果性。

## Snapshot scope 和时间版本

Snapshot 的 `scope` 是最近 `H` 个 Snapshot 的 canonical edge-row 区间，形状
为 `[H, 2]`。`H` 包含当前 Snapshot；历史不按 split 起点截断。一次 native
调用完成所有 history 和 GNN layer，Python 不能逐 history/layer 调 sampler。

每层使用同一组 H 个区间，并在每个区间内独立 uniform 采样。若节点特征形状
是 `[K,N,F]`，read identity 是 `(snapshot_id, node_id)`；同一 node 在不同
Snapshot 不能合并。Native 保留 `node_ptr`/history id，Stage B 按 history 调用
同一个 FeatureManager。静态特征可由 provider 再跨 history 合并。

Snapshot neighbor 不与 `chunk_decay` 组合。

## Paper 的 boundary retention

Event 在基础 neighbor sample 后、Route 生成前，可在同一次 native 调用中只对
remote neighbor 做 retention；local neighbor 保留：

```text
uniform:   P_keep = theta
temporal:  P_keep = theta * exp(-(cutoff_ts - edge_ts) / tau)
```

`theta`、`tau` 和 RNG seed 在初始化时绑定，不增加另一个 sampling-policy
框架。Paper scalability 路径固定 `theta=0.1`：TGAT/GDELT 使用 uniform，
TGN/GDELT 使用 temporal，`tau` 是该 target 最近候选邻居时间差的 native 均值。
过滤、compact、read grouping 和 scatter 必须全部在 Torch/native 中完成。

## Native 输出的正式字段

```text
blocks[history][layer]  每层 MFG，保留 node/cutoff 与 edge_rows
node_ids / node_ptr     feature/state read table 及 history 分段
edge_rows               去重后的 canonical 物理边行
root_rows               正负端点在输出节点表中的位置
read groups             owner-local rows、counts 和 peer ptr
scatter rows            read tensor 放回 block/target 的位置
```

这些 correctness 字段进入 `GraphBlock.srcdata/dstdata/edata` 或 task route 的正式
tensor，不能藏入 `cache`；`Batch` 不提供 `meta`。Stage B 直接消费 native read/scatter
结果，不再做第二次 argsort、search 或 Python 分组。

## Sample-dependent Route

`node_dist_index[node_ids]` 和 `edge_dist_index[edge_rows]` 只能生成 requester 侧
分组；owner 还不知道 peer 请求了哪些行。Stage B 固定执行：

```text
owner 分组
  -> all_to_all(counts)
  -> all_to_all(packed owner-local request rows)
  -> owner 一次 index_select
  -> all_to_all(payload)
  -> native scatter rows
```

counts、request 和 payload 按固定函数顺序执行。没有请求但 peer 仍有请求的 rank
发送空 tensor。Snapshot 预计算 Route 可省去 request 交换，但最终 payload 和
scatter 仍复用同一组 Route 通信实现。

## 当前未对齐

- 运行时仍从 `src/dst` 重建并再次排序 T-CSR。
- 物理边行仍使用 `edge_feature_ids` 并藏在 block cache。
- Snapshot 仍在 Python 逐 history 调 sampler，native 还不能直接接收 `[H,2]`
  edge-row ranges。
- Event 输出尚未把 `(node_id, cutoff_ts)` 固定为 MFG row identity。
- Boundary temporal retention 的 `tau` 与 paper 定义尚未对齐，Route 也未保证在
  retention 后生成。
- 动态 Route 尚未用两 rank 空 request 测试覆盖 counts/request/payload 全协议。
