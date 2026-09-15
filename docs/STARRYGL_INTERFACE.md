# StarryGL 接口总览

状态：Paper-v1 统一接口和当前实现基线。下文标记为**已确认**的决定和公共入口
已经固定；短 tuple 与双队列已落地，完整 native Route 和论文规模
性能仍在实现。本文件不声明当前性能已经与论文一致。

各模块相邻的 `CONTRACT.md` 仍是权威实现契约。本文件只负责提供便于阅读的摘要
和审查清单，不是第二份实现规范。

## 参考顺序

1. [包与计划契约索引](../src/starrygl/CONTRACT.md)
2. [Paper 方法覆盖与发布门槛](../src/starrygl/PAPER_METHOD_COVERAGE.md)
3. 本文末尾链接的各模块契约
4. [Paper 方法原文](../../paper_method/method.tex)
5. [迁移状态日志](migration/status_log.md)

MemShare 和 FlareDTDG 是物理实现参考。可以移植它们的算法、tensor 布局、
native kernel 和 stream 调度，但其 wrapper、模型 registry、loader 和平行 runtime
不是 StarryGL 接口。

## 公共入口

```python
import starrygl as sg

trainer = sg.compile(graph=graph, model=model, task=task, spec=spec)
print(trainer.plan.explain())
```

`compile` 把 graph/model/task 语义降低成可观测的 `ExecutionPlan`。Event、
Snapshot、sampled、full-graph、coupled 和 decoupled 执行共用一个 runtime。
差异只体现为绑定的图访问、依赖位置和 `compile` 生成的静态算子顺序，不能选择
第二套 runtime。

当前 `ExecutionPlan.execution_order` 是高层语义 lowering，只描述图访问、依赖、
model/task/state 的相对顺序；`plan.explain()` 会明确标为
`semantic_lowering`。它不承诺 backward、optimizer 或逐 collective 的精确指令，
也不再生成另一份 `CommPlan`/slot 表。具体通信顺序由固定运行路径负责。

生命周期为：

```text
编译语义计划
  -> Prepare 生成 PartitionPlan 和所需视图
  -> 一次绑定 store、graph accessor、Route、薄通信上下文和 CUDA stream
  -> 全局 window row -> task slice -> 可选 negatives -> graph accessor
  -> request_queue -> Stage B materialize/H2D/prefetch -> ready_queue
  -> Stage A exact dependency/model/task/backward/state handling
```

## 已确认决定

- 只保留一个公共 `StarryModel.encode(Batch)` 和可选的
  `state_update(Batch, ModelOutput)` 接口。
- 复用算子与依赖接口，由 `compile` 改变静态执行顺序；不能按模型家族切换
  runtime。
- 只保留一条 Stage C/B/A 主链、两个深度为 1 的队列、一个
  `Batch` 和一个 `GraphBlock`。
- Event、Snapshot、node 和 edge 执行可以绑定不同 graph accessor 或 task
  service，但不能各自拥有 loader、target constructor、queue 或 training loop。
- Prepare 按 owner 写入一份 `task_ptr/payload`。Event、Snapshot-neighbor 和
  Snapshot-CSC 都先切同一任务行、再访问图、最后补同一 `TargetRoute`；DataLoader
  不接收 task 名称。Snapshot 的下一步目标和 horizon 尾行由任务表表达，不能跨
  train/validation/test 边界取下一行。
- Snapshot chunk permutation 是 **per epoch**：每个 rank 在 epoch 开始时生成一次
  可复现 permutation，该 epoch 内所有 Batch 共用它。旧 Snapshot 使用 nested
  prefix，最近 `F` 个 Snapshot 保持完整；只有进入新 epoch 才重新生成。
- Snapshot-CSC、nested chunk prefix、layerwise 调度和 EvolveGCN 算子顺序优先
  参考 FlareDTDG。
- Event T-CSR 采样、memory/mailbox 访问、historical cache、动态通信分组、pinned
  buffer 和 stream overlap 优先参考 MemShare。
- 状态读取只保留 `exact` 和显式的 `stale_cache` 近似；不提供独立的
  `bounded(K)` 保证，也不按 generation 或 producer temporal-unit age 验证缓存。
  `max_skip` 只限制 refresh filter 可以连续跳过多少次候选更新。
- 扩展现有 native sampler，使其直接接收 native `[H,2]` Snapshot history range，
  并返回动态 Route/read/scatter tensor。Python 热路径不能逐 history、layer、node、
  edge 或 peer 循环。
- DTDG coupled 模型必须通过可复用算子和依赖接入同一执行主链；模型不能调用
  私有 coupled scan。
- layer embedding 与 historical state 只共用 Route、CUDA
  event/await、scatter 和 profiling，不共用通用 cache 类。
- 不新增 registry、adapter layer、provider hierarchy、Stage wrapper、第二套
  runtime 或通用 `CacheManager`。

## 模型侧接口

```python
class StarryModel(nn.Module):
    def encode(self, batch: Batch) -> ModelOutput:
        ...

    def state_update(
        self,
        batch: Batch,
        output: ModelOutput,
    ) -> StateDelta | None:
        return None
```

模型只执行 tensor 数学，不访问 `StateManager`、historical cache、owner
watermark、通信 handle、`CommScheduler` 或 `torch.distributed`。Runtime 负责验证
和提交模型返回的 `StateDelta`。

`Batch` 不传 rank、epoch、step、split 等 loader 控制信息。模型数学只读取
`blocks`、`graph`、`features`、`state` 和 `targets`。canonical `edge_rows`、Event
cutoff identity、history id、read groups 和 scatter rows 等正确性字段必须是正式
tensor，不能藏在 `GraphBlock.cache` 中。

## 内部可复用算子接口

当前建议的最小内部接口复用已有 `runtime_cell` 边界。它不是另一个公共模型 API，
也不是通用 DAG：

```python
class TemporalRuntimeCell(Protocol):
    state_kind: str
    num_spatial_layers: int

    def spatial(
        self,
        layer: int,
        block: GraphBlock,
        x: Tensor,
        dependency: Tensor | None,
    ) -> Tensor:
        ...

    def advance_state(
        self,
        previous_state: Tensor,
        update_input: Tensor,
    ) -> Tensor:
        ...
```

只有确实需要图上下文的 `model_recurrent` cell 额外提供：

```python
context_reduction: str  # "sum_count" 或 "max"

def context(
    self,
    block: GraphBlock,
    x: Tensor,
) -> tuple[Tensor, Tensor | None]:
    ...
```

不需要 context 的模型不增加 identity `context` wrapper。Runtime 执行声明的
reduction，模型不能自行发起通信。

模型通过现有边界把 runtime 结果转换成公共输出：

```python
def runtime_output_from_scan(
    self,
    batch: Batch,
    scan: WindowScanResult,
) -> ModelOutput:
    ...
```

当前共享结果保存最终 embedding、状态、图块和各 temporal unit 输出：

```python
@dataclass(frozen=True)
class WindowScanResult:
    embeddings: Tensor
    state_embeddings: Tensor
    final_block: GraphBlock
    window_embeddings: tuple[Tensor, ...]
```

GConvGRU 已通过 `neighbor_recurrent` cell 进入 runtime-owned coupled scan；
EvolveGCN 已通过 `model_recurrent + context()` 进入 runtime-owned context reduction
和权重 scan。`reads_neighbor_state` 当前仍是 runtime scan 的内部分派提示，后续由
compile 绑定的 operator schedule 替代；它不选择第二套 runtime。

## Compile 生成的静态顺序

| State kind | 代表模型 | Tensor 依赖 | 编译顺序 |
| --- | --- | --- | --- |
| `node_recurrent` | T-GCN、MPNN-LSTM | 前一 same-node state 只进入时间更新 | 对可并行 Snapshot 执行 layerwise spatial，然后按时间顺序执行 `advance_state` |
| `neighbor_recurrent` | GConvGRU、DCRNN | 当前空间聚合读取前一 neighbor state | 每个 temporal unit 按时间执行 dependency Route -> spatial layers -> `advance_state` |
| `model_recurrent` | EvolveGCN | 当前空间算子消费演化后的模型权重 | local context -> global reduction -> 按时间演化权重，再执行 Flare 风格的 layerwise spatial |

这里改变的是 plan 的静态算子顺序，不是 runtime 对象。不能用模型名称推断
state kind 或选择 executor。

Event 模型也走同一外层主链。TGAT 没有持久时间状态；JODIE/TGN/APAN 在实际
consumer operator 前声明 memory/mailbox 依赖，并通过 `state_update` 返回 delta。

## Coupled DTDG 状态语义

对 exact `neighbor_recurrent`，temporal unit 消费所需 predecessor 产生的 live
state。远程行使用固定、支持 autograd 的 Route。这个 live tensor 不能在同一
Batch 内先 detach、提交到 `StateManager`，再重新 fetch。

对当前 `bounded_stale`（语义为选择性刷新的 stale-cache）coupled Snapshot：

- Prepare 将 owner 节点与 hot replica 纳入本地计算集合，loss 和权威提交仍属于
  node_master。
- detached cache 按滑动窗口分配 W 个输出槽和一个前驱槽；W 包含完整快照和
  chunk-decay 历史前缀。窗口前进时复用旧槽，前驱槽保留过滤后仍可用的旧状态。
  实际版本只作为元数据：快照 s 输出的版本是 s+1，快照 t 只读版本 <= t，
  并优先使用本 Batch 刚算出的上一快照输出。
- 本地计算历史和接收的 shared-hot 历史分开保存。缺少本地前驱的 cold 输入使用
  `cached + (t - cached_version) * cumulative_mean_increment`，不乘 gamma。
  启用 hot smoothing 时，各槽输出使用
  `sigmoid(gamma) * local_update + (1 - sigmoid(gamma)) * shared_prediction`；
  shared 槽尚无观察时保留 local_update。累计增量及其计数按槽保存，重算覆盖原槽。
- 每个 Batch 的中间实算状态在本地保存；只发布最终 boundary。hot 候选继续
  使用原来的 cosine/norm filter 和 max_skip，通过 all_gather 发布；冷 boundary
  由 owner 沿预先建立的 all_to_all subscriber Route 推送。GC/DC 未启用 hot
  smoothing 时，接收的 hot 值没有消费者，初始化时统一省去该 hot collective。
- cache 缺槽时使用更早的合法槽（包括零初始状态）进行近似，无逐快照 hidden-state
  owner fetch。max_skip 只限制跳过发布次数，不保证 snapshot age 上限。
- DCRNN 当前 reset gate 仍逐快照交换并等待；本策略不宣称该依赖已隐藏。

例如 `[1,2,3] -> [2,3,4]`，第二个 Batch 计算快照 2 时读取输出 E1；当前 Batch
重新算出的 E2 供快照 3 使用。缺失节点按各自快照的 predecessor 时间查缓存和补偿，
不把 E3 作为快照 2 的初始状态。这替代此前 coupled approximate
last-boundary-only carry 草案；远程 temporal gradient 仍被 detached cache 截断。
细节与验证见 [snapshot boundary history](design/current/snapshot_boundary_history.md)。
2026-09-12 已将验证过的 V3 合入主包，见
[integration record](design/current/rebuttal_v3_integration.md)。GC/DC 的完整滑窗
复用静态图和 DGL layout，保留独立的 Batch 特征/状态；exact 仍逐快照交换依赖。
训练时全 rank 均无有效监督的窗口保留 backward/state commit，但不推进 Adam。

Event memory/mailbox 仍跨 Batch 持久化。

Runtime 只保留 exact 路径和 refresh 所需的控制字段：

```text
committed_through
refresh_skip_count
```

`max_skip` 是连续跳过 refresh 的上限，不是状态最多陈旧多少 temporal unit 的
承诺。边界生命周期固定为：

- split 只切换监督范围。切换前 drain 当前边界要求完成的通信和 state commit，
  但不重置 temporal state、stale cache、`committed_through` 或 skip counter；同一
  epoch 的 train -> validation -> test 按同一时间线继续，并保留最后一个 Snapshot
  的 boundary carry。
- 新 epoch 会从时间线起点重放。进入前先 drain，再调用 temporal `reset()`：恢复
  model-defined initial state，清空 stale cache 和 skip counter，并把 watermark
  恢复到初始值；模型参数和 optimizer 不重置。
- checkpoint save 先 drain，再保存模型/optimizer、epoch 与 next temporal-unit
  cursor，以及恢复该位置所需的 owner state、watermark、cache 和 filter state；
  保存动作本身不 reset。
- checkpoint load 若从保存位置续跑，就恢复整组状态且不 reset；若只加载权重并
  开始新 epoch，则按 epoch 规则 reset。缺少 runtime state 的 checkpoint 不能直接
  从 epoch 中间继续，只能从 epoch 起点重放或恢复一个完整边界 checkpoint。

v1 的所有 refresh 只从 StateManager 的固定提交路径应用，因此不增加
generation/version 分支或后台 free-form 通信。
Event timestamp 仍只是模型数据。

## EvolveGCN 语义

EvolveGCN 的 state kind 是 `model_recurrent`。Runtime 编译：

```text
local graph context
  -> runtime 固定的 global reduction 调用
  -> W_t = advance_state(W_{t-1}, context_t)
  -> spatial(A_t, x_t, W_t)
```

实现应复用 FlareDTDG 的 MatGRU、context、按时间扫描权重和 layerwise GCN 数据流。
StarryGL 额外通过小型 exact context reduction 保持 model-state replica 一致；
不同 rank 不能用不同的 local context 私自更新同一个逻辑权重。
`model_recurrent` 不使用 node historical compensation。

`model_recurrent` 不使用上述 node embedding boundary carry。它从 plan 声明的
window-entry recurrent weight 开始；non-overlap chronological execution 可以
exact carry。split 不触发 reset，新 epoch 才恢复模型初始 recurrent weight。
rank 0 负责 `model_recurrent` checkpoint 输出；从同一 cursor 恢复时向所有 rank
广播保存值。

## 图访问与资源控制

graph accessor 只绑定一次，并返回：

```text
(window_id, targets, blocks, node_ids, edge_rows)
```

Event 与 Snapshot-neighbor 执行共用一个 native 入口：

```text
sample_neighbors(roots, scope) -> (blocks, node_ids, edge_rows)
```

- Event row identity 是 `(node_id, cutoff_ts)`。
- Snapshot-neighbor 的 `scope` 是形状为 `[H,2]` 的 canonical edge-row range。
- 一次 native 调用处理全部 history 和 layer。
- Snapshot-neighbor 使用 T-CSR，不能同时要求 Snapshot-CSC。
- 不支持 `chunk_decay + neighbor` 组合。

Full Snapshot 选择 prepared Snapshot-CSC。每个 temporal unit 只 materialize 一个
CSC，由所有 spatial layer 复用。decayed history 使用 epoch-static nested chunk
prefix、local induced CSC 和空 Route；最近的 full history 使用 prepared boundary
Route。decayed prefix 在所有 rank 都是本地执行时不发起 collective；recent full
history 需要 boundary Route 时，无本地行的 rank 仍以空 payload 参加。

Event boundary decay 在基础 neighbor sampling 之后、Route 构造之前执行：

```text
remote uniform:  P_keep = theta
remote temporal: P_keep = theta * exp(-(cutoff_ts - edge_ts) / tau)
local neighbor:  保留基础采样结果
```

filter、compact、grouping 和 scatter 必须保持 native/vectorized。

## 通信与流水线

sample-dependent remote access 使用固定函数顺序：

```text
按 owner 对 requester row 分组
  -> all_to_all(counts)
  -> all_to_all(request rows)
  -> owner index_select
  -> all_to_all(payload)
  -> scatter
```

不生成全局 slot template。通信归属固定为：

```text
DataLoader 双缓冲       feature prefetch / H2D
runtime Snapshot scan   layerwise embedding / Evolve context
endpoint operator       endpoint Route
autograd / DDP          reverse Route / gradient sync
StateManager            owner commit / cache refresh
```

现有 `CommScheduler` 只保存 process group、通信 stream 并提供 async
launch/finish；它不调度这些阶段。双缓冲握手固定 Stage B/A 的跨线程提交顺序，
layerwise scan 独立固定 history/layer 顺序。它们使用同一 process group 时，所有
rank 必须经过相同的 collective 调用点；没有 payload 的 rank 发送空 tensor。

固定流水线为：

```text
Stage C / CPU native
  -> request_queue(maxsize=1)
Stage B / pinned H2D + scheduled dependency communication
  -> ready_queue(maxsize=1)
Stage A / exact dependency + GPU compute + task + state handling
```

`request_queue` 只传递：

```text
(window_id, targets, blocks, node_ids, edge_rows)
```

目标实现中不存在 `BatchRequest`、per-window callable、第三个队列或 Stage wrapper。
稳态最多同时持有 A(k)、B(k+1) 和 C(k+2)。CUDA event 和 `record_stream` 管理
buffer 复用；热路径不能使用全局 `synchronize`、`.cpu()` 或 `.item()`。

## 物理实现参考图

MemShare 参考位于 `/home/zlj/MemShare-public/MemShare`：

- `csrc/_sampler/`：temporal sampling 与 compact native output；
- `starrygl/sample/count_comm.py`：动态通信分组；
- `starrygl/sample/graph_core/`：pinned buffer 与 feature transfer；
- `starrygl/sample/memory/shared_mailbox.py`：memory/mailbox 布局；
- `starrygl/module/historical_cache.py`：historical filter/cache 算法；
- `starrygl/sample/data_loader.py` 和 `stream_manager.py`：overlap 模式。

FlareDTDG 参考位于 `/home/zlj/FlareDTDG/flare2`：

- `core/route.py`：支持 autograd 的 Route；
- `data/stc_loader.py`：Snapshot-CSC、chunk prefix 与 H2D 数据流；
- `nn/async_module.py` 和 `nn/graphconv.py`：layerwise 调度；
- `nn/light/evolvegcn.py`：EvolveGCN MatGRU 与算子顺序；
- `nn/light/tgcn.py` 和 `nn/light/mpnn_lstm.py`：decoupled Snapshot 数学。

应把参考算法和布局移植到 `src/starrygl`，不能把任一参考仓库作为 runtime 依赖。
实质复制实现时必须保留来源说明并遵守许可证。

## 当前 No-Go 项

实现已按本文开始收敛，但至少完成以下项目之前，论文发布状态仍是 `No-Go`：

- 同一 process group 上 feature/layerwise/endpoint/gradient/state 的空参与者和
  偏斜到达多 rank 正确性；
- exact state watermark，以及 stale-cache filter、`max_skip`、split preserve、
  epoch reset 和 checkpoint restore；
- native Event cutoff row、Snapshot `[H,2]` 及 counts/request/payload Route；
- 已落地双队列、逐批 pin 和 non-blocking prefetch stream；仍需 native reusable
  pinned buffer、安全回收和 GPU trace；
- 一个 runtime-owned Snapshot operator executor，覆盖 decoupled、
  `neighbor_recurrent` 和 `model_recurrent` 顺序；
- Evolve replicated state 及 coupled-DTDG exact/stale-cache 多 rank parity；
- paper-scale quality、convergence、memory、overlap 和 4-to-16-GPU gate。

论文评测的吞吐门槛仍为 TGN `3.23x`、TGAT `4.17x` 和 T-GCN `2.84x`。单元测试
和 plan-lowering 示例不能单独满足这些门槛。

## 权威模块细节

- [Prepare](../src/starrygl/prepare/CONTRACT.md)
- [Partition](../src/starrygl/partition/CONTRACT.md)
- [Store](../src/starrygl/store/CONTRACT.md)
- [Native sampler](../src/starrygl/native/CONTRACT.md)
- [View 与 GraphBlock](../src/starrygl/view/CONTRACT.md)
- [Batch](../src/starrygl/batch/CONTRACT.md)
- [Task](../src/starrygl/task/CONTRACT.md)
- [Runtime Stage A/B/C](../src/starrygl/runtime/CONTRACT.md)
- [DataLoader queues](../src/starrygl/runtime/dataloader/CONTRACT.md)
- [T-CSR sampling](../src/starrygl/runtime/sample/CONTRACT.md)
- [Snapshot-CSC execution](../src/starrygl/runtime/snapshot/CONTRACT.md)
- [State、await 与 cache](../src/starrygl/runtime/state/CONTRACT.md)
