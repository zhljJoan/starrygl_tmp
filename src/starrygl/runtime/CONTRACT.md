# 运行时主流程契约

状态：这是 Event、Snapshot、node 和 edge 共用的目标流程。只有 graph accessor、
任务服务、依赖 provider 和 Stage-A operator schedule 可以特化。

## 训练开始前只初始化一次

进入窗口循环前，runtime 完成：

- 绑定全局窗口范围、`task_ptr`、task、negative sampler 和 graph accessor；
- 需要 neighbor sampling 时创建一个 task-static native sampler；
- 绑定 FeatureManager、StateManager、prepared Route 和 model/task callables；
- 创建 optimizer、compute/prefetch stream，并绑定共享 process group/通信上下文；
- 固定 DataLoader、runtime scan、endpoint、autograd 和 StateManager 各自拥有的
  通信调用位置。`ExecutionPlan.execution_order` 只描述高层语义 lowering。

这些对象不能保存进每个 Batch、逐层转发或在 epoch/window 热循环重新创建。

## 公共流程

```text
global window k
  -> prepared task slice
  -> local negatives
  -> bound graph accessor
  -> Stage C emits (window_id, targets, blocks, node_ids, edge_rows)
  -> Stage B materializes every prefetchable dependency
  -> Stage A satisfies exact dependencies and executes model/task/state
```

一个 Snapshot Batch 可以包含多个 temporal unit。状态依赖的 producer/consumer
始终使用全局 temporal-unit id，不能把 dataloader Batch 编号当作 state version。

### Stage C：CPU/native 目标和图访问

Stage C 在有界后台 worker 中：

1. 取得 request slot 后，用 `task_ptr` 切本窗口 target；
2. 在 sampler 所在设备在线生成 negatives，并把 task/state roots 合并；
3. 一次调用 T-CSR native sampler，或按 ptr 选择 Snapshot-CSC；
4. 产生短 tuple 后放入 `request_queue`。

首版 native 路径全程 CPU并释放 GIL。Stage C 不读取远程 feature/state，不调用
模型，也不把 GPU roots 同步回 CPU；Stage B 负责 pin 当前 CPU tensor。native
直接填充可复用 pinned buffer 仍是后续性能项。

### Stage B：H2D 和可预取依赖

Stage B 使用一个 materializer：

1. 从短 tuple 组装 `Batch`；
2. 在 prefetch stream non-blocking 搬运图和本地 tensor；
3. 发起 exact node/edge feature 和允许提前的 stale-cache dependency；
4. dynamic sampled Route 先做 counts/request rows，再做 payload；prepared Route
   直接做 payload；
5. 记录 ready event，并把内部短 tuple `(Batch, ready_event)` 放入 `ready_queue`；
   Stage A 用 compute-stream `wait_event` 接续，不做 host wait。

Ready 表示所有可提前依赖已完成，不表示 exact state 已填充。通信 handle、event、
plan 和通信上下文不能进入 `Batch` 或 `GraphBlock.cache`。

### Stage A：exact、模型、任务和提交

主线程依次：

1. 满足当前 consumer 的 exact state watermark；
2. 调用 `model.encode(batch)`；
3. 在 model output 产生后精确收集 edge task endpoint embedding；
4. 计算 loss，执行 backward、反向 Route、gradient sync 和 optimizer step；
5. 调用 `model.state_update`，由 runtime 校验、owner commit、可选 cache refresh；
6. 计算任务指标。

StarryGL 共享分布式图算子由 runtime 在初始化时绑定，拥有 Snapshot scan 和
Route/Await；模型只提供 tensor math，不能直接读写 Store、
StateManager、cache 或调用 `torch.distributed`。EvolveGCN context reduction 也由
runtime operator schedule 发起。

空 target rank 仍运行 active graph、与正常 output 相连的零 loss、backward、
gradient sync、state commit 和所有必要 collective 调用点。同步训练在梯度归约后
通过 CommScheduler 对实际 post-hook supervision 做一次标量 MAX；任一 rank 有
监督时所有 rank 推进 optimizer，全 rank 无监督时都不推进 Adam 参数、moment
或 step。非同步训练使用本地监督标志。

## D_remote 的 consumer 位置

| 数据 | consumer 前的位置 | freshness / cache |
| --- | --- | --- |
| node feature `x` | Stage B，首个图算子前 | exact；provider-local cache only |
| edge feature `edge_feat` | Stage B，message 前 | exact |
| prefetchable temporal state | Stage B | `stale_cache` 显式近似；local/shared_hot |
| exact/mixed temporal state | Stage A，对应 GCN/RNN 前 | exact 或整组保守 exact |
| layer embedding | Stage A，下一 GNN layer 前 | exact；cache=none；autograd |
| task endpoint embedding | Stage A，model output 后/task head 前 | exact；cache=none |
| state delta | Stage A，producer 完成后 | owner commit；cache refresh 可选 |

本地命中时 payload 可以为空，但依赖含义不变；只要 peer 仍需要该 collective，
本 rank 就必须以空 payload 参与。

## 通信归属与顺序

不生成第二份静态 `CommPlan` 或全局 slot 表。通信顺序由已有执行路径直接确定：

```text
DataLoader              feature prefetch / H2D
runtime Snapshot scan   layerwise embedding / EvolveGCN context
endpoint operator       endpoint Route
autograd / DDP          reverse Route / gradient sync
StateManager            owner commit / cache refresh
```

`ExecutionPlan.execution_order` 仍只到图访问、依赖、model/task/state 粒度；
它不表示 backward、optimizer、stream event 或逐 collective 顺序。task loss 位于
owner state commit 之前；Snapshot 的 Batch 内 recurrent carry 属于 runtime scan。

初始化完成 `prefetch(k0)`；稳态先提交 `prefetch(k+1)`，再执行 A(k) 的 exact、
layer forward、endpoint、autograd reverse、gradient 和 commit。Stage B 在
A(k) ack 前不能提交 `k+2`。该握手固定同一 process group 上 Stage B/A 的相对
launch 顺序；每个子系统内部按固定代码路径执行。peer 仍参加的调用点中，空 rank
发送空 tensor，不能按本地 Route/target 大小跳过。

`reschedule` 只能改变本地 ready operator，不能跳过、重排 collective 或绕过
watermark。首版只允许 `wait_policy=block`。

## Device、边界和退出约束

- Stage C owns CPU/native scratch；Stage B owns pinned/H2D/NCCL buffers；Stage A
  owns model tensors until backward/state delta 完成。
- Buffer 通过 CUDA event/`record_stream` 回收，steady state 不用全局
  `synchronize`、热路径 `.cpu()`/`.item()`。
- split 切换前 drain 当前边界要求完成的 owner commit/collective，但保留 temporal
  state、cache、watermark 和 filter counter；split 只切监督范围，同一 epoch 不
  reset。重叠 chunk-decay 只保留最后 Snapshot 的 detached boundary carry，不保留
  其他 Batch-local live tensor。
- 新 epoch 从时间线起点重放：先 drain，再 reset temporal state、cache、
  watermark、filter counter 和 boundary carry；模型参数与 optimizer 保留。
- checkpoint save 先 drain 并保存 cursor 及完整 resume state，不 reset；load 同一
  cursor 时恢复运行态，load 到新 epoch 时执行 epoch reset。没有 runtime state 的
  checkpoint 不能直接从 epoch 中间继续。
- 正常结束、异常和 debug early stop 都 stop/join worker，并 drain 或 cancel owner
  commit 与已 launch collective。

## 当前未对齐

- `CommScheduler` 已收敛为薄通信上下文，不再记录 ordinal，也不承担全局调度；
  类名暂不改动，避免没有行为收益的迁移 churn。
- coupled scan、EvolveGCN context reduction 和 endpoint collect 已由 runtime
  发起；endpoint 已由训练循环直接传入 scheduler。layer/state Route 仍有部分路径
  通过 block cache 取得 scheduler。
- Stage B 已使用 pinned/non-blocking prefetch stream、compute-stream event wait 和
  `record_stream`，但 dynamic Route 的 counts/request-node 子阶段仍同步；
  native/prepared Route 和 reusable buffer pool 尚未落地。
- 同一 process group 上 feature、layerwise、endpoint、DDP 和 state commit 的
  空参与者/偏斜到达两 rank 测试仍不完整。
- 空 target 的 Stage-A/backward/gradient/state 路径已有两 rank Gloo/NCCL
  聚焦覆盖；任意用户模型/自定义 callback 的完整 collective 行为仍需各自验证。
