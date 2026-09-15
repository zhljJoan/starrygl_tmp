# StarryGL 执行契约索引

状态：这是目标契约。实现、分布式正确性和 paper gate 全部通过后，迁移才完成。

模块目录中的 `CONTRACT.md` 是唯一有效版本。`docs/design/current/` 只保留
索引，不复制正文。

## 编译阶段固定什么

`compile` 生成一个 `ExecutionPlan`，一次固定高层语义 lowering：

1. Prepare 需要生成哪些图视图。
2. 一个训练 Batch 对应哪个全局窗口，是否包含多个 temporal unit。
3. 图访问、状态依赖和模型算子的执行顺序。
4. 每个远程依赖的 consumer、owner、freshness、cache、wait 和通信位置。
5. 每个远程依赖由哪个 runtime 组件发起和等待。

`execution_order` 不是 backward、optimizer 或逐 collective 指令表，也不再派生
另一份 `CommPlan`/slot 表。具体顺序由 DataLoader、runtime scan、autograd 和
StateManager 的固定代码路径负责。

`D_remote` 表示“某个 consumer 算子执行前必须可用的远程数据”，不是
“Stage B 要读取的所有数据”。特征和可预取的 stale-cache state 属于 Stage B；
exact state、层间 embedding、端点 embedding 和状态提交发生在 Stage A。

节点和边特征、层间 embedding、任务端点 embedding 必须精确。只有计划明确
列出的 temporal state 可以使用 `stale_cache`，且必须标记为显式近似；该策略
不承诺最大陈旧 temporal-unit 数。标签和负样本由任务服务在本地生成，不是远程
读取依赖。

## 唯一执行主链

```text
PartitionPlan
  -> Prepare 生成任务表、特征和图视图
  -> 加载 StoreBundle
  -> 一次绑定 sampler、graph accessor、Route、通信上下文和 CUDA streams
  -> Stage C：任务切片 -> 负采样 -> 图访问
  -> request_queue: (window_id, targets, blocks, node_ids, edge_rows)
  -> Stage B：Batch 组装 -> H2D -> 可预取依赖
  -> ready_queue
  -> Stage A：exact 依赖 -> 模型/任务 -> backward -> 状态提交
```

Event、Snapshot、节点任务和边任务只能在 graph accessor、任务服务或状态依赖
处不同。它们不能各自拥有 loader、队列、训练循环或配置翻译路径。一个
Snapshot Batch 可以含多个 temporal unit；状态依赖必须指向 producer temporal
unit，不能用“上一个 Batch”代替版本语义。

## 通信执行边界

现有 `CommScheduler` 只保留为薄通信上下文：保存 process group、按设备复用 CUDA
stream，并提供 Route collective 的 async launch/finish。它不调度 Stage，不拥有
compile-time slot，也不管理 backward、gradient 或 state 的语义顺序。

```text
DataLoader 双缓冲       决定 feature prefetch 的提交时机
runtime Snapshot scan   决定 history/layer embedding 的提交和等待顺序
endpoint operator       在 model output 后执行 endpoint Route
autograd / DDP          决定 reverse 和 gradient collective
StateManager            决定 owner commit 和 cache refresh
```

这些组件使用同一 process group 时，所有 rank 必须经过相同的 collective 调用点；
本地没有 payload 的 rank 发送空 tensor。Stage B/A 的固定握手约束跨线程提交顺序，
不能让线程竞争决定 NCCL 顺序。若以后要无序并发，必须先用性能数据证明有必要，
再为相应路径建立独立 process group。

Layerwise embedding 与 historical state cache 只统一以下控制面：

```text
Route -> async launch -> CUDA event/await -> scatter -> profile counters
```

二者不能统一成一个 Cache 类。Layerwise embedding 是 exact、带 autograd、仅在
当前 forward/backward 存活的临时 tensor；historical state 是 detached、按
`max_skip` 选择性刷新、可跨窗口存活的近似副本。

## 模块文档

- [图划分](partition/CONTRACT.md)
- [Prepare 与产物格式](prepare/CONTRACT.md)
- [数据存储与读取](store/CONTRACT.md)
- [C++ 邻居采样](native/CONTRACT.md)
- [GraphBlock 与 Route](view/CONTRACT.md)
- [Batch](batch/CONTRACT.md)
- [任务与负采样](task/CONTRACT.md)
- [运行时主流程](runtime/CONTRACT.md)
- [DataLoader 与双缓冲](runtime/dataloader/CONTRACT.md)
- [T-CSR 采样](runtime/sample/CONTRACT.md)
- [Snapshot-CSC 执行](runtime/snapshot/CONTRACT.md)
- [状态读取与提交](runtime/state/CONTRACT.md)
- [Paper 方法与验收](PAPER_METHOD_COVERAGE.md)

## 所有模块共同遵守

- 全局窗口编号同时索引任务、Event、T-CSR、Snapshot-CSC 和 exact state 顺序。
- `node_master` 保存节点特征、标签和节点状态的权威副本；`edge_master` 保存
  边特征，并负责边任务的 loss 和指标。
- 热点缓存只加速输入读取，不改变 owner、loss、checkpoint 或提交权限。
- Python 只枚举 epoch、训练 step 和窗口。逐节点、逐边、逐邻居、排序、通信
  打包和状态更新必须使用 Torch、DGL 或 C++/CUDA。
- CPU native 路径不做 GPU -> CPU roots 往返；GPU 热路径不做 `.cpu()`、
  `.item()` 或 default-stream 全局同步。
- 模型的公共输入只有 `Batch`。scheduler、通信 handle 和 cache 控制字段不进入
  `Batch` 或 `GraphBlock.cache`；StarryGL 共享分布式图算子由 runtime 绑定并拥有
  Route/Await，模型只描述 tensor math。
- 空 target 不等于空计算。所有 rank 仍执行该固定路径要求的图计算、零 loss、
  反向、梯度同步和 collective 调用点。

## Paper-v1 冻结门槛

Paper-v1 只以论文实际评测路径作为发布门槛：Event sampled 的 TGAT 与
JODIE/TGN/APAN，Snapshot full/chunk 的 T-GCN、MPNN-LSTM 与 EvolveGCN。
Snapshot neighbor 和 coupled GConvGRU 是契约扩展，不能代替上述基准。

发布前必须同时通过：

1. artifact、逐 Batch、两 rank 空参与者、exact state 和 stale-cache 质量对照；
2. 论文配置下的 AP/Macro-F1/MSE 与收敛复现；
3. 4 -> 16 GPU normalized throughput：TGN `3.23x`、TGAT `4.17x`、
   T-GCN `2.84x`，并且论文规模不 OOM；
4. steady-state throughput、峰值 CPU/GPU memory、通信量和 Stage overlap 的
   可复现记录。

功能单测或 plan lowering 测试不能单独证明性能达标。
