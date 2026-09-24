# DataLoader 与双缓冲契约

状态：`DataLoader` 类已固定两个深度为 1 的队列，不增加 Stage 包装类、逐批
request/result 对象或第三个队列。

## 固定流水线

```text
Stage C / CPU native
  -> request_queue(maxsize=1)
Stage B / H2D + dependency communication
  -> ready_queue(maxsize=1)
Stage A / GPU compute + task + state commit
```

稳态最多同时存在：Stage A 的 `k`、Stage B 的 `k+1`、Stage C 的 `k+2`。
Stage C 必须在调用负采样或 sampler **之前**取得 request slot；不能先构造
`k+3` 再阻塞在 `put`。Stage B 也只有在 Stage A 消费当前 ready slot 后才能
继续下一批。两个 semaphore/ack 足够，不引入生命周期对象。

## 队列传什么

`request_queue` 直接传一个短 tuple：

```text
(window_id, targets, blocks, node_ids, edge_rows)
```

它只包含本窗口变化的数据。Store、native sampler、策略、Snapshot rolling
cache、通信上下文和 materializer 在 `DataLoader.__init__` 中绑定一次，
不通过逐窗口配置、plan 或 request wrapper 转发。

`ready_queue` 内部传短 tuple `(Batch, ready_event)`。CUDA event 不写入 `Batch`；
Stage A 取出 tuple 后在当前 compute stream 上执行 `wait_event`，登记
`record_stream`，然后只把 `Batch` 交给训练循环。`ready` 不表示 exact state 已
填充，Stage A 完成 exact hydrate 后才是 model-ready。

## CPU、GPU 和 buffer 所有权

- Stage C 使用 CPU/native sampler，native 调用释放 GIL。负样本在 sampler 所在
  设备向量化生成，避免 GPU -> CPU roots 同步。
- Stage B 把 CPU tensor 转为 pinned memory，再在构造时创建的 prefetch stream 上
  non-blocking H2D；feature payload 通过共享通信上下文的 async collective
  发起。Stage-B worker 记录 event 后立即交付，Stage A 用 stream dependency 等待，
  不执行 host `synchronize()`。
- Batch tensor 至少存活到 backward 和 state delta 提取完成。Stage A 已对 Batch
  中的 CUDA storage 调用 `record_stream`；当前没有复用 pinned/send/recv buffer，
  引入 buffer pool 时仍需为池中 CPU/通信 buffer 增加独立回收事件。
- owner state、historical cache、filter counter 和热 Route 与其计算 tensor
  位于同一 GPU。稳态禁止全局 `synchronize`、默认流隐式同步和热路径 `.cpu()`。
- 正常结束、异常和 debug 提前停止都必须 stop/join worker，并 drain 已发起的
  owner commit 与 collective，不能遗留仍引用复用 buffer 的后台工作。

## Stage B 与 Stage A 的固定握手

初始化先完成 `prefetch(k0)`。稳态每个 `k` 固定为：

```text
Stage B 提交 prefetch(k+1) 的全部通信
  -> 唤醒 Stage A(k)
  -> Stage A: exact state / layer forward / endpoint / reverse / gradient / commit
  -> Stage A ack
  -> Stage B 才能提交 prefetch(k+2)
```

这里的“Stage A 同步点发起”指 Stage A 在上一批 DDP/commit 返回后释放
`ready_slot`，授权 Stage B 提交下一批 collective，并在进入下一批 compute 前确认
launch 已完成；不要求把 packing、owner read 和 launch 调用搬到训练主线程。

不再增加唯一 dispatcher 或内部 `CommPlan`。双缓冲握手直接保证每个 rank 都先
提交 `prefetch(k+1)`，再进入 A(k)；runtime layerwise scan 独立保证各层顺序。
使用同一 process group 时，各 rank 必须经过相同的 collective 调用点。缺少
`k+1` 的尾部 step、没有 target 或没有 Route 行时，只要 peer 需要通信，本 rank
仍提交空 payload。

## 空 target 的语义

某 rank 没有本地监督 target 时仍生成同一 `Batch`，只是 `targets` 为空：

- full Snapshot 仍执行本 rank 的 active CSC，并向 peer 提供边界 embedding；
- Event roots 是 task roots 和 state-update roots 的并集；
- work 也为空时，仍走与模型输出相连的零 loss、autograd reverse、梯度同步和
  必要 collective 调用点，保证参数不分叉。

`drop_last`、split 和 debug maximum 必须在 owner 分片前按全局窗口决定，不能
让各 rank 得到不同 step 数。

## 如何枚举窗口

DataLoader 的队列逻辑只遍历 source binding 给出的全局窗口 range。Snapshot
horizon、下一 Snapshot 目标和 split 尾行已由 prepared `task_ptr` 表示为空切片；
DataLoader 不接收或解析 task 名称。Event 的全局 `drop_last` 仍由 source binding
根据真实 batch size 处理。通用队列逻辑不再：

- 搜索 `time_ptr_2` 恢复另一套 Snapshot 编号；
- 为每个窗口创建 `RuntimeBatchPlan`；
- 在迭代热循环里解析任务名称或标签字符串；
- 用 split 起点截断历史图；
- 看到 `drop_last=True` 就无条件删除最后一行。

Event 的不完整尾批在 Prepare 时根据真实 batch size 决定是否保留。
`max_batches_per_epoch` 只用于调试，并由所有 rank 取相同值。

## graph accessor

任务初始化时只绑定一个函数：

```text
neighbor：sample_neighbors(roots, scope)
snapshot：选择 Snapshot-CSC 历史行和 chunk 前缀
```

两个绑定函数都直接返回
`(window_id, targets, blocks, node_ids, edge_rows)`，图部分分别来自 native T-CSR
或 prepared Snapshot-CSC，之后共用一个 Stage B materializer。

## 当前未对齐

- 通信上下文不负责跨 rank 排序；当前仍缺少覆盖 Stage B/A 偏斜到达和所有空参与者
  调用点的完整两 rank 测试。
- dynamic feature/state Route 的 counts 和 request-node 完成仍是同步子阶段；payload
  已异步发起并由上一批 Stage A 隐藏。只有 native/prepared Route 消除这个动态
  握手后，才能宣称端到端全异步。
- 目前每批 pin CPU tensor，尚未使用 native 直接填充的可复用 pinned buffer；虽已
  删除 host wait，仍需用 GPU trace 证明 pin/H2D 成本被 Stage A 覆盖。
- 空 target rank 尚未覆盖 forward、反向、梯度同步和退出 drain 的两 rank 测试。
