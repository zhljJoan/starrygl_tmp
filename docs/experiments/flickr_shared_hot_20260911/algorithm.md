# 本次实际运行的算法

每个快照 t 使用当前图、当前入度/出度特征，预测下一快照的
`log(1 + in_degree)`。两个 rank 共享模型参数；节点 owner 负责输出、损失
和权威状态提交。远端副本只提供模型输入。窗口之间保留 detached 隐藏状态，
每轮训练开始时重置状态，不跨窗口反向传播。

## 状态读取

```text
exact:
    本地节点：读取本地 owner 状态
    远端节点：拉取远端 owner 状态

bounded_stale/shared_hot:
    本地节点：读取本地 owner 状态
    远端 hot 节点：直接读取 shared_hot 缓存的状态
    其余远端节点：拉取 owner 状态并等待完成
```

因此近似作用于远端 hot 邻居状态。例如窗口 t 需要 h[t-1]，缓存可能返回
h[t-2]。没有人为固定延迟，没有把本地 owner 状态也统一延迟。
本次 `access_pipeline=False`、`wait_policy=block`，没有评估 reschedule
隐藏通信的收益。读取实现为
`src/starrygl/runtime/memory/historical.py:materialize_bounded`。

## 模型计算

记 x' = W_x x，h~ 为实际读到的历史状态。GConvGRU 先计算
`m = GCN(A, concat(x', h~))`，再计算本地节点
`h_new = GRU(m, h_owner_previous)`。

DCRNN 的门和候选状态使用双向扩散卷积：

```text
u, r = sigmoid(DiffConv(A, concat(x', h~)))
c = tanh(DiffConv(A, concat(x', r * h~)))
h_new = u * h_owner_previous + (1 - u) * c
prediction = Linear(h_new)
```

本实现的扩散包含零跳、正向一跳和反向一跳，使用单位自环与全局方向度
归一化。跨分区计算候选状态前，runtime 通过现有 autograd Route 交换
当前窗口的 reset gate r。这个当前值通信没有用 stale 缓存替代。
这是节点回归的 DCRNN cell 实验；具体单元约定见设计说明。

## 权威提交与缓存发布

窗口损失只在 owner 节点上计算；各 rank 同步梯度并执行 Adam 更新。
模型返回 detached `StateDelta`，runtime 提交 owner 状态。hot 节点的候选
状态另经过 `SharedStateRefreshFilter`：

```text
distance = 1 - cosine(new_state, last_accepted_state)
publish = (distance > 0.3) or (consecutive_skips >= 1)
if publish:
    last_accepted_state = new_state
    consecutive_skips = 0
else:
    consecutive_skips += 1
```

获准候选通过调度器的 shared-hot collective 发布，缓存保存状态和版本。
发布过滤不会阻止 owner 提交新状态，也不会改变损失或检查点归属。

**当前 max_staleness=1 限制连续跳过发布的次数。读取时没有按实际版本差
再次检查并强制刷新，因此配置本身不构成严格的窗口年龄保证。** 实验独立
审计实际返回值：快照 s 产生的状态标记版本 s+1；窗口 t 期望版本 t；
`lag = t - returned_version`。当前观测到 shared lag 0/1，owner lag 0，
没有未来状态行。

## 补偿

模型保留 `h_comp = h_cache + sigmoid(gamma) * estimated_increment`。
但增量更新要求被读取的 shared 节点也是本 rank 计算的新状态节点。在本次
owner-only 全快照布局中，shared 读取属于远端节点，输出属于本地 owner，
两者不相交，增量更新为零。补偿开关不能在这次运行中提供非零修正。

## 对照与计时

每个模型分别训练 exact/shared 两组，使用同样的种子、分区、特征和标签。
按实际读取策略下的验证 MSE 选择最佳检查点。对 shared 组的同一个检查点
重放 operational 与 exact 两种状态读取，区分训练差异和读取差异。

计时比较相同轮数的训练时间，以及首次观测到相同验证误差所需的累计时间。
hot 命中只替代部分状态拉取；cold-node 拉取、GCN 计算、hot 发布、梯度同步
和 DCRNN reset-gate 交换仍然保留。共享 GPU 且流水线关闭，因此当前计时
不能单独证明共享缓存的独占吞吐收益或开销归因。
