# GLM5.2 free-running DP:删除协调者的架构设计

> **TL;DR:** 设计文档(未实现)。EP16 P/D 难支持暴露的不是 P/D 问题,是 DP 架构问题:现在的
> DP 只切了状态没切控制,每个 rank-local 异步事实(P/D pull、offload、断连)都要经协议送回中央
> coordinator 才能生效,于是每个新 feature 都在给中心发明新协议(rank-host、Event plane)。
> 本设计把 coordinator 删除:每个 DP rank 是完整独立 engine(自己的 scheduler/BlockPool/HTTP
> endpoint),loop 无条件全速跑,唯一耦合是固定节拍的 DeepEP collective 链本身。换来的义务是
> 三条静态纪律(固定链、保守 bound、padding 即协议)。**§8 gate 1–3 已在 GB300 tray03 全部
> GO(2026-07-30):跨 rank 流量不变性 bit-exact、异构 token 数 graph 回放 bit-exact、
> 保守 bound 税 = 0(180.9 vs 180.2 µs/层)。** 方向放行,进入 §10 迁移第 1 步。
> 取代 `cross-node-scaling.md` 的 Event plane 与 SMR 方向(该文档的 NVL72 实测数据仍有效)。
>
> **Last touched:** 2026-07

## 1. 问题:DP 只切了状态,没切控制

现状(`scheduler/mod.rs` 的 `run_dp8_coordinator`)是"一个大脑、N 只手":admission、bucket
规划、launch-ahead lease、MTP round 协商、输出应用、client 回复全部集中在一个线程,rank worker
只执行。DP 切分的只有 KV 和 slot 状态。

这个形态下,每个 rank-local 的异步事实都必须传送回中心才能生效:

- P/D pull 完成 → `StartKvPull` 命令 + Event frame 回报 + parked 队列 + all-idle 时 5ms
  sleep 节流(`scheduler/mod.rs:386-393`);
- offload save 落地 → `SavePin::drop` 回调 → 未来的 Event frame;
- client 断连 → coordinator 的 `token_tx.is_closed()` 探测。

`cross-node-scaling.md` 的 rank-host 协议和 Event plane 设计,本质是给中心化架构打的远程补丁
——**补丁的存在本身就是架构问题的证据**:如果 rank 天生独立,这些协议根本不需要发明。EP16 P/D
只是下一个穿不过 coordinator 循环的 feature,不会是最后一个(`run_dp8_coordinator` 已经 14 个
参数)。

Repo 里两条模型线其实各选了一边:kimi-k2(`models/kimi-k2/dp-design.md`)选了 per-rank
独立 engine + EP 天然 sync,glm52 选了中央 coordinator。本设计是把 kimi 的选择泛化到 glm52,
并补上 kimi 文档没处理的 graph/MTP/padding 细节。

## 2. 不可约同步分析

EP 约束下真正删不掉的同步,穷举后只有:

1. **Collective 步调**:每 rank 进入每层 dispatch/combine 的次数和顺序必须一致。错位
   (mispair)= 某 rank 的第 N 层和别人的第 N+k 层配对 → 字节确定性垃圾,不 crash
   (`fail_step` 注释描述的场景)。这是最恶性的失败模式。
2. **协议上界**:collective buffer 按 `num_ranks × GLM52_MAX_BATCH_PER_RANK` 的最大形状
   分配。注意——**这不等于"每 rank 相同 bucket"**:`moe_ep_wo.rs` 的 dispatch 传 rank-local
   `num_tokens`,`global_tokens` 只用于收紧 masked GEMM 的 tile bound,recv 侧真实行数从
   device 上的 `psum_expert` 读。"bucket 全局一致"是 `plan_step_shapes` 取 hungriest rank
   的**选择**,不是 DeepEP 的要求。
3. **Fate-sharing**:一步失败 collective group 无法重新对齐,全 fleet 一起死。物理事实,
   任何架构下都在。

由于每 step 的 collective 链由模型代码写死(75 层 MoE,顺序无自由度),第 1 条退化为:
**所有 rank 对 step 计数一致**。唯一的自由度是"step N 跑不跑"——把 loop 改成无条件跑,
这个自由度也消失,不变量从运行时协议保证变成代码结构保证。

其余一切——admission、bucket、KV pool、sampling、client 回复、prefix cache、offload、
P/D pull、launch-ahead lease——都是 rank-local 的。集中在 coordinator 是历史选择,不是必然。

Idle 协调明确不做:部署姿态是机器默认满负荷,空 rank 全速跑 padding step,不引入任何
全局活跃度协议。

## 3. 目标架构

```
                        ┌──────────────────────┐
                        │  外部 router(无状态)   │
                        │  KV 亲和 / least-load  │
                        └───┬────┬────────┬────┘
                            │    │        │      ← 普通 HTTP,每 rank 一个 endpoint
                   ┌────────┘    │        └────────────┐
                   ▼             ▼                     ▼
            ┌────────────┐ ┌────────────┐        ┌────────────┐
            │ Engine 0   │ │ Engine 1   │  ...   │ Engine N-1 │
            │ HTTP       │ │ HTTP       │        │ HTTP       │
            │ scheduler  │ │ scheduler  │        │ scheduler  │
            │ BlockPool  │ │ BlockPool  │        │ BlockPool  │
            │ offload/PD │ │ offload/PD │        │ offload/PD │
            │ GPU worker │ │ GPU worker │        │ GPU worker │
            └─────┬──────┘ └─────┬──────┘        └─────┬──────┘
                  │              │                     │
                  └──────────────┴─────────────────────┘
                      DeepEP collective 链(唯一的运行时耦合)
                      每 step 固定:75 层 dispatch/combine + 5 个 MTP forward
```

- **控制面:不存在。** 没有 coordinator、step bell、Event frame、rank-host 协议。同步就是
  collective 的 back-pressure 本身。
- **数据面:一条固定链。** 每 step 每 rank 跑同一条编译期确定的 collective 序列。
- **请求面:普通 HTTP。** router 无状态(dynamo 路由已有先例,
  `subsystems/router/kv-aware-routing.md`);engine 间对彼此的请求/KV/slot 一无所知。
  local/remote 的区分从概念里消失:跨节点部署 = 每节点起进程加入同一个 DeepEP communicator,
  没有中心要连。

### 单个 engine 的 loop

```
loop {
    drain HTTP 请求 → 本地 admission(BlockPool 全生命周期预留,honor-or-reject)
    P/D:带 hash 的请求 → 本地 reserve → 发起 pull → 本地 parked 队列
    pull 完成(本地回调)→ 下一轮 admit
    按本地 slot 状态选 bucket → 选对应 graph
    step:固定 collective 链(有活带活,没活 padding 进场)
    apply 输出 → 直接回自己的 client
}
```

唯一让它区别于单卡 engine 的规则:**forward 无条件、collective 不跳过**。除此之外它就是
一个普通的自治推理引擎。TP8/TP4 mirrored 拓扑本来就是这个架构的 N=1 特例,原样不动。

### Draft/verify 两条 lane

- **Verify 免费**:verify 是 target step 里的 `SpanKind::Speculative` span,改变行数不改变
  collective 条数。行数差异由 gate 1 覆盖。
- **DSpark 零改动**:drafter 是 5 层 dense,rank-local 无 collective(`run_draft_round`
  注释自证),原样保留。
- **Native MTP 是真正的手术点**:layer 78 是 MoE 层,draft forward 也是 EP collective。
  现在 `select_round_kind`(`scheduler/mtp.rs`)按全 fleet 状态协商 Reset/Context/Propose
  ——每步 collective 总条数是变量,这正是中心化的残留。改法:**每 step 无条件跑固定 5 个
  layer-78 forward**,没活的 rank 以 padding 进场。steady decode 下 Propose 本来就几乎每步
  跑,固定链在主流工况零额外成本;round kind 协商、`source_bucket` 一致性 ensure、bucket
  全局 max 全部删除。

### Launch-ahead

lease 的 all-ranks-or-none 是 bucket 全局一致的推论;bucket per-rank 化 + 步进无条件后,
lease 降级为 rank-local bit:rank A replay 投机步、rank B 跑普通步,collective 按计数照样
配对。

## 4. 三条纪律(写进 conventions,是本架构的承重墙)

1. **固定链纪律**:任何 feature 不得引入"有条件的 collective"。省 collective 的唯一正确
   姿势是让空 rank 以零负载进场、kernel 内部便宜地穿过去;跳过靠 kernel,不靠 host 协商。
   CI 守法:数一个 step 的 collective launch 次数,断言是常量。
2. **形状本地纪律**:任何进 collective 的 buffer 按协议最大值做保守 bound,不得依赖
   "别人此刻的真实行数"。
3. **Padding 即协议**:任何进 collective 的 dummy 行,其全部输入(token、position、
   seq_len、KV 页内容、MTP shifted token)必须构造性确定,且有 byte-stability gate 守护。
   禁止"输出会被丢弃所以输入无所谓"——输出被丢弃,但路由和字节已经上了 wire,影响的是
   别人的 step。

## 5. Padding corner cases(纪律 3 的展开)

Free-running 后 padding row 从"本地丢弃的废行"升级为**协议表面**:

- **空 rank 的进场姿势:选整 bucket dispatch(路 B),不选 `num_tokens=0`(路 A)。**
  路 A 语义干净但 `num_tokens` 是 kernel 实参,每步变化破坏 whole-step graph;路 B 是今天
  的实际行为(`global_tokens = ep_ranks * batch`),graph-safe,代价是空 rank 发真实 a2a
  流量——满负荷部署下是零头。`token: None` 路径保留给 prefill-only。
- **Padding row 路由必须确定**:现有 `GLM52_PADDING_STEP` 契约(固定 token、position 0、
  seq_len 1、写 padding page 位置 0)大概率已构造性确定,但 indexer sparse top-k 在
  seq_len=1 上的行为和 fp8 quant 两个环节未验证——gate 3 把它从"碰巧对"升级成契约。
- **Lease × padding 位置走漂**:leased replay 的 `slot_mapping += 1` 不得推进 padding row
  (现有"padding rows reset by each full prologue"在全局 lease 下成立,本地化后需重新确认
  连续多步 lease 的复位边界)。
- **MTP dummy round**:固定 5 forward 后,零 proposal 的 rank 跑 bucket-1 dummy forward,
  需要明确的 `MTP_PADDING_STEP` 契约(layer 78 消费 shifted token),不得复用 capture
  buffer 残值——固定链下残值读取会变成每步都发生的事。
- **输出侧**:padding row 的 argmax/sampling 输出本地丢弃,现状已对,零新语义。

## 6. 失败模型

**Engine 内部错误 → crash early(现有姿势);任何 rank 死 → 全 fleet 经 collective 超时
数秒内 fail-stop → router 摘流量 → 全体重启。** 没有部分存活,没有脑裂可能——没有需要
一致的共享状态。KV 温数据在 pegaflow host tier 等重启后 restore。

变化只在检测与收尸的去中心化:每 rank 自己的 step watchdog(超时 → fail 自己的请求 →
进程退出),router 健康检查摘除,不再有"负责宣布死亡"的线程。fate-sharing 靠超时传染完成。

启动期协调无法归零但一次性:DeepEP 的 `ncclUniqueId` 分发 + graph precapture 全员就位,
退化为最小 bootstrap rendezvous(单机进程内;跨节点约定 rank0 节点发 id),fail-stop,
与运行时控制面无关。

## 7. 代价清单(诚实版)

- **Prefill 延迟税还在**:fleet step 时间 = 最慢 rank,rank A 跑 prefill span 时全员 TPOT
  变差。per-rank bucket 删掉的是算力税(别人不再陪跑 bucket-8),延迟税是 EP 物理。
  **这正是本架构与 P/D 互相成就之处**:P/D decode fleet 没有 prefill,step 时间天然均匀
  ——架构最成立的部署形态恰好是 ep16 P/D decode 端,而 ep16 P/D 也只有在此架构下不需要
  Event plane。
- **Debug 变难**:单状态机 + contract tests 是现有资产;N 个独立状态机后,交织类问题复现
  变难。缓解:决策核心(admission/plan/slot)本就是纯函数,per-rank 复用后 contract tests
  原样保留;`cross-node-scaling.md` SMR 章节的 replay journal 片段单独实现,per-rank 挂
  本地 journal。
- **Load 不均从调度问题变 router 问题**:`lessons/moe-dplb-decode-imbalance.md` 已预言
  ——engine 吐原始 progress,router 负责均衡。
- **N 份 HTTP/tokenizer**:小钱,权重本来就 per-rank。

## 8. Go/no-go kernel gates(先于任何架构代码)

**结果(2026-07-30,GB300 tray03 单 tray 4 GPU,`susun-dev`,commit `16d95344`):
gate 1–3 全部 GO。** 前三个 gate 实现于 `openinfer-glm52/src/oracle/freerun_ep4.rs`,
按 EP4 形状写(一个 GB300 NVL72 tray = 4 GPU,走 weight-only 链——正是 NVL72 上的
生产链)。运行(**每个 gate 必须单独一个进程**,见下面的 pitfall):

```bash
for g in freerun_hetero_traffic_gate freerun_hetero_graph_gate freerun_bound_tax_probe; do
  OPENINFER_TEST_MODEL_PATH=/mnt/shared/weights/GLM-5.2-FP8 EP_DISABLE_GIN=1 \
    cargo test --release -p openinfer-glm52 --lib "$g" -- --ignored --nocapture
done
```

1. **`freerun_hetero_traffic_gate` — 跨 rank 流量不变性。✅ PASS。** 同一组 DeepEP
   context 跑两遍 layer-6 oracle walk:pass A 旁路 rank 全 token-less,pass B 旁路
   rank 每 position 推 0..=8 变化 token 数。验收:两遍都过 oracle probes,且 rank 0
   两遍输出逐值 bit-identical。实测:quiet 与 hetero 各 63/64 probes(同一个已知
   router tie-flip outlier,与既有 EP4 oracle gate 一致),200×6144 个输出值零 bit
   抖动——"一个 rank 的行的计算与别人的流量无关"成立。
2. **`freerun_hetero_graph_gate` — 异构 token 数的 graph 回放。✅ PASS。** 4 个 rank
   各以不同 token 数(1/2/4/8)capture routed 链的 CUDA graph 并回放 16 次,每次
   combined 输出与 eager 参考 bit-identical。whole-step graph(含 attention/采样)的
   同类验证留到迁移第 1 步的 e2e gate。
3. **`freerun_bound_tax_probe` — 保守 bound 的性能税。✅ GO,税 = 0。** 每 rank
   1 token 的 steady-decode 形状,256 次均值:tight(`global_tokens=4`)180.9 µs/层,
   protocol-max(=32)180.2 µs/层——**差异在噪声内,方向还是反的**。整条 weight-only
   链对 `global_tokens` 不敏感(它只收紧 tiles kernel 的扫描上界,GEMM 工作量由
   device 侧 psum 决定)。原判读标准(≤0.5ms/step → go)以最强形式满足,"per-rank
   静态 bound 档位"退让方案不需要。EP16 复测仍保留(shim 常量不同),但 EP4 的零税
   使不同结论的先验概率很低。
4. **Padding 字节恒定(未实现,需 engine 级 harness)。** 同一 rank 以
   `GLM52_PADDING_STEP` 输入空转 N ≥ 64 步,每步 D2H 抓 router 输出。**验收:
   `topk_idx`/`topk_weight` 字节逐步恒定**,覆盖 indexer seq_len=1 与 fp8 quant 环节。
   实现挂在迁移第 1 步(whole-step 路径上加 probe),因为它测的是完整 step 的 padding
   行,不是孤立 kernel。
5. **MTP 固定链(未实现,需 layer-78 harness)。** per-rank context/draft 行数不等 +
   空 rank padding 下 layer-78 五个 forward 的正确性(对 `oracle/mtp.rs` 既有 probe),
   加 Reset/Context 工况强制跑满 5 forward 的开销测量。**验收:probe 过 + 空 round
   开销 ≤ 0.5 ms**(预期是零头,须实测)。同样挂迁移第 1 步。

**Pitfall(实测踩中):DeepEP context 是一进程一次性的。** 三个 gate 在同一个 test
进程串行时,第二个 gate 的 `ctx_create` 撞 NVLink barrier timeout →
`unspecified launch failure`——与 rank-host 契约记录一致("worker drop does not
return all hosted GPU state; process exit is the release mechanism")。gate 必须
每个单独一个 `cargo test` 进程。这也是 free-running 架构的一条部署事实:engine
进程的生命周期 = DeepEP context 的生命周期,重启即换进程。

## 9. 代码映射(删多于加)

| 现在 | 之后 |
|---|---|
| `run_dp8_coordinator`(14 参数,持全 fleet 状态) | 降维成单 rank engine loop;`Vec<RankSlots>` → `RankSlots`,`for rank` 循环消失 |
| `plan_step_shapes`(hungriest rank) | 纯本地 `plan_step_shape(&my_wants)`;函数仍纯,contract tests 保留 |
| `launch_ahead_flags`(all-ranks-or-none) | 纯本地 bit |
| `select_round_kind` + MTP 全局 bucket/ensure | 删除(固定 5-forward 链) |
| `remote.rs`(978 行)+ rank-host + Event plane | 退役;跨节点 = 每节点进程加入同一 communicator |
| `VllmPdState`/`NativePdState`(coordinator 持有) | engine 本地字段;parked 5ms sleep 节流消失 |
| mirrored TP8/TP4 | 原样(本就是 N=1 特例) |
| `fail_step` 全局收尸 | 本地 watchdog + router 健康检查 |

## 10. 迁移路径

不 big-bang。gates 绿后两步:

1. **协议先变,结构后变**:保留 coordinator 的壳,把 bucket、lease、MTP round 全部
   per-rank 化(kernel 侧已按 gate 验证支持)。每步有现有 golden/e2e gate 兜底。
2. **拆壳**:coordinator 循环拆成 N 个 engine 线程,各自挂 HTTP endpoint;bootstrap
   rendezvous 收编启动逻辑。跨节点 = 同一 binary 每节点一进程。

## 与 cross-node-scaling.md 的关系

该文档的 NVL72 实测数据(EP4→EP32 bucket-1 p50 平坦、IMEX/teardown 坑)仍然有效且
load-bearing。被本设计取代的部分:framed-TCP rank-host 作为**长期架构**(短期已 shipped
可用)、Event plane(facts plane)设计、SMR coordinator 方向。replay journal 片段被本
设计吸收(第 7 节)。

## Next step

Gate 1–3 已 GO(§8)。下一步是 §10 迁移第 1 步:保留 coordinator 壳,把 bucket、
lease、MTP round per-rank 化,并在 whole-step 路径上补 gate 4(padding 字节恒定)
和 gate 5(MTP 固定链)的 probe。EP16 的 bound-tax 复测挂在第一次跨 tray 部署时顺带跑。
