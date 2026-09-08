# X-Trainer-SmolVLA 官方异步推理参考代码调研与融入计划

> 适用范围：`scripts/xtrainer/run_real.py` + `deploy/xtrainer/`（WebSocket 推理服务 + X-Trainer 真机部署栈）
> 参考基线：上游 [huggingface/lerobot](https://github.com/huggingface/lerobot) main（2026-09 抓取）；本仓库（`GC-SHIRO/X-Trainer-SmolVLA`）已内嵌对应的官方模块，路径见 §2
> 撰写日期：2026-09-07

---

## 0. 结论（TL;DR）

1. **你的 `run_real.py` 不是"纯同步开环"**：它已经实现了一版相当接近官方语义的**异步预取**——单飞行请求（single-flight）+ 队列阈值 g 触发 + 新旧 chunk 重叠步 0.3/0.7 加权聚合 + 队列耗尽兜底（保持上一动作并立刻补请求）。这与 SmolVLA 论文 Algorithm 1、LeRobot 官方 `async_inference` 的 `RobotClient` 是同一类设计。
2. **官方参考代码有两条路线，且都已在本仓库内**，无需外找：
   - **路线 A（论文同源实现）**：`src/lerobot/async_inference/` —— SmolVLA 论文 §3.3 的官方开源代码（gRPC + 双线程 + 服务端"只推理最新观测 + 相似性过滤"）。你的实现缺的正是**服务端部分**（最新观测覆盖、相似性去重）。
   - **路线 B（上游最新执行引擎 RTC）**：`src/lerobot/rollout/inference/rtc.py` + `src/lerobot/policies/rtc/` + `lerobot_rollout.py --inference.type=rtc`。RTC 在预取之上解决了"新 chunk 与已执行前缀不连续/观测过时"的问题；**本仓库的 SmolVLA 已实现 `supports_rtc()` 与 `predict_action_chunk(inference_delay, prev_chunk_left_over)`**，可直接启用。
3. **建议的融合路线**：先小步补齐 A 的服务端语义（低风险），再把 RTC 语义透过现有 WebSocket 协议融进部署栈（阶段 2），最终可选迁移到官方 rollout 引擎（阶段 3）。具体见 §5。

---

## 1. 背景：官方"异步推理"到底指什么

- **SmolVLA 论文**（arXiv:2506.01844）§3.3 + §4.6：把系统拆成 **RobotClient / PolicyServer**。客户端消费动作队列，当剩余比例 `|A_t|/n < g`（阈值，论文推荐 g≈0.7）时抓取新观测、**非阻塞**触发下一 chunk 推理；新旧 chunk 重叠时间步做**聚合**；并用关节空间相似度过滤冗余观测。论文以 30 fps（Δt=33 ms）为典型控制周期推导：`g ≥ (E[ℓs]/Δt)/n` 可保证队列不排空（ℓs 为推理延迟）。
- **LeRobot 官方实现 = 路线 A**（`lerobot/async_inference`）：gRPC client/server + 双线程 + 可配置聚合函数（默认 `weighted_average = 0.3*old + 0.7*new`，与你代码里的常量一致）+ 服务端只推理最新观测 + `observations_similar` 过滤 + FPS/延迟度量。
- **RTC（Real-Time Chunking，路线 B）**：论文 arXiv:2506.07339（Physical Intelligence / Kevin Black、Sergey Levine）。它把"执行上一 chunk 的同时思考下一 chunk"当作第一公民：后台线程持续推理，用 **flow-matching inpainting + soft masking** 让新 chunk 与已执行的前缀平滑衔接，并用延迟追踪器自适应决定执行余量。对 >300ms 的推理延迟鲁棒、比同步快 ~20%。LeRobot 已将其移植为 rollout 的 inference 引擎（`--inference.type=rtc`），SmolVLA 在 LeRobot 里官方支持。

---

## 2. 参考代码盘点（越官方越好：全部为上游 LeRobot 官方实现）

### 路线 A：`lerobot/async_inference`（= SmolVLA 论文异步栈的官方代码）

| 文件（上游 main / 本仓库同路径） | 角色 |
|---|---|
| `src/lerobot/async_inference/configs.py` | `PolicyServerConfig`（host/port/fps/`inference_latency`/`obs_queue_timeout`）、`RobotClientConfig`（`chunk_size_threshold`=g、`fps`、`actions_per_chunk`、`aggregate_fn_name` 等）、聚合函数注册表 `AGGREGATE_FUNCTIONS` |
| `src/lerobot/async_inference/policy_server.py` | `PolicyServer`：gRPC servicer；`observation_queue = Queue(maxsize=1)`（**只保留最新观测**）；`_predicted_timesteps` 预测去重；`Ready()` 重置会话；`SendPolicyInstructions` 动态加载策略 |
| `src/lerobot/async_inference/robot_client.py` | `RobotClient`：`receive_actions`（daemon 线程收 chunk）+ `control_loop`（主线程按 fps 执行）；队列阈值触发发送观测；`_aggregate_action_queues` 重叠聚合；`start_barrier` 双线程同步 |
| `src/lerobot/async_inference/helpers.py` | `Observations`/`TimedAction` 数据结构、`FPSTracker`、`observations_similar`（服务端相似过滤）、队列尺寸可视化 |
| `examples/tutorial/async-inf/policy_server.py`、`robot_client.py` | 官方最小上手示例（client 侧 `chunk_size_threshold=0.5`、`actions_per_chunk=50`） |

依赖：`pip install "lerobot[async]"`（grpcio）。注意官方 `RobotClient` 绑定的是 LeRobot `Robot` 抽象（so100/koch 等），**X-Trainer 的 Dobot 硬件不是 LeRobot Robot 类型**，直接换用不现实——见 §5 阶段 1 的"语义移植"策略。

### 路线 B：RTC 官方引擎（上游最新 rollout 架构）

| 文件（本仓库路径） | 角色 |
|---|---|
| `src/lerobot/rollout/inference/base.py` | `InferenceEngine` 抽象：`get_action` / `notify_observation` / `predict_action_chunk` 查询模型 |
| `src/lerobot/rollout/inference/sync.py` | 同步引擎（每 tick 一次策略调用）——"传统行为"的官方定义 |
| `src/lerobot/rollout/inference/rtc.py` | **RTC 引擎**：后台线程 + `ActionQueue` + `LatencyTracker`，异步产 chunk |
| `src/lerobot/policies/rtc/`（`action_queue.py`、`latency_tracker.py`、`modeling_rtc.py`、`relative.py`…） | RTC 底层：延迟预测、前缀再锚定等 |
| `src/lerobot/scripts/lerobot_rollout.py` | CLI：`--inference.type=sync|rtc`（fork 内示例见该文件头部注释，RTC 例子：`--policy.path=lerobot/pi0_base --inference.type=rtc --inference.rtc.execution_horizon=10`） |
| `src/lerobot/rollout/context.py` | 装配：`is_rtc` 分支里校验 `supports_rtc_inference(policy)` 并注入 `rtc_config` |

**关键事实（已在本仓库验证）**：`src/lerobot/policies/smolvla/modeling_smolvla.py` 第 148 行 `def supports_rtc(self) -> bool`、第 232 行 `predict_action_chunk(...)`（接受 `inference_delay` / `prev_chunk_left_over`）、`configuration_smolvla.py` 含 `rtc_config: RTCConfig | None`。即：**官方 RTC 引擎可以直接驱动 SmolVLA**，不需要改模型代码。

### 论文/文档对照

- SmolVLA: arXiv:2506.01844（§3.3 Asynchronous inference、Algorithm 1、§4.6 实验）
- RTC: arXiv:2506.07339；上游 docs：`docs/source/policy_rtc_README.md`（上游 docs 仓库）、`src/lerobot/policies/smolvla/README.md`
- 若想对比差异：`git diff` 本仓库文件与上游 main 同名文件即可（上游 URL：`https://github.com/huggingface/lerobot/blob/main/src/lerobot/async_inference/...`）

---

## 3. 现状盘点：`run_real.py` 已实现 vs 官方（逐项映射）

| 能力 | 官方路线 A（async_inference） | 官方路线 B（RTC 引擎） | 你的 `run_real.py` 现状 | 差距 |
|---|---|---|---|---|
| 控制节拍 | 双线程 + fps 循环 | 主循环 + 后台推理线程 | `asyncio` 单事件循环 + `deadline += period` 节拍（`--control-hz`，默认 20，可调） | 无本质差距；单事件循环内无阻塞点即可 |
| 预取触发 | `chunk_size_threshold`（g） | 后台线程持续预测，无显式阈值 | `_should_prefetch`：`queue/action_horizon <= prefetch_threshold`（默认 **0.7**，与论文 g 同值） | ✅ 等价 |
| 飞行中请求 | client 每轮可发新观测（服务端串行取最新） | 常驻推理线程 | **单飞行** `pending_request`（asyncio.Task），完成才合并 | ✅ 更保守；慢服务端下不会堆积请求 |
| chunk 重叠聚合 | `_aggregate_action_queues` + `aggregate_fn`（默认 0.3/0.7） | TE 融合 + inpainting 前缀对齐 | `_merge_action_queue`：同 timestep 0.3/0.7 加权 | ✅ 与 A 等价；缺 B 的 inpainting |
| 队列耗尽行为 | 空转等待（论文 g=0 病态） | 执行余量自适应 | 保持 `last_sent_action` + 立刻补请求 | ✅ 优于 A |
| 观测冗余过滤 | 服务端 `observations_similar` + `Queue(maxsize=1)` 只推理最新 | 推理线程每步取最新观测 | **无**（`--observation-similarity-epsilon` 参数已预留但未启用） | ❌ 主要缺口 |
| 新 chunk 与已执行前缀的连续性 | 无（靠聚合缓解） | **soft masking / inpainting** | 无（仅靠限速 + 加权） | ❌ 主要缺口（B 的价值） |
| 延迟自适应 | `inference_latency` 仅配置 | `LatencyTracker` 预测 d、`execution_horizon` 自适应 | `request_timeout` 兜底 | ❌ 主要缺口 |
| 动作平滑/安全 | 无 | 无（RTC 保证连续性） | `_rate_limit_action` + 环境安全层 | ✅ 你的独有优势，保留 |
| 可观测性 | `FPSTracker` + 队列可视化 | debug tracker | JSONL `ControlActionLog`（`--log-control`） | ✅ 更好；可补空转统计 |

**一句话**：你的实现已经覆盖了"论文 Algorithm 1 的客户端侧"与官方路线 A 的绝大部分客户端语义；**真正的增量来自 (a) 服务端 latest-observation + 相似性过滤（A），(b) RTC 语义：前缀对齐 + 延迟自适应（B）**。

---

## 4. 修改内容与部分（按文件）

> 原则：**以"新增/小改"为主，不推翻现有可运行栈**；安全层（`deploy/xtrainer/real/environment.py`、硬件层）一律不动。

### 4.1 小改：补齐服务端最新观测 + 相似性过滤（路线 A 语义）

- `deploy/xtrainer/smolvla_policy.py`（`SmolVLAXTrainerPolicy.infer` 调用链）：
  - 现为"single-client, single-batch"串行推理。改为在**服务端入口**（见下）维护"仅最新观测"语义，避免串行推理期间的请求排队造成陈旧结果。
- `deploy/xtrainer/websocket_policy_server.py`（handler 主循环）：
  - 增加与官方 `PolicyServer` 等价的三个机制：
    1. **latest-only**：推理进行中到达的新观测只替换"待推理的最新观测"（等价官方 `Queue(maxsize=1)`），不排队；
    2. **相似性过滤**：关节空间距离 < ε 的连续观测跳过推理（官方 `observations_similar`；复用 `run_real.py --observation-similarity-epsilon` 已预留的参数，打通两端）；
    3. **预测去重**：已对该 timestep 预测过则不重复预测（官方 `_predicted_timesteps`，可选）。
  - 保持协议兼容：`run_real.py` 不需要为此改动。
- `scripts/xtrainer/run_real.py`：仅需把 `--observation-similarity-epsilon` 从"预留、警告无效"改为实际下发/启用（若过滤放客户端）或透传给服务端（若放服务端）。推荐放服务端，与官方一致。

### 4.2 新增：RTC 化的部署路径（路线 B，推荐主投入）

由于 SmolVLA 已官方支持 RTC 输入，有两种做法：

- **方案 2a（改动最小，推荐先做）**：把 RTC 语义"透过 WebSocket 协议"扩展，不引入后台线程：
  - client（`run_real.py`）在发起预取请求时附上 `inference_delay`（用滑动窗口统计最近 N 次推理耗时，官方 `LatencyTracker` 思路）与 `prev_chunk_left_over`（当前队列剩余动作的最后一段）；
  - server（`smolvla_policy.py`）暴露 `predict_action_chunk(obs, inference_delay, prev_chunk_left_over)` 调用路径（模型层已支持，只需把 feature_transform/processor 的入口从 `select_action` 换成 `predict_action_chunk` + RTCProcessor 前缀处理，参考 `modeling_smolvla.py` 内 RTC 相关分支与 `src/lerobot/policies/smolvla/README.md`）；
  - client 收到 chunk 后不再需要 0.3/0.7 硬加权，而是采用 RTC 的"前 d 步已冻结、余下 soft mask"语义落地。
- **方案 2b（全官方化，改动大）**：放弃 WebSocket，改为单进程运行官方 `rollout` 栈：`RolloutController + RTCInferenceEngine + --inference.type=rtc`。前提是把 X-Trainer 硬件包一层 LeRobot `Robot` 适配（`make_robot_from_config`），工作量集中在硬件适配与观测/动作特征映射。适合长期想"零自研、全部跟随上游"时做。

### 4.3 度量与验收（可选新增）

- `run_real.py` 或 `ControlActionLog` 增加三个指标：每 episode **队列空转/兜底次数**（现在只有日志）、预取完成时的**平均剩余动作数**、推理延迟分布（server 端 `infer_ms` 已上报 `server_timing`，client 可直接记录）。用于阶段 5 的对照。

---

## 5. 融入计划（分阶段，每阶段可独立合入）

### 阶段 0：基线固化（0.5 天）
- 用现版本跑 N≥10 个 episode，`--log-control` 落盘；写一个小脚本统计：兜底次数、平均队列水位、`server_timing.infer_ms` 分布、实际控制周期抖动。
- 产出基线 JSON，作为后续所有阶段的对照。
- 验收：基线指标文件存在且可复现。

### 阶段 1：服务端语义补齐（路线 A，1–2 天）
- 按 §4.1 改 `websocket_policy_server.py` + `smolvla_policy.py`：latest-only + 相似性过滤（先固定 ε=0 验证链路，再调 ε）。
- `run_real.py` 打通 `--observation-similarity-epsilon`。
- 在 `tests/xtrainer/` 用现有 fake policy/fake environment 测试模式（参照 `test_run_real.py`）补两个用例：并发观测到达时只推理最新；连续相似观测只推理一次。
- 验收：与阶段 0 相比兜底次数下降或持平、成功率不降；协议向后兼容（旧参数照常工作）。

### 阶段 2：RTC 语义融入（路线 B 主投入，3–5 天）
- 先做 §4.2 方案 2a 的最小闭环：server 暴露 `predict_action_chunk` 路径；client 统计延迟并传 `inference_delay`；`prev_chunk_left_over` 先传空、验证纯 inpainting 前缀效果。
- 再做前缀传参：client 把"当前队列剩余动作"作为 `prev_chunk_left_over` 上行（注意只传最近 H 步、且与观测时间戳对齐）。
- 调参：`inference_delay` 窗口大小、soft-mask 的 d/s 比例（参考 RTC 论文 Table 4：真机 H=50、`s_min`≈25）。
- 验收：① 推理延迟 >300ms 时任务成功率较阶段 1 显著提升（对照 RTC 论文结论）；② 无兜底空转；③ 控制频率保持设定值。
- （可选）评估方案 2b：写硬件适配原型，跑通 `lerobot_rollout --inference.type=rtc`，与 2a 横向对比，再决定是否长期切换。

### 阶段 3：工程化收尾（1–2 天）
- 参数文档化（README/docs 同步）；`--observation-similarity-epsilon`、RTC 参数接入 `_validate_args` 与帮助文本；补齐 CI/单测。
- 全量消融：同步（g=0 模拟）vs 阶段 1 vs 阶段 2，在固定任务集上比较成功率、任务耗时、轨迹平滑度（关节加速度能量）、兜底次数。

---

## 6. 风险与注意

1. **观测过时问题（最重要）**：预取推理基于旧观测，chunk 前半段在执行时已"过时"（参考 arXiv:2601.20130 Masked Action Chunking 的论点）。RTC 的 inpainting 只保证与已执行前缀的**运动学连续**，不保证视觉反应最新。若任务需要强视觉反馈（如动态抓取），仍需考虑：缩短 chunk、提高 g、或引入轻量逐 tick 修正（arXiv:2509.23224 A2C2）。
2. **RTC 与绝对/相对动作**：SmolVLA 走绝对动作 + flow matching，RTC 兼容（上游 sync 引擎的 relative-action 限制不适用于 RTC，见 `rollout/inference/sync.py` 注释）。但你的 `select_action` 兼容 LoRA 适配器路径要确认 `predict_action_chunk` 同样兼容 PeftModel 包装（阶段 2 需实测）。
3. **单事件循环 + 长推理**：`asyncio.wait_for(policy.infer(...), timeout)` 下，服务端串行推理期间若超过 `request_timeout`（默认 10s）会取消任务——阶段 1 的 latest-only 能缓解，但真正瓶颈在服务端推理时长，必要时升级 GPU/量化或用 `torch.compile`（fork 的 smolvla 配置含 `use_compile` 相关选项，注意与线程安全）。
4. **g 值选择**：用论文公式 `g ≥ (E[ℓs]/Δt)/n` 粗算。例如 30Hz（Δt=33ms）、n=50、单次推理 2s：E[ℓs]/Δt ≈ 60，阈值已超 1 → 单飞行 + 50 步 chunk 必然周期性兜底，此时优先缩短 `action_horizon` 或压缩推理延迟，而不是调 g。
5. **不要动安全层**：RTC 输出、聚合输出都必须继续经过 `environment.apply_action` 的安全限速/限位，阶段 2 全程保留。
6. **版本漂移**：本仓库内嵌的 `async_inference`/`rollout` 与上游 main 可能有小差异，动手前先 `git diff` 对照；后续可考虑把这两个目录作为上游子模块/定时同步。

---

## 7. 参考链接

- SmolVLA 论文：https://arxiv.org/abs/2506.01844 （§3.3 / Algorithm 1 / §4.6）
- RTC 论文：https://arxiv.org/abs/2506.07339
- LeRobot 上游：https://github.com/huggingface/lerobot
  - `src/lerobot/async_inference/`：https://github.com/huggingface/lerobot/tree/main/src/lerobot/async_inference
  - `examples/tutorial/async-inf/`：https://github.com/huggingface/lerobot/tree/main/examples/tutorial/async-inf
  - `src/lerobot/rollout/inference/rtc.py`：https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/inference/rtc.py
  - `src/lerobot/policies/rtc/`：https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/rtc
- 本仓库对应路径：`src/lerobot/async_inference/`、`src/lerobot/rollout/inference/`、`src/lerobot/policies/rtc/`、`examples/tutorial/async-inf/`
- 延伸阅读：Masked Action Chunking（arXiv:2601.20130）；A2C2 Leave No Observation Behind（arXiv:2509.23224）；VLA-RAIL（arXiv:2512.24673，模型无关的通用异步推理 linker）
