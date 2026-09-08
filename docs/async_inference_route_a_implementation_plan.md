# X-Trainer 路线 A：基于当前代码的具体修改计划

日期：2026-09-08。状态：路线 A 已按本文落地并完成假策略/假环境测试；尚未验证真实模型和真机效果。

本文仅规划路线 A：在现有 WebSocket 部署链中实现观测发送与动作接收解耦、latest-only、相似观测过滤及必要的生命周期管理。不引入 RTC、gRPC、官方 Robot 硬件适配，也不改训练流程。

## 1. 代码核对结论与原计划修正

| 当前代码位置 | 实际行为 | 对修改计划的影响 |
|---|---|---|
| `scripts/xtrainer/run_real.py::_request_action_chunk` | `wait_for(policy.infer(...))`；观测 timestep 只保存在客户端，未传到服务端 | 异步接收必须让服务端原样返回来源 timestep，不能把返回时刻当动作起点 |
| `run_control_loop` | `pending_request` 非空时不再发观测；控制循环继续消费动作 | 已有异步预取，但单飞行请求下没有多个候选观测供服务端覆盖 |
| `_merge_action_queue` | 按 `observation_timestep + index` 对齐，丢弃过去动作，重叠动作按 0.3/0.7 加权；返回新 chunk 覆盖范围内的队列 | 继续复用；过滤通知不能进入此函数，否则可能清空原队列 |
| `deploy/xtrainer/websocket_client_policy.py::request` | 每次 send 后直接 receive，无请求 ID 或统一接收分发 | 不能简单并发调用 `infer()`，否则响应对应关系不成立，也可能并发 receive |
| `websocket_policy_server.py::_handle_websocket` | 每收到一帧，等待完整推理及发送结果后才读取下一帧 | 必须解耦收帧和推理，单独添加容量 1 队列无效 |
| `websocket_policy_server.py::_dispatch` | 直接执行 `infer(payload)`，再判断返回值是否可 await | 同步模型计算会阻塞服务端事件循环，需移到专用串行 worker |
| `smolvla_policy.py::infer` | 已调用 `self.policy.predict_action_chunk(transition)`，随后 postprocess、截断、日志和返回动作 | 原文关于从 `select_action` 切换的描述不符合当前代码；路线 A 无需修改模型调用方式 |
| `smolvla_policy.py::infer` 返回值 | 只有动作字段，没有 `server_timing` | 延迟字段需新增，不能按已有字段编写统计 |
| `run_real.py::parse_args/run` | epsilon 默认 None、帮助称为预留的 12 关节过滤，传入后仅警告无效 | 必须新增参数验证、配置下发和真实过滤行为 |
| `websocket_policy_server.py::stop` | 先 policy.close，再 runner.cleanup | 引入 worker 后必须调整关闭顺序，避免推理仍写动作日志时关闭文件 |

参考本仓库 `src/lerobot/async_inference/policy_server.py`：官方实现把 `SendObservations` 与 `GetActions` 分开，容量 1 队列替换待处理观测，使用 `must_go` 绕过过滤。`helpers.py` 使用状态差的 L2 范数。这里移植这些行为，不直接依赖其 LeRobot Robot 特征映射、gRPC 或 pickle。

**修正实施规模：完整 latest-only 需要同时修改客户端、服务端和控制循环，不能再描述为“仅小改服务端且客户端不动”。** 仍然可以分阶段验证，但是否完成以行为验收为准，不沿用原文未经验证的 1–2 天估计。

## 2. 目标行为与边界

新增模式 `--async-observation-mode latest|legacy`，默认使用 `latest`；`legacy` 保留为现场回退路径。两种模式都保留动作绝对时间步对齐、0.3/0.7 聚合、保持上一实际下发动作的兜底，以及现有限速与环境安全入口。

latest 模式的数据流：

```text
主控制循环：消费动作 → apply_action → 达到预取阈值时采样观测
                              ↓
客户端单发送任务：1 个待发观测槽 → WebSocket
                              ↓
服务端接收：校验/确认接收 → 1 个待推理观测槽（新观测覆盖旧观测）
                              ↓
专用串行 worker：取最新观测 → 过滤判断 → policy.infer
                              ↓
客户端唯一接收任务：按类型分发 → 主循环合并带来源时间步的动作
```

关键区别：允许“一个正在推理的观测 + 一个可被替换的待推理观测”，但同一模型始终最多执行一次推理。只替换尚未开始的工作；正在执行的 GPU 推理不取消，也不因为有更新观测就自动丢弃其结果。

初版只服务一个控制会话。共享 policy 的并行 infer/reset 本身缺乏隔离，latest 会话占用期间，其他连接仍可查看 metadata/health，但不得 infer、reset 或创建另一控制会话；返回明确 busy 错误。所有模式的模型操作共用串行执行入口。

## 3. WebSocket 扩展协议

保留既有 `protocol_version=1`、NumPy MessagePack 格式与 legacy 的 metadata/reset/infer 请求响应。握手新增 `capabilities.async_observation_v1=true`，用能力协商启用扩展；未启用扩展的连接不得收到主动动作消息。

新客户端选择 latest 时，必须先确认能力，且在 `enable_arms()` 前完成协商；旧服务器无能力时明确报错，用户可用 legacy 回退，不静默改变运行语义。

建议新增消息如下，具体字段固定在实现及测试中：

| 方向/类型 | 字段 | 语义 |
|---|---|---|
| C→S `start_async` | `request_id`, `options.observation_similarity_epsilon` | 取得控制会话所有权并设置过滤参数 |
| S→C `async_ready` | `request_id`, `session_id`, 实际生效 options | 确认模式；session_id 由服务端生成 |
| C→S `observation` | `session_id`, `observation_id`, `observation_timestep`, `must_go`, `payload` | payload 仍为 `{state, images, task}`；调度字段留在外层 |
| S→C `observation_ack` | 同 session/id，`status=accepted` | 只表示校验后已接收，不表示已推理或必然有动作 |
| S→C `observation_result` | 同 session/id/timestep，`status=actions`, `payload`, `server_timing` | 成功动作；payload 与当前 infer 返回契约一致 |
| S→C `observation_result` | 同 session/id/timestep，`status=superseded/similar/duplicate/error`，reason | 被覆盖、过滤、重复或失败的终态；不伪造 action、不返回空 chunk |
| C→S `reset` | `request_id`, `session_id` | 清空当前调度状态并串行执行 policy.reset |
| S→C `reset` | `request_id`, 新 `session_id`, `ok=true` | 新 episode 屏障；确认后才接受新会话观测 |

latest 中每次合法 observation 提交先得到 ack，随后得到一个终态（连接断开/reset 失效的情况由会话屏障统一终止）。ack 必须先于同一观测的终态发出。普通 RPC 响应用 request_id 对应，观测事件用 session_id + observation_id 对应。

`observation_id` 在会话内单调递增；`observation_timestep` 是控制步编号，可从 0 开始。相同步骤的重复提交直接回 duplicate；reset 后允许重新从 0 开始。使用最后接受的 ID/timestep 高水位拒绝重复及倒序观测，不维护无限增长的历史集合。must_go 只绕过相似性过滤，不绕过格式校验、会话校验或重复判定。

## 4. 按文件修改内容

### 4.1 `deploy/xtrainer/websocket_policy_server.py`

1. 保留 legacy 分发逻辑，增加协商后的 async 分支。收帧协程只处理解码、边界校验和更新待处理槽，不等待模型推理。
2. 新增专用单 worker 执行入口，例如 `ThreadPoolExecutor(max_workers=1)`：同步 policy.infer/reset/close 在其中运行；已有异步 fake policy 仍可 await，但受同一串行锁保护。所有锁、待推理槽和会话状态在网络事件循环管理，线程只接收独立 payload 并返回结果。
3. 模型 worker 完成当前观测后，从容量 1 槽取最新项。B 被 C 覆盖时，为 B 返回 superseded；不把每条 observation 提交到 executor 的内部任务队列，否则只是把积压搬到另一处。
4. 过滤安排在真正取出候选、准备开始推理时。与最后一次成功推理的原始观测比较；更新比较基准仅在模型成功后进行，失败不能成为后续跳过依据。
5. 推理成功时回传原始 observation_timestep 和原始 ID。不要重写为服务器完成时刻；服务器不计算客户端“现在是第几步”。
6. 加入有界输出队列与唯一发送任务，保证 ack/终态/RPC 帧有序且无并发发送。初版可固定容量 32；若持续背压导致队列满，关闭该控制连接并清理会话，不能无限堆积或默默丢动作。
7. HTTP health 在慢同步推理时仍可响应；第二控制连接不能穿透串行保护修改模型。

### 4.2 新增 `deploy/xtrainer/async_observation.py`

只承载可独立测试的观测数据结构、ID/timestep 校验和相似判断，不实现通用调度框架。WebSocket 会话与任务生命周期仍放在 server/client 中。

过滤规则明确为：

- epsilon 为 None 或 0：关闭相似性过滤；负数、NaN、Infinity：拒绝配置。
- epsilon > 0：计算 12 个关节差的 L2 范数 `norm(new[[0:6,7:13]] - old[[0:6,7:13]]) < epsilon`，单位为弧度；文档中解释这是联合距离，不是每关节独立阈值。
- 夹爪位于索引 6、13，单位是归一化开度，不混入弧度范数。初版要求两夹爪原始 float32 值相同才允许过滤；变化即推理，避免另加未经调定的夹爪阈值。传感器噪声可能降低过滤命中率，这是保守取舍。
- task 变化、无成功基准、首次观测、must_go=true：不作相似跳过。
- state 必须是有限数值 `(14,)`，task/images 等契约必须在过滤前校验；不能让无效 payload 因“相似”而成功跳过。
- 仅根据关节和夹爪判定不能识别“机器人没动但物体动了”。过滤默认关闭；正 epsilon 的真机验收必须包括这种场景，不能把关节相似宣称为视觉相似。

其中夹爪和 task 保护是 X-Trainer 的适配规则，不声称与官方 helper 完全一致。

### 4.3 `deploy/xtrainer/smolvla_policy.py`

保留现有 `predict_action_chunk → postprocessor → actions_per_chunk 截断 → 动作日志` 链路，不修改 SmolVLA/LoRA 模型实现。

将现有 `_validate_payload` 暴露为可复用的校验入口（保留内部调用），供服务端在调度/过滤前验证完整 payload。避免在 server 复制 camera_keys 等模型契约；generic fake policy 可不提供该方法，此时 async 基础字段仍由 transport 验证。

计时先在服务端 worker 外围记录：`queue_wait_ms` 和 `policy_call_ms`，后者包含预处理、模型、后处理及策略日志，不能命名成“纯模型 infer_ms”。不为了路线 A 拆改模型内部代码。

### 4.4 `deploy/xtrainer/websocket_client_policy.py`

1. 维持 legacy 一问一答行为；同一连接进入 async 后，只有一个接收协程调用 ws.receive，RPC 响应和观测终态由该协程分发。
2. 新增 `start_async(...)`、非阻塞观测提交、主循环可取出的观测事件接口。不要用并发 `infer()` 模拟流式提交。
3. 单发送任务维护一个最新待发槽，并采用“上一条收到 observation_ack 后才发下一条”的背压：推理期间仍可连续上报，但客户端不会无限向 socket 推入三路图像。等待 ack 时新采样替换本地尚未发出的观测，不为其创建服务端 Future。
4. 首次观测必须按 timestep=0 发出；空队列恢复期间，待发 must_go 被更新观测覆盖时保留强制标志。已经在推理的强制观测不被取消。
5. 接收事件缓冲有界；主循环每 tick 按消息顺序处理。缓冲溢出报错退出，不把静默丢包当 latest-only。
6. 分开管理 RPC/ack 超时和“没有可用新动作”的进展超时。similar/superseded/ack 不能无限刷新动作等待期限。断线时让所有等待者失败；关闭/重连清理旧任务和会话状态。

### 4.5 `scripts/xtrainer/run_real.py`

1. 添加 `--async-observation-mode legacy|latest`，默认 legacy。epsilon 正值只允许 latest；legacy 配置正 epsilon 明确报参数错误，替换现在的“警告但无效”。
2. 保留现有 `run_control_loop` 为 legacy；新增 `run_async_control_loop`，复用 `_policy_payload`、`_extract_action_chunk`、`_merge_action_queue`、`_rate_limit_action`，避免修改旧模式的单飞行测试语义。
3. latest 首次发送 timestep=0、must_go=true，等待首次 actions 后开始控制时间轴。之后每 tick 先处理动作事件，再消费当前步动作。
4. 正常预取统一使用 `_should_prefetch` 的比例阈值：达到阈值时，每个控制 tick 最多提供一条观测，即使已有推理在运行。执行第 step 步后采集的观测标记 step+1，与现有约定一致；同一步不得同时发送兜底和普通预取两条观测。
5. 收到 actions：先验证非空、有限 `(H,14)`，再用服务端返回的来源 timestep 构造 InferenceResult 并合并。收到 similar/superseded/duplicate：只记录状态，保留现有动作队列。
6. 整个返回 chunk 已经过期时，记录 stale_chunk 并丢弃该结果，保留旧队列中仍有效的动作；不能调用当前 merge 用空结果覆盖原队列，也不能将旧 chunk 重新从当前步起播。
7. 队列耗尽：保持 last_sent_action，并发出最新 must_go 观测；强制恢复需求在获得可用动作前持续存在，通过有界待发槽合并，避免每 tick 增长请求列表。超出 request_timeout 且仍无可用结果时，走现有异常退出/资源关闭流程。
8. 观测读取和图像处理仍通过已有入口，当前它们是同步操作。此次不改硬件线程模型；测量每 tick 采样/图像处理/发送开销，若控制周期验收失败，再将降低观测发布频率作为有证据的后续调整。

`scripts/xtrainer/serve_policy.py` 初版无需新增调度配置：服务端公布能力，过滤参数在会话协商时接收。后续若需要持久化服务器默认值，再扩展 deploy.yaml，避免两套 epsilon 配置冲突。

## 5. reset、断开与停服的明确顺序

reset：立即标记旧 session 失效 → 暂停接收新观测 → 清空待发/待推理槽及过滤基准 → 等当前模型调用真正结束（结果不再发布）→ 同一 worker 执行 policy.reset → 发回新 session_id → 接收新观测。

断开：失效会话、停止收发任务、释放待处理数据；当前同步模型调用自然结束前，不把 policy 所有权交给另一个会话。

停服：停止接受新的模型工作 → 失效会话、清理网络任务 → 等 worker 当前任务结束 → 串行调用 policy.close → 关闭 executor/runner。取消 asyncio Future 不代表底层线程或 GPU 已停止，不能据此提前 reset/close。

若底层模型调用永久卡住，线程方案不能强制终止它；客户端超时关闭控制资源，服务端记录需重启。独立推理进程和自动恢复不纳入本轮。

## 6. 测试计划与实施顺序

按下列四步提交，每一步具备独立检查项；最终完成标准是四步合起来可跑通路线 A，而不是仅有一个容量 1 队列。

| 步骤 | 文件 | 必须验证的行为 |
|---|---|---|
| 1. 协议与模型串行入口 | server/client、transport tests | legacy health/metadata/reset/infer 回归；能力协商；慢同步 fake 推理时 health/收帧仍可运行；模型并发数始终 1 |
| 2. latest-only 与生命周期 | server、async_observation、新增 async transport tests | A 推理被测试闸门阻塞时提交 B、C，释放后仅推理 A、C，B 收 superseded；reset 后旧动作不发布；断开/停服无悬空任务；第二控制会话 busy |
| 3. 控制闭环与日志 | client/run_real、run_real tests、新增 async loop tests | 初始动作时间步；延迟 chunk 丢弃过去步；0.3/0.7 聚合；被过滤保留队列；空队列 must_go；无可用动作超时释放资源；背压缓冲有界 |
| 4. 相似过滤与回归 | async_observation、policy validation、tests、README | None/0 禁用；epsilon 边界及非法值；夹爪/task 变化放行；must_go 放行；失败不更新基准；reset 清基准；畸形图像不能被过滤掩盖 |

新增建议测试文件：`tests/xtrainer/test_async_observation.py`（纯状态/过滤），`test_async_websocket_transport.py`（协议/线程/生命周期），`test_run_real_async.py`（控制闭环）。沿用现有 fake policy/fake environment，不连接真机或下载权重。并发测试用 threading.Event/asyncio.Event 控制先后顺序，避免依赖脆弱的固定 sleep。

现有 `test_control_loop_blends_returned_prefetch_and_keeps_one_request_in_flight` 保留给 legacy；latest 的约束是“模型执行并发数为 1”，不是“全链路只存在一条观测”。现有 CLI 预留参数测试需更新，新模式未启用时的默认值仍有覆盖。

已执行的针对性检查命令：

```powershell
python -m pytest --confcutdir=tests/xtrainer tests/xtrainer/test_async_observation.py tests/xtrainer/test_async_websocket_transport.py tests/xtrainer/test_run_real_async.py tests/xtrainer/test_websocket_transport.py tests/xtrainer/test_run_real.py tests/xtrainer/test_deploy_e2e.py -q
git diff --check
```

上述范围共 37 项测试通过。`test_smolvla_policy.py` 和 `test_serve_policy_cli.py` 的收集被当前环境缺少
`draccus` 阻塞，因此模型包装层的改动只完成了编译检查，尚未声称测试通过。

## 7. 日志、验收与回退

扩展现有 ControlActionLog，记录 session/id/timestep、观测采样到收到结果的客户端单调时钟耗时、queue_wait_ms/policy_call_ms、similar/superseded/stale_chunk 计数、动作队列水位、fallback 步数、控制 tick 间隔。网络两端不直接相减各自 monotonic 时间，客户端耗时使用自己保存的发送记录计算。

验证分三个配置：legacy 基线、latest+epsilon=0、latest+正 epsilon。先用假延迟集成测试检查上述确定性行为，再在人工授权的真机测试中用同任务、相同模型、相同控制频率和 horizon 比较。每组建议至少 10 次 episode 作为初步回归，不能据此承诺统计显著提升。

必须满足：

- 工程正确性：无动作来源错位、无重复执行旧步、无过滤导致清空队列、无 reset 跨会话污染；模型并发数为 1，缓冲不随运行时间无限增长。
- 控制性能：相同条件下记录 tick 周期 P50/P95/max、兜底比例和任务成功次数；若最新观测发送导致控制明显超期，不能标记为通过。真实阈值应结合基线实测制定。
- 相似过滤：确实减少静态场景下模型调用；夹爪/task 变化和强制恢复可以继续获得动作；机器人不动但目标物移动的场景必须单独检查。
- 算力边界：latest-only 减少的是待处理观测陈旧程度，不降低单次模型计算时长，不承诺兜底归零。chunk 可执行时长 `H / control_hz` 若小于推理耗时，仅调阈值无法解决；原文“优先缩短 action_horizon”的建议不能作为这一问题的通用修复，缩短反而减少可覆盖时间。

回退方式：使用 `--async-observation-mode legacy`，关闭正 epsilon，沿用原控制链。latest 模式的成功标准不包含 RTC 的前缀生成约束，也不声称能保证视觉观测最新或动作绝对连续。

## 8. 本次交付范围

本次按本文完成路线 A 的协议、服务端调度、客户端收发、默认 latest 控制循环和对应测试。原 `async_inference_integration_plan.md` 保留作调研记录；路线 A 的实现与验收以本文为准。
