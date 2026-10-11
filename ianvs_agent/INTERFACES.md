# Ianvs Agent MVP v0.1 内部接口

全部 12 层进入验收。主模型 `ark-code-latest`，手动切换模型 `glm-5.3`。界面及自然语言提示使用中文，identifier 保持英文。代码修改使用 apply_patch；不得使用 Mock、替代推理、导入 fallback 或手写成熟格式 parser。

## 公共约定

- 各层业务记录使用可 JSON 序列化的 dict；Path 在保存前转换为字符串。
- `AgentSettings` 及 `ModelProfile` 位于 `ianvs_agent.config`，实例化设置时不调用模型。
- 身份字段：`instance_id`、`environment_id`、`session_id`、`task_id`、`run_id`。
- `LocalStore(root)`：`read(relative_path, default=None)`、`write(relative_path, dict)`、`append(relative_path, dict)`、`read_events(relative_path)`、`allocate_id(prefix)`；只写入 root 范围；同目录原子快照与 filelock。
- Linux 的 Journal 追加缓存最多 128 份文件元信息及末尾序号，通过设备、inode、大小、mtime_ns、ctime_ns 判断是否可复用。首次访问、外部修改和重启执行完整 JSONL 校验；Windows 始终完整校验。文件锁、fsync、尾部中断恢复与完整记录损坏检查保留。
- `EventBus(store, session_id=None, instance_id=None, task_id=None)`：`emit(event_type, payload, **identities)` 先持久化再调用订阅者；`subscribe(callback)`；所有事件含递增 sequence 和 UTC 时间。
- 长输出在任务目录保存全文，模型获得预览、引用、截断范围和继续读取方式。
- `run_example`、`verify_run` 的公开结果包含 `outcome`，分别保存 `run_id,run_status,exit_code,verification_passed,verification_stale,repair_completed,full_run_ref`。正文截断后仍保留该概要；`status` 表示工具调用状态，运行和验证结果由正式返回值提供。

## Provider / Usage / Context

- `Provider(profile, events, usage)`；`async complete(messages, tools=None, purpose="plan", retry_reason=None, *, max_attempts=None) -> dict`。`max_attempts` 为可选正整数，实际尝试上限为 `min(profile.max_retries + 1, max_attempts)`；默认沿用模型配置，非法值在请求前拒绝。
- 响应：`content`、`tool_calls`（`id,name,arguments`，arguments 为 dict）、`usage`、`model`、`request_id`、`finish_reason`；原始消息另存 `raw_message`。
- Provider 使用 SDK 流式接收并聚合完整响应，发布 `llm.delta`、`llm.progress` 和 `llm.retry`；仅可见回复进入界面。`connecting/receiving` 阶段的明确连接、读取、写入、远端协议故障、请求总超时及 HTTP 408/409/429/5xx 可在配置次数内重试。失败记录保存 `retryable`，最终异常附带正式 `request_failure`；取消、认证、格式、上下文超限和存储错误分别处理。每次重试创建独立聚合状态，完整响应通过检查后才交付工具调用，原始失败片段与用量继续保留。
- `context_input_budget` 默认 `None`（YAML `null`），输入空间按当前模型窗口减本次输出预留计算；显式正整数只用于缩小空间。自动压缩阈值为模型窗口扣除输出预留后的 78%，上限为显式输入预算。`auxiliary_output_tokens` 默认 2048。Observe 只发送当前完整工具交互及正式任务、状态与证据引用。主模型的辅助请求使用已核实支持的 `auxiliary_thinking: disabled`，GLM 保持服务默认配置。`max_input_tokens` 用于周期压缩；单次输入空间仍按模型配置检查。
- `UsageTracker(store, session_relative_path, events)`：`record(record)`、`summary()`、`check_budget(settings)`；实际用量未知保持 null，估算另行标注。
- `ContextManager(settings, store, events, provider)`；`async prepare(task, state, history, tools, role="plan", force_compact=False, correction=None) -> dict`，返回 `messages,tools,estimated_tokens,context_fraction,checkpoint_id`；`async compact(task,state,history,tools,manual=False)`。`correction` 仅用于 Observe，并在完整请求估算之前加入。
- `async complete(task,state,history,tools,role="plan",purpose="plan",force_compact=False,correction=None,retry_reason=None,max_attempts=None,allow_text=False)` 负责准备请求、Plan 周期压缩、调用模型及服务明确报告上下文超限后的单次压缩重试；Observe 格式纠正请求遇到服务上下文超限时直接失败。`prepare(...,allow_text=False)` 同样支持该参数；纯文字 Plan 后使用 Observe 的 `allow_text=True`，只发送当前末尾完整交互与正式状态，模型判断继续或现实阻塞。
- Plan 在 `task_effective_input_tokens - budget_base_input_tokens + estimated_tokens >= max_input_tokens * context_threshold` 时检查周期压缩，默认参考量500000、比例78%。基准来自上次成功检查点的 `task_input_tokens_at_completion`；旧检查点依据 `model_requests` 对应的真实 Usage 恢复基准。候选摘要须有完整旧交互、预计节省后续输入且符合模型单次输入窗口。后续 Plan/Observe 的估算继续保存供统计，不构成累计额度门禁。摘要调用使用 `purpose="compaction",max_attempts=1`，完成后重建 Plan。
- 输入周期压缩记录保存于会话的 `context/input-budget/<task_id>.json`。实际摘要前保存 `status=attempted`，完成后更新为 `completed` 并保存检查点。摘要请求的正式失败记录标记 `retryable=true` 且已准备 Plan 仍符合输入窗口时，保存 `status=deferred`、失败请求引用和失败后的累计用量，发布 `context.compaction_deferred`，继续该 Plan。请求历史 fingerprint 与当前检查点共同限制相同历史的重复摘要，新增完整历史可以再次触发。检查点与事件带 `trigger="task_input_budget"`，手动压缩和窗口压缩独立处理。
- 预检无法压缩时保存 `status=skipped` 与原因，相同历史、检查点及已用量不重复检查，相同原因不重复发布 `context.compaction_skipped`。新增完整历史可以重新检查；无法节省时继续当前任务。工具发送范围按模型单次窗口限定，保留配对 ID 与全文证据引用，范围选择不产生摘要或压缩事件。
- `compact(..., manual=True)` 没有可压缩的旧交互时返回 `status=skipped,reason=no_older_history,message`，发送一次 `context.compaction_skipped`，不改正式任务与验证结果、不调用模型。`prepare(...,force_compact=True)` 没有旧交互时继续收缩工具证据预览，以原请求估算的 68% 和窗口阈值的 68% 中较小者作为收缩目标；保留全文引用、配对 ID 和正式目标，重新估算后仍超出模型单次输入空间则明确报告失败。Harness 消费排队请求后清除标志。
- 用量同时保存当前 task 与累计 session 的实际使用量；模型、工具和压缩记录带当前正式 `task_id`。`UsageTracker.check_budget()` 保留既有调用参数并返回真实统计，累计输入、输出、模型次数与工具次数不拒绝执行。`max_input_tokens=500000` 用于压缩周期，其余历史累计配置字段保留作统计参考。摘要和恢复均不重置已用量；任务累计活动时间与运行尝试次数同样只记录。
- history 是 OpenAI 格式消息列表；切分保留 tool_calls 与对应结果完整配对。正式 task/state 的发送视图始终重新注入，压缩历史摘要不覆盖正式记录。
- STATE 视图选取最早五项、最近十五项 facts 以及最近十项 hypotheses，其他程序字段完整保留。TASK 的 `acceptance.file_identities` 超过二十四项时选取最早八项及 `repair_progress.files_read` 中已有路径；其他任务与验收字段完整保留。视图附 `context_selection` 数量与全文引用；全文保存于会话 `tool-output/STATE-<task_id>.json` 和 `TASK-<task_id>.json`。选择范围不调用模型或产生压缩记录，Checkpoint 仍保存完整输入 task/state；全部身份约束继续交给独立 Verifier 核对。
- 每次请求从当前 `instance.json` 重新读取环境摘要，包含环境身份、完成时间、解释器、包数量、准备步骤及证据。历史 Run 的环境字段只表示当时的执行条件。
- Plan 可调用工具；Observe 不开放工具，返回 facts/hypotheses/next_goal/decision 的 JSON。权限、patch、Run 与 verification 状态由程序更新。
- Observe 的标准 JSON 解析或字段检查失败后，保存 `observation.failed` 及原始请求引用，最多追加一次 `retry_reason=observation_format,max_attempts=1` 的格式纠正请求。纠正用量与活动时间累计记入原任务统计，已完成工具不重复执行；合法响应通过检查后才更新调查事实，第二次格式失败明确停止。

## Runtime

- `ExampleCatalog(project_root).list_examples() -> list[dict]`：`benchmark_path,name,readme_path,inventory_entry`。
- `InstanceManager(settings,store,events)`：`create(benchmark_path,name=None) -> dict`；`list_instances()`；`get(instance_id)`；`check_changes(instance_id)`；记录 `code_root,benchmark_path,readme_path,environment,status,verification`。
- `check_changes()` 的包清单检查与 Verifier 共用 `runtime_environment()`，清除控制进程的 `PYTHONPATH/PYTHONHOME` 后使用实例解释器，按正式包名称和版本核对环境变化。
- `EnvironmentManager(settings,store,events)`；`async prepare(instance,install=True,plan_id=None) -> dict`；实例独立 venv，解释器实际完整路径，原有准备脚本按声明执行。`preview_prepare(instance)` 生成稳定准备计划，`revalidate(instance,plan_id)` 核查实例、环境代次和输入文件版本。
- `async inspect(instance,requirements_files=None,modules=None)` 保存真实解释器、包版本、pip check、模块导入及依赖文件版本。`async plan_install(instance,requirements_files=None,packages=None,constraints_files=None,index_url=None,timeout=1800,modules=None)` 保存安装输入、计划与审批依据；`async install(instance,plan_id)` 创建候选环境，执行安装及检查后激活，返回 `postcheck`。`recover(instance)` 核查所属进程和中断记录，`async cancel(instance=None)` 终止所属操作。
- `EnvironmentCommands(settings,store,events)`；`preview(instance,argv,cwd=".",timeout=300,writable_directories=None,network=False)` 返回具体审批内容；`async execute(...,approved=None)` 核查审批内容并执行，保存独立命令记录。源码和现有环境只读，生成目录及网络权限显式审批。
- 环境辅助执行使用 Linux Bubblewrap。依赖解析使用目标解释器的 pip，保存递归 requirements/constraints 与本地依赖制品版本；控制程序正式记录不挂载为可写目录。
- pip 在真实解析阶段保存 `resolution_ref,constraints_ref`，审批绑定解析结果及制品版本。候选环境仅在安装与检查通过后激活；原环境处于 `ready` 或 `modified` 时，候选失败、取消或中断均保留原环境身份和异常状态。预检、进程启动和命令执行通过 `phase,command_executed,exit_code` 区分，失败保存正式日志引用。
- 依赖计划保存 `pip_cache_dir,pip_cache_ref`，pip 解析与安装显式绑定当前实例缓存，审批包括实际 `--cache-dir` 命令。依赖声明的 loopback HTTP `find-links` 可登记确切主机及端口的 `--trusted-host`，记录在 `cache_trusted_hosts`；其他 HTTP 来源不会自动扩展信任。旧 prepare 计划缺少缓存依据时保留原命令并失效，新计划重新审批。
- Environment 保存 `summary,warnings,preparation_scope,steps,failed_step`。准备步骤发送 `environment.step_started/completed/failed`，失败保留 `error_tail,log_ref`；复用环境补充说明时保持已有验证判定。
- `Runner(settings,store,events)`；`async run(instance,overrides=None) -> dict`；`async cancel()`；`get_run(instance_id,run_id)`；绑定独立 cwd/解释器/workspace，事件通道区别于 stdout/stderr。
- 若子进程在采集创建时间之前已结束，`process_created_at` 保持 null，继续读取真实退出码和 stdout/stderr；恢复时不把缺少身份的 PID 判为所属活动进程。
- 新 Run 启动前若仍缓存重启前的所属进程，重新核查 PID、创建时间与所属身份；存活时保持运行互斥，已退出时将旧 Run 保存为 interrupted、退出码保持未知，允许启动新 Run。
- Run 保存 `run_id,instance_id,environment_id,status,exit_code,timed_out,interrupted,started_at,completed_at,command,cwd,output_dir,stdout_ref,stderr_ref,events,testcases,metrics,artifacts,config_snapshot`，未知字段为 null。`execution_configuration` 保存入口和实际启动环境中的 `mock_runtime_enabled,pythonpath_override`，来源类型为 `runner_launch_environment`；该字段说明启动配置，模型正确性仍由执行与独立检查证据判断。
- `Verifier(settings,store,events)`；`async verify(instance,run,acceptance=None) -> dict`；`async validate(instance) -> dict`。判定 `passed,execution_completed,results_complete,checks_passed,goal_achieved,repair_completed,checks,reasons,evidence_type`；必需检查 SKIP、旧 CSV、缺失本轮指标不能通过。
- `validate()` 的检查附 `source_kind=static_validation` 及本次 `report.json` 引用；通过该来源明确区分静态检查材料与实际 Run。验证条件保持独立。
- `validate()` 使用 Agent 控制解释器，允许检查尚未准备或依赖异常的实例。`verify()` 另外检查真实目标环境的 `pip check`、当前与本次 Run 的准备状态；静态检查通过不能代替原任务和环境通过。
- core 增加可选轻量事件回调，普通 CLI 保持兼容；`--agent` 延迟导入 TUI，且与 `-f` 互斥。

## Harness / Tools / TUI

- `AgentHarness(settings)`：`.catalog,.instances,.environment,.runner,.verifier,.store,.events,.usage,.provider,.context,.session,.current_instance,.busy`。
- `select_instance(instance_id)` 用于初始化；`async restore_instance(instance_id)` 用于界面切换；`async create_instance(benchmark_path)`；`async prepare_instance()`；`async run_current()`；`async send_message(text)`；`async compact()`；`async cancel()`；`switch_model(profile_name)`；`async approve(approval_id,approved)`；`history()`；`diff()`；`usage_summary()`；`verification_report()`；`async close()`。
- Harness 负责浏览零模型调用；实例选定后的首次用户任务才调用 Provider；会话与正式任务持续保存。
- 已结束的独立 Run 开始新修复时，创建新的 task_id，采用用户修复目标并继承原验收。进行中的修复保留原目标、验收及用量。`run_current()` 保存本次独立运行的结束状态；工具 `run_example` 仍允许循环依据新的失败继续修复。
- 每轮工具结束后，Harness 从正式工具记录更新 `state.repair_progress`：`phase,tool_rounds,rounds_without_change,files_read,recent_actions,last_run_id,updated_at`。已读文件保留版本、范围、截断状态及证据引用，存在冻结身份时附比较结果。Observe 仅添加或修正事实与假设，不能覆盖该程序记录。
- 文件工具使用 UTF-8/BOM 编码识别与严格解码；`read_file.ends_with_newline` 表示完整文件的末尾换行状态，分页时保持相同。搜索保留明确范围、编码、跳过原因及分页信息，`limit` 与 Schema 同为 1～1000。读取错误保持失败。`read_evidence` 开放当前实例正式登记的验证、环境计划、检查、操作、pip 报告和执行日志，以及允许的会话证据引用。
- `read_file` 对目录及其他非普通文件返回统一工具错误，保存 `tools.jsonl` 与 `tool.failed`，模型可以继续选择工具；缺失文件保留 `file_version=absent`，供新文件补丁绑定来源。
- `apply_patch` 工具只接受 `patch`；程序从当前实例、会话和任务的正式 `read_file` 结果绑定文件版本及证据引用。新文件需先读取并确认不存在。审批前后核对绑定版本，GNU patch 在 Bubblewrap 的 staging 目录先预检再应用；统一 CRLF 文件使用标准文本 I/O 转换 staging 换行，提交时保留原换行样式与末尾状态。原始 `input.diff` 全文与 GNU 使用的 LF 副本 `execution.diff` 分别保存，两次操作和变更记录包含 `execution_input_ref`，副本只读挂载。提交前再次读取原始字节核对版本，完整检查通过后更新实例文件与验证状态。
- Harness 结束任务保存 `state.result`，包含完整事实、假设、下一步、验证状态及可读正文；诊断结果通过 `agent.message(role=result)` 与终止事件发送。只读对话的最终回复保持一次展示。
- TUI 位于 `ianvs_agent.app.tui`，`run_app(settings,harness=None)` 为启动函数；只消费 events 或读取 Harness 数据；后台任务使用 Textual Worker。
- `approval.requested` 事件 payload 为 `approval_id,tool_name,arguments,description,diff`；存在动态操作时附 `resolved_operation`，审批前后重新核查具体计划、环境和文件版本，handler 接收 `_approved`。审批决定绑定具体参数及修改依据。
- 新工具为 `inspect_environment`、`install_dependencies`、`execute_environment_command`、`configure_runtime_environment`。安装 `plan` 只生成记录，`apply` 仅接受 `action,plan_id`，依照已审批计划执行。凭据工具仅保存 YAML 允许的环境变量名称引用，Runner 读取值后注入当前 Example，模型、工具返回和事件仅获得名称及可用状态。
- Harness `async resume_current(text=None)` 保留当前 task_id、目标、验收、历史、累计用量与活动时间；`/resume`、`/继续` 通过 RPC 调用。持久化 `active_execution_seconds` 累计活动时间，关闭期间不计入。累计 Token、模型和工具调用次数、运行尝试次数及活动时间均不终止任务，也不阻止失败、阻塞、取消或中断任务恢复。
- repair 完成要求正式独立验证中的 `passed=true,repair_completed=true`，验证未失效、实例无需重验，且任务身份与当前最新 Run 引用一致；模型文字答复不能代替此判定。maintenance 完成需要当前实例、任务、环境代次和文件版本一致的正式环境检查。Observe 根据证据判断现实阻塞；能够继续安装、修正或重试时继续执行。Runner 单次超时、用户取消和安全审批仍保留各自的实际结果。
- TUI 中文区域为“示例与实例”“对话与工具记录”“运行与验证”“上下文与用量”；模型名、工具名及路径保持原名。显示工具、模型和验证的真实状态；Run、Diff、Usage、Compact 快捷键可用。
- TUI 恢复旧记录时读取正式诊断与环境事件，不产生模型调用；`ReflowRichLog` 保存原始文本，在页面可见及尺寸变化后重新排版。
- 高频事件置脏并由 100 毫秒定时器合并刷新；审批及终态立即刷新。运行输出按 Run/通道维护未完成行，回车更新当前进度，完整行进入 RichLog，终态冲刷剩余输出。显示状态可依据正式 `outcome` 标记 error，保留原工具调用状态。继续按钮及 F7 仅在已有可恢复任务、连接正常且空闲时调用 `resume_current()`。
- 测试使用真实文件、进程、App/Pilot、Provider；禁止 Mock。缺少模型授权或设备应保存未验证原因。

## Windows 前端与 WSL 执行端

- Windows 默认由 CMD 启动本机 Textual TUI，WSL 启动 `python -u -m ianvs_agent.transport.server`；Linux 本机支持直接使用 `AgentHarness`。
- `RemoteHarness(settings,command,backend_label="WSL")` 构造只读缓存和事件订阅，`async start()` 创建真实后台进程并加载快照。同步读取兼容 TUI 的 catalog、instances、history、store.read_events、runner.get_run、diff、usage_summary 和 verification_report；执行动作通过异步 RPC，`switch_model` 在远程接口中为异步方法。
- JSONL 协议 v2 有请求 ID、响应、正式事件与递增快照版本；固定允许方法及参数，单帧最大 16 MiB。更大的完整消息以 1 MiB 内容片段编码，每片带 message_id/index/count；同一消息连续发送，身份、顺序及长度通过检查后原子发布。连接中断丢弃未完成片段。后端 stdout 仅用于通信，诊断信息进入 stderr。
- 初次连接或切换会话发送 `replace=true` 完整快照；其他快照只发送变化字段，对话以 `history_delta={start,items}` 替换变化后的尾部。Example 列表在当前连接内缓存，持续输出与模型进度直接发送事件。RPC 结束前等待已产生的事件入队并发布，再同步终态与响应，确保界面能退出忙碌状态。
- 大请求在发送中收到取消时，先完成当前消息的全部分片，再发送取消操作，避免交错消息。事件与输出队列在关闭等待中同时监控消费任务，消费异常立即传播。完整恢复仍保留全部历史，所需内存随会话增长。
- 后端管理全部正式记录，影响正式状态的事件先同步快照再发布；流式回复及请求进度直接发布，避免重复传输实例文件清单。客户端显示真实 backend_info、连接状态和错误；连接中断使未完成请求明确失败。
- 后端并行接收审批、取消和压缩控制请求；关闭或 stdin EOF 时处理所属活动任务并保存会话。前端不加载凭据、不改写源码或验证记录。
- 直接 `execute_tool` 请求在参数校验后建立或沿用正式任务，工具记录和补丁均关联任务身份；批准后真实修改，拒绝或取消保持文件原内容。
