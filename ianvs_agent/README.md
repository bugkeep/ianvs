# Ianvs Agent MVP v0.1

Ianvs Agent 通过中文 Textual 界面浏览 Example、创建独立源码与 Python 环境、调用模型调查错误、审批修改并运行原始评测。修复完成由独立验证器根据本轮执行、TestCase、产物、原指标和正式验收要求判定。

v0.1 包含 TUI、Session Harness、Provider、Context、Usage、Tools、诊断 Skill、Instance、Environment、Runner、Verifier、Persistent Store 和文件维护检查。主模型配置为 `ark-code-latest`，可手动切换至同一方舟账户的 `glm-5.3`。

## 安装和启动

Agent 使用 Python 3.11 或更新版本。Windows 默认从 CMD 启动 Windows Python 中的 Textual TUI，执行端使用 WSL `Ubuntu-24.04`。Agent Harness、模型请求、上下文、工具、实例环境、Ianvs Runner、验证器和正式记录都由 Linux 后端管理。Windows 前端只提交用户动作、显示正式状态和事件。Linux 本机与 SSH 终端可以直接运行完整 Agent。

在 Windows CMD 中安装界面依赖和命令入口：

```bat
set PYTHONUTF8=1
python -m venv workspace\ianvs-agent\windows-control
workspace\ianvs-agent\windows-control\Scripts\python.exe -m pip install -r ianvs_agent/frontend-requirements.txt
workspace\ianvs-agent\windows-control\Scripts\python.exe -m pip install --no-deps -e .
```

安装会生成 `workspace\ianvs-agent\windows-control\Scripts\ianvs.exe`。在当前 CMD 中激活界面环境后，可直接使用 `ianvs`：

```bat
workspace\ianvs-agent\windows-control\Scripts\activate.bat
```

在 WSL 的 Ianvs 源码根目录创建后端控制环境并安装全部 Agent 依赖：

```bash
python3 -m venv workspace/agent-control
workspace/agent-control/bin/python -m pip install -r ianvs_agent/requirements.txt
workspace/agent-control/bin/python -c "import sys; print(sys.executable)"
```

Linux 执行端需要 `bubblewrap`（命令 `bwrap`）与 GNU `patch`。环境检查、依赖解析、安装和辅助命令使用独立进程及文件权限限制；补丁在独立目录先检查再应用，保留文件末尾换行、统一 CRLF 换行及原始补丁。缺少所需组件时明确停止操作并说明原因。在 Ubuntu 中可通过系统包管理器安装 `bubblewrap` 和 `patch`。

回到 Windows CMD 启动界面，`--backend-python` 使用上一步解释器的 Linux 绝对路径：

```bat
workspace\ianvs-agent\windows-control\Scripts\python.exe -X utf8 -m ianvs_agent --backend wsl --wsl-distro Ubuntu-24.04 --backend-python /path/to/agent-control/bin/python
```

将上一步打印的 Linux 解释器路径填写到配置文件的 `wsl_python_executable`；配置示例见 [config.example.yaml](config.example.yaml)。将示例保存到 `workspace/ianvs-agent/config.yaml`，同时将 `state_root` 设置为 WSL 用户目录中的独立目录，例如 `/home/<用户名>/ianvs-agent-state`。实例源码、venv 和大量日志放在 Linux 文件系统中，能够减少跨系统文件访问的等待。

激活界面环境后，在 Ianvs 源码根目录的 CMD 中直接启动：

```bat
ianvs --agent
```

命令从当前 Ianvs 目录读取配置、连接 WSL 后端，并恢复正式状态目录中保存的会话与实例。以上安装步骤中的 `PYTHONUTF8=1` 适用于当前 CMD 会话。

Linux 终端直接启动本机执行端：

```bash
workspace/agent-control/bin/python -m ianvs_agent --backend local
```

Linux 终端安装完整 Agent 可选依赖后，也可使用统一 CLI：

```bash
python3 -m pip install -e '.[agent]'
ianvs --agent
```

普通 `ianvs -f <benchmarkingjob.yaml>` 保持原用途。`--agent` 与 `-f` 互斥。Agent 模块按需导入，普通 CLI 展示版本不要求 Agent 或评测依赖已安装。

模型凭据由后端的 `ARK_API_KEY` 提供，也可放入被忽略的 `workspace/ianvs-agent/.env`，仅后端加载此文件：

```dotenv
ARK_API_KEY=填写自己的账户凭据
```

启动与浏览不会请求 LLM。只有提交自然语言任务或执行上下文压缩时调用模型。独立源码副本与评测子进程不会获得方舟凭据。

## 操作

左侧筛选 Example，创建或恢复实例，准备环境。实例保留原始 benchmark 路径、来源身份和独立工作目录。缺失配置会显示“配置待修复”，已准备环境会显示真实解释器与环境状态。

输入“分析当前错误”“解释这个 Example”或“修复当前 Example，保持原任务并重新运行”。普通对话的纯读取工具结果直接交给模型生成答复，减少一次额外请求；修复流程保留 Plan、工具执行和 Observe。Observe 明确禁止工具调用，并只更新带证据引用的调查事实、假设与下一步；正式 Run、补丁和验证状态由程序更新。Observe 返回的 JSON 或字段不合法时，保存原始响应并最多请求一次格式纠正，继续累计原任务时间与 Token；再次失败明确停止，已有工具不重复执行。

独立运行结束后开始修复，会建立包含用户修复目标的新任务并继承原验收条件。进行中的修复继续使用原任务及累计用量记录。程序记录已读取的文件版本、范围、近期工具动作及修复阶段；直接证据充分时，模型提交版本绑定补丁，随后执行原任务。README 与静态检查提供调查材料，实际执行方式以本轮 Run 和环境记录为准。

界面底部持续显示当前阶段、等待动画与累计时间；模型回复逐步显示。工具按调用合并到独立列表，可查看参数、结果摘要、错误及完整证据引用；内部事件和评测输出显示在日志页，切换页面与调整窗口后重新排版。普通对话取消或失败后、运行任务失败后输入分析问题，会建立新的对话任务，旧任务记录保持保存；未完成的修复任务继续保持原始目标和验收条件。

诊断结束或任务阻塞时，对话中显示具体原因、证据引用、待核实内容、下一步与独立验证状态。较长记录展示最多六项证据和三项待核实内容，并提供完整 `state.json` 引用；正式记录保留全部内容。已有会话可以恢复显示诊断结论，无需重新调用模型。

| 操作 | 按键或指令 |
| --- | --- |
| 创建实例 | `Ctrl+N` |
| 准备环境 | 界面“准备环境”按钮 |
| 执行原任务 | `F2`、`/run` |
| 查看 Diff | `F3`、`/diff` |
| 查看用量 | `F4`、`/usage` |
| 查看独立验证 | `F5`、`/verify` |
| 查看工具记录 | `F6` |
| 压缩上下文 | `Ctrl+K`、`/compact` |
| 取消当前任务 | `Ctrl+X`、`/cancel` |
| 继续原任务 | 界面“继续原任务”按钮、`F7`、`/resume`、`/继续` |
| 保存并退出 | `Ctrl+Q` |

运行期间可以查看其他视图。压缩请求在完整工具交互结束后处理；工具调用与对应结果一起保留。取消会处理当前工具审批和所属评测进程，保留已经取得的材料。

任务失败、阻塞、取消或中断后，连接正常且后台空闲时可点击“继续原任务”，保留原任务身份、验收条件、历史与累计用量。工具页把调用状态与 Example 执行、独立验证结果分开展示；调用正常返回但运行或验证失败时显示 `error`，结果正文被截断也保留 Run 身份、退出码与判定概要。完整结果继续保存在后端正式记录中。

高频事件按约 100 毫秒合并界面刷新，审批、任务结束和连接异常及时显示。运行输出按 Run 和 stdout/stderr 分别拼接，未结束的进度行在当前输出区更新，完成行进入日志页；原始输出与事件逐条保存。

对话页和日志页重新显示时，按照当前可用宽度处理隐藏期间收到的内容；即使终端尺寸没有变化，也会刷新最后一行输出。

安装任务支持“按照当前 YAML 准备环境，核查 Example 的 requirements 并补齐缺少的依赖”等自然语言指令。Agent 读取声明和日志，使用 `inspect_environment` 检查实际解释器、包版本、`pip check` 与模块导入；通过 `install_dependencies(action="plan")` 保存具体安装方案，再用 `action="apply", plan_id=...` 请求审批并执行。批准的方案绑定当前实例、环境代次、源码及依赖文件版本和实际命令。

安装在当前实例的新 venv 中进行，成功检查后激活新环境；安装失败、取消或中断时保留候选环境和日志，当前原有环境继续保留，已有的依赖异常也保持原状态。新增 requirements 经过审批后作为该实例的依赖方案持久保存，后续准备沿用它。辅助命令通过 `execute_environment_command` 使用当前实例 Python，源码、验证程序和已安装环境只读，生成文件需要指定并批准新的实例内目录，网络权限单独审批。系统依赖或权限不足会明确报告阻塞。

pip 缓存属于当前实例，解析、安装与后续计划复用同一目录，实际命令及缓存引用写入正式计划。其他实例使用自己的目录；包版本与制品校验继续执行。本机 HTTP 依赖来源仅为确切的 loopback 主机及端口登记 pip 缓存适配参数，参数随安装计划一起审批。

环境检查通过显示“环境准备成功”。原目标要求修复 Example 时，Agent 仍需执行原任务并取得独立验证通过，才显示“修复完成”。依赖变化会使历史验证失效。取消、请求异常或关闭后使用 `/resume`，继续原任务目标、验收条件和累计 Token 记录；任务时间累计活动执行时间，关闭期间不计入。

默认 `approval_policy: manual`。修改、环境准备和模型发起的运行会显示具体动作、参数与 Diff。审批只对应当前参数与文件版本。受限的 `pwd`、`ls`、`stat` 属于只读调查，仍检查命令选项与实例路径。`deny` 拒绝修改动作；`auto_scoped` 用于明确配置的自动实验，仍执行范围与版本检查并记录固定策略决定。

## 配置

默认读取 `workspace/ianvs-agent/config.yaml`，也可使用 `python -m ianvs_agent --config <配置文件>` 或 `ianvs --agent --agent-config <配置文件>`。参考配置见 [config.example.yaml](config.example.yaml)。`wsl_distribution` 与 `wsl_python_executable` 指定执行端；后端解释器选择顺序为启动参数、`IANVS_BACKEND_PYTHON` 环境变量、配置。CMD 中可以直接传递 Linux 解释器的绝对路径。`state_root` 与 `python_executable` 表示 Linux 后端状态目录和 Example 环境的基础解释器。地址为方舟 OpenAI 兼容 Chat API；价格未提供时费用显示“未知”。

前后端通过 stdio JSONL 通信，后台 stdout 专用于协议。后端在正式事件持久化之后发送状态快照和事件，前端刷新视图时读取缓存。长任务期间继续接收审批、取消和压缩请求；连接中断会显示错误并停止接受操作，重新启动后从后端正式记录恢复。前端不会迁移 venv、推断验证结果或把 Linux 路径作为 Windows 文件处理。

协议 v2 在首次连接和切换会话时同步完整快照，后续只发送变化字段及对话增量。超过 16 MiB 的完整消息分片传输，全部片段通过校验后再更新前端；中断的片段不进入会话缓存。更新前后端代码后应重新启动两端，协议版本不一致会明确报错。

工具包括源码读取、搜索、证据读取、版本绑定补丁、受限目录调查、环境准备、原任务运行、配置检查、验证与文件变化检查。`bash` 调查工具支持 `pwd`、`ls`、`stat`，搜索使用 `search_text`。工具不能访问实例外文件、凭据或管理记录，也不能修改运行控制、验证程序或原评测指标。

文件读取和搜索支持 UTF-8 以及带 BOM 的 UTF-16/UTF-32。搜索结果注明检查范围、编码、跳过文件和后续分页位置；二进制及无法明确识别编码的文件单独记录，权限和读取错误直接报告。`search_text.limit` 支持 1～1000。`read_evidence` 可以读取当前实例正式登记的验证报告引用，仍检查实例归属和允许路径。

环境准备复用 inventory 中声明的准备步骤，逐步展示名称、状态、实际命令和日志位置。完成后说明基础依赖、Example 声明的准备范围，以及尚未检查的模型和数据。发现未声明的 `requirements.txt` 或 `requirement.txt` 时显示提示；失败时显示具体步骤、最后一段输出和完整日志引用。旧 Run 早于环境准备时，明确提示需要重新运行取得新结果。Shell 脚本在 Windows 副本中使用 CRLF 时，以保留来源记录的 LF 执行副本运行。依赖变化产生新的环境代次。

创建候选 venv 前，使用真实 pip 解析器检查准备计划将直接安装的依赖文件、递归引用、约束和本地制品；该检查关闭网络，读取只读输入快照。缺少文件或版本语法错误时保存具体原因、文件版本与日志引用，原有环境和验证记录保持原状态。修改声明后可以再次准备。脚本内部动态生成的安装输入由脚本执行和后续环境检查确认。

需要服务凭据的 Example 使用 `runtime_credentials` 配置引用。例如 `DASHSCOPE_API_KEY: DASHSCOPE_API_KEY` 表示允许使用后端同名环境变量。Agent 通过 `configure_runtime_environment(credential_names=["DASHSCOPE_API_KEY"])` 请求当前实例绑定审批，仅名称进入正式记录。凭据值由后端环境提供，界面和模型不会获得该值，Provider 凭据不能绑定给 Example。相关 stdout、stderr 和阶段事件脱敏，含凭据的输出产物隔离并使本次 Run 失败。

## 记录和判定

`state_root` 默认指向 `workspace/ianvs-agent`。会话保存任务、对话、工具执行、事件、用量、Diff、Run、验证与上下文检查点。关闭后重启恢复最近会话，未完成工具调用只使用当前交互的实际完成记录补全。运行记录中的 stdout、stderr 和阶段事件相互独立。

创建实例时另存创建文件的原始字节、内容摘要和原目标记录。原配置路径失效时，只能在同一 Example 内按 `testenv/` 或 `testalgorithms/` 的完整目录后缀选择唯一候选，候选内容必须匹配创建时版本。恢复记录保留原声明、实际来源、核对依据与版本；初始 `task_spec` 和任务验收要求保持不变。多个候选、原版本缺失或配置结构尚未支持时，界面和工具报告具体未确认项目，仍可继续调查、准备环境和运行。

旧实例可以从尚未改变的实例文件、原目录文件、初始 `task_spec`、补丁修改前文本或历史 Run 配置中恢复原字节，每次均核对创建摘要。无法匹配时继续保留来源不足。来源补充后，任务和新 Run 绑定同一目标版本；历史 Run 必须重新执行才能验证新版本。每次模型请求重新读取正式目标，压缩摘要不能覆盖它。`read_evidence` 可以读取来源报告，验证页显示来源版本与检查能力说明。

独立验证比较原算法与测试环境的完整声明、指标实现、已登记的配置路径，以及创建时已有的模型和数据文件；本地制品目录还核对完整文件集合。配置来源完整只证明创建声明与已有文件身份，创建时不存在的远端模型或数据仍需其实际运行证据，来源报告会说明这一限制。

依赖方案保存在 `instances/<instance_id>/environment-plans/`，包含审批依据、输入快照、依赖文件版本、真实 pip 解析结果、带版本与制品校验信息的安装约束和安装命令；环境检查位于 `environment-checks/`，辅助命令位于 `environment-commands/`。各环境的实际包版本、安装步骤、pip 安装报告和执行日志保存在 `environments/<environment_id>/`，按正式引用读取。执行前的预检失败保存 `phase=preflight,command_executed=false,exit_code=null` 及错误日志。恢复时核查 PID、创建时间及操作身份，未确认完成的操作保持中断状态，不作为成功证据。

任务、状态、对话及会话信息保存在同一个原子快照；分文件用于查看正式记录。恢复长工具结果仍使用预览和原始全文引用。Journal 的末尾中断记录会连同原文件保存恢复证据，再恢复有效完整前缀；完整记录损坏会立即报错。

Linux 后端追加未被外部改写的日志时复用有界文件游标，避免反复解析全部旧事件。每次写入仍持有文件锁并执行 `fsync`；进程重启、文件替换、截断或元信息变化后重新校验全文。Windows 的文件时间语义不同，追加时保持全文校验。

验证区分执行完成、结果完整、检查通过、目标达成与修复完成。必需检查 `SKIP`、旧 Rank 产物、缺失 TestCase/指标、改变评分实现、运行后修改文件，都不能证明修复成功。修复复验 Run 必须在当前任务和最新补丁之后启动。

单次输入空间默认按所选模型的 `context_window` 减去本次输出预留计算；`context_input_budget` 默认为 `null`，不再附加 32000 Token 上限。如需主动缩小输入空间，可显式配置正整数。自动压缩阈值默认为模型窗口扣除输出预留后的 78%；显式输入预算更小时，阈值不超过该预算。估算包含 System、工具 Schema、正式状态、README、历史与证据。Observe 只发送当前工具交互，诊断和压缩输出默认限额为 2048 Token。手动压缩与服务明确报告上下文超限后的单次恢复使用同一 Context Engine。每个检查点保存原摘要、来源事件范围、完整历史前缀身份、模型与实际消耗。摘要用于继续推理；原始目标、事实证据和验收状态持续从正式记录读取。

每次准备请求时，STATE 发送视图包含最早五项与最近十五项事实、最近十项假设，其余程序状态完整保留。超过二十四项的文件身份清单在 TASK 发送视图中展示最早八项及当前已读文件对应身份。完整 task/state 保存在会话 `tool-output/TASK-<task_id>.json` 与 `STATE-<task_id>.json`，模型可以按引用读取。未展示的验收身份继续由独立验证器逐项检查。这项发送范围选择不调用摘要模型，也不计为压缩；原始记录与检查点保持全文。

上下文空间与任务累计用量分别管理。默认主模型配置窗口为 256000、输出预留 8192，单次输入空间为 247808 Token；这取决于模型配置，服务实际限制仍需遵守。为兼容已有配置，`max_input_tokens` 保留原名称，默认 500000，作为自动压缩周期的输入参考量。累计输入可以超过此数值，重启会话保留全部已用量。

每次 Plan 请求前，当“距离上次成功压缩的输入增量 + 下一次请求预计输入”达到周期参考量的 78% 时，Context Engine 检查自动压缩。这个比例复用 `context_threshold`，与模型窗口的压缩检查分别计算。存在可压缩的完整旧交互、预计能减少后续输入且摘要符合模型单次窗口时，发送摘要请求，再准备后续 Plan。每个检查点保存完成时的累计输入量，作为下一周期基准；旧检查点通过其真实模型请求与 Usage 恢复基准。累计用量持续增加。没有新增可压缩历史或预计不能节省时，继续当前任务。

周期摘要按请求历史身份与当前检查点去重，开始前保存尝试记录，完成后保存新的检查点；相同历史恢复后不会重复请求摘要，新增完整历史重新评估。相同跳过原因只提示一次。周期摘要遇到可恢复的网络失败时，保存 `deferred` 及失败请求引用；已准备的 Plan 仍符合模型窗口便继续执行，新增完整历史后再检查压缩。摘要调用的输入与输出计入原任务用量。累计 Token、模型和工具次数、运行次数及活动时间均用于统计，不阻止执行或恢复。Plan 只返回文字且尚未完成时，Observe 结合正式状态作结构化判断，能够继续时安排下一动作，确认存在无法解决的现实条件时说明证据并返回 `blocked`。修复成功持续读取当前任务与本轮 Run 的独立验证结果；固定次数和耗时不参与完成判定。

手动 `/compact` 在没有可压缩的旧交互时只提示“当前无需压缩”，保持原任务结果与停止原因；已排队的请求也会正常结束。服务超限恢复缺少旧交互时，继续缩减工具结果的发送预览并重新估算，完整证据和配对调用 ID 保留；后续请求仍需符合模型窗口。

Provider 分别记录连接、首个响应、接收过程及总时限。连接故障、读取超时、响应断流和请求总超时进入有限重试；每次尝试单独保存原始片段、错误和用量。只有完整响应通过检查后才交付工具调用，重试不会执行上次未完成的工具参数。认证、响应格式和存储错误保留明确失败，用户取消立即结束当前请求。`Provider.complete()` 支持可选正整数 `max_attempts`，将本次请求的尝试次数限制在该值和模型配置上限的较小值；未提供时沿用模型配置。累计预算摘要请求使用 `max_attempts=1`。主模型辅助请求的关闭思考选项已经真实接口验证，GLM 使用服务默认配置。

用量区分实际输入/输出、缓存读取、当前上下文估算、成功压缩、模型重试与工具调用。缓存读取包含在输入 Token 中。Provider 缺少字段时保留未知值，已知部分和估算分别显示。

`timeout_seconds` 控制模型流中的单次空闲读取，默认 120 秒；`request_timeout_seconds` 控制完整请求，默认 600 秒。持续收到响应时不会因累计超过空闲时限而中止；总时限结束保留部分响应和失败证据，任务可继续恢复。

文件、准备声明和已安装依赖变化后，维护检查会标记需要重新验证。日常循环检查实际运行的独立实例；需要排查源项目更新时，使用 `get_change_impact(include_source_updates=true)` 主动扫描来源目录，并保存检查时间。旧验证不会继续证明变化后的实例已经通过。Linux 原生目录的日常检查复用版本未变化的内存摘要，独立运行与验证继续完整读取文件。

## 检查

完整后端检查在已安装全部依赖的 WSL 控制环境运行：

```bash
python -m unittest discover -s ianvs_agent/tests -p 'test_*.py' -v
python -m ianvs_agent.tests.check_provider_live --profile ark --profile glm --compact
```

Windows 界面检查使用 `ianvs_agent.tests.test_tui_remote`。正式恢复测试需要 `IANVS_TUI_BACKEND_CONFIG` 指向已有真实修复会话配置，`IANVS_TUI_BACKEND_PYTHON` 指向 Linux 控制解释器；两种启动失败检查可在 Windows 界面环境直接运行。Runtime 取证测试可使用 `IANVS_VERIFIER_TEST_ROOT` 指定 Linux 文件系统中的目录。跨环境真实运行、模型与界面报告见 [ACCEPTANCE.md](ACCEPTANCE.md)。

真实模型验收需要账户授权，会产生实际调用消耗。真实修复入口 `check_agent_live` 接受已经准备并通过健康检查的独立实例，注入配置缺失故障，再由真实 Agent 应用补丁、压缩、运行、验证和恢复会话。原始失败记录持续保留。

完整层次与本地验收证据见 [ACCEPTANCE.md](ACCEPTANCE.md)。高级能力如多 Agent、分布式运行、自动模型路由、容器调度和向量检索不属于 v0.1。

已有 Python 代码修复实验覆盖方法调用错误与生成输出切片错误，由模型修改源码、复跑原任务并取得独立验证。范围为固定的单 QA 冒烟任务，完整本地记录保留失败工具调用；可公开的实验范围与汇总见 [验证记录](ACCEPTANCE.md)。
