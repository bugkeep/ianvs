# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import platform
import time
from contextlib import asynccontextmanager

from rich.syntax import Syntax
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.message import Message
from textual.widgets import Button, DataTable, Footer, Input, LoadingIndicator, ProgressBar, RichLog, Select, Static, TabbedContent, TabPane
from textual.worker import Worker, WorkerState

from ianvs_agent.app.records import assistant_text, conversation_messages, environment_text, short_text, tool_display_status, tool_parameters, tool_summary
from ianvs_agent.app.logs import ReflowRichLog, RunOutputBuffer
from ianvs_agent.app.screens import ApprovalScreen, ExitScreen, ToolDetailsScreen
from ianvs_agent.message_types import format_task_result


STATUS_LABELS = {
    "created": "已创建", "pending": "等待执行", "ready": "准备完成",
    "preparing": "准备环境", "running": "正在执行", "completed": "执行完成",
    "succeeded": "执行成功", "success": "执行成功", "passed": "检查通过",
    "failed": "执行失败", "error": "发生错误", "blocked": "任务阻塞",
    "cancelled": "已取消", "interrupted": "已中断", "timed_out": "执行超时",
    "needs_verification": "需要重新验证", "not_passed": "尚未通过",
    "not_run": "尚未运行", "unknown": "未知状态", "PASS": "通过",
    "FAIL": "失败", "SKIP": "未执行", "plan": "规划", "observe": "观察",
    "compaction": "压缩上下文", "diagnose": "诊断", "repair": "修复",
    "configuration_invalid": "配置待修复", "creating": "正在创建",
    "creation_failed": "创建失败", "valid": "检查通过", "invalid": "待修复",
    "verified": "独立验证通过", "verification_failed": "独立验证未通过",
    "WARNING": "提示", "ERROR": "错误",
    "awaiting_approval": "等待审批", "denied": "已拒绝",
    "unconfirmed": "结果待核查",
}

CHECK_LABELS = {
    "Example directory exists": "示例目录存在",
    "benchmarkingjob.yaml exists": "评测任务配置存在",
    "requirements file exists": "依赖声明文件存在",
    "dataset prepare script exists": "数据准备脚本存在",
    "Environment preparation contract": "环境准备声明",
    "Mock LLM runtime contract": "模拟模型运行声明",
    "YAML syntax": "YAML 语法",
    "Repository path references exist": "引用的仓库路径存在",
    "Repository path parent references exist": "引用路径的父目录存在",
    "Hardcoded local path check": "固定本机路径检查",
    "Local model path check": "本地模型路径检查",
    "CUDA-only device check": "CUDA 设备限定检查",
    "Metric empty-pair guard": "指标空输入检查",
    "Dependency file declared": "依赖文件已经声明",
    "Dependency file exists": "依赖文件存在",
    "Dependency declaration syntax": "依赖声明语法",
    "Dependency file is not empty": "依赖文件包含内容",
    "Python version marker compatibility": "Python 版本声明兼容性",
    "Runtime imports declared": "运行依赖已经声明",
    "pip install check": "pip 安装检查",
    "pip availability": "pip 可用性",
    "Environment preparation": "环境准备",
    "Environment preparation configuration": "环境准备配置",
    "Smoke benchmark config exists": "冒烟评测配置存在",
    "Runtime smoke test": "运行冒烟测试",
    "Runtime smoke test (mocked_llm)": "模拟模型冒烟测试",
    "Dataset preparation": "数据集准备",
    "JSONL dataset structure": "JSONL 数据结构",
    "Validation runner internal error": "验证程序内部错误",
    "Dynamic validation eligibility": "动态验证条件",
}

EVENT_LABELS = {
    "session.started": "会话已开始", "session.resumed": "会话已恢复",
    "llm.started": "模型正在处理", "llm.completed": "模型响应完成",
    "llm.failed": "模型请求失败", "llm.retry": "模型正在重试",
    "llm.usage": "模型用量已记录", "tool.started": "工具开始执行",
    "tool.completed": "工具执行完成", "tool.failed": "工具执行失败",
    "context.compacted": "上下文压缩完成", "context.preparing": "正在准备上下文", "context.prepared": "上下文已准备",
    "context.limit_recovery": "上下文超限，正在压缩并重试",
    "context.compaction_queued": "上下文压缩请求已排队",
    "context.compaction_skipped": "当前无需压缩",
    "context.compaction_deferred": "周期压缩暂缓，继续任务",
    "run.started": "评测开始执行", "run.stage_changed": "评测阶段更新",
    "run.completed": "评测执行完成", "run.failed": "评测执行失败",
    "run.output": "评测输出", "run.cancelled": "评测已取消",
    "run.pipeline_completed": "评测流程完成", "run.pipeline_failed": "评测流程失败",
    "patch.applied": "代码补丁已应用", "verification.completed": "独立验证完成",
    "agent.completed": "智能体任务完成", "agent.blocked": "智能体任务阻塞",
    "agent.cancelled": "智能体任务已取消", "approval.requested": "等待操作审批",
    "agent.message": "智能体回复", "assistant.message": "智能体回复", "agent.failed": "智能体任务失败",
    "model.selected": "模型配置已切换",
    "verification.started": "正在进行独立验证", "testcases.built": "测试对象已创建",
    "testcase.started": "测试对象开始执行", "testcase.completed": "测试对象执行完成",
    "testcase.failed": "测试对象执行失败",
    "approval.completed": "操作审批完成", "approval.decided": "操作审批完成", "instance.created": "实例已创建",
    "instance.selected": "实例已选择", "instance.changed": "实例文件已变化",
    "environment.started": "正在准备环境", "environment.completed": "环境准备完成",
    "environment.prepared": "环境准备完成", "environment.failed": "环境准备失败",
    "environment.step_started": "环境步骤开始", "environment.step_completed": "环境步骤完成",
    "environment.step_failed": "环境步骤失败",
    "environment.inspected": "环境检查完成", "environment.changed": "环境已变化",
    "environment.command_started": "环境命令开始", "environment.command_progress": "环境命令执行中",
    "environment.command_completed": "环境命令完成", "environment.command_failed": "环境命令失败",
    "environment.interrupted": "环境操作已中断", "environment.credentials_configured": "运行凭据引用已绑定",
}

STAGE_LABELS = {
    "configuration": "读取配置", "configure": "读取配置", "config": "读取配置",
    "initialization": "初始化", "initialize": "初始化", "dataset": "准备数据",
    "algorithm": "加载算法", "testcase": "执行测试对象", "test_case": "执行测试对象",
    "training": "训练", "train": "训练", "evaluation": "评估", "evaluate": "评估",
    "inference": "推理", "predict": "推理", "ranking": "生成排行", "rank": "生成排行",
    "prepare": "准备", "verify": "验证", "cleanup": "清理",
    "runtime_import": "加载运行依赖", "configuration_load": "读取评测配置",
    "simulation_prepare": "准备仿真环境", "dataset_prepare": "准备数据集",
    "testcase_build": "创建测试对象", "testcase_execution": "执行测试对象",
    "rank_output": "生成排行结果", "algorithm_initialization": "初始化算法",
    "algorithm_execution": "执行算法", "metric_computation": "计算指标",
}


def format_number(value):
    if value is None:
        return "未知"
    if value >= 1000000:
        return f"{value / 1000000:.2f}M"
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    return str(value)


def status_label(value):
    return STATUS_LABELS.get(value, str(value or "未知状态"))


class HarnessEvent(Message):
    def __init__(self, event):
        super().__init__()
        self.event = event


class IanvsAgentApp(App):
    TITLE = "Ianvs 智能体"
    SUB_TITLE = "MVP v0.1"
    CSS_PATH = "tui.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("f2", "run_example", "运行"),
        Binding("f3", "show_diff", "修改对比"),
        Binding("f4", "show_usage", "用量"),
        Binding("f5", "show_verification", "验证报告"),
        Binding("f6", "show_tools", "工具记录"),
        Binding("f7", "resume_task", "继续原任务"),
        Binding("ctrl+k", "compact", "压缩上下文", priority=True),
        Binding("ctrl+n", "create_instance", "创建实例"),
        Binding("ctrl+e", "prepare_environment", "准备环境", priority=True),
        Binding("ctrl+x", "cancel_task", "取消任务", priority=True),
        Binding("ctrl+q", "request_exit", "退出", priority=True),
        Binding("ctrl+r", "connect_backend", "连接后端", priority=True),
    ]

    def __init__(self, settings, harness=None):
        super().__init__()
        self.settings = settings
        if harness is None:
            from ianvs_agent.agent.harness import AgentHarness
            self.harness = AgentHarness(settings)
        else:
            self.harness = harness
        self._remote = hasattr(self.harness, "connected")
        self._connection_worker = None
        self._close_task = None
        self._harness_closed = False
        self.examples = self.harness.catalog.list_examples()
        self.selected_example = None
        self._refreshing = False
        self._active_action = None
        self._approvals = []
        self._approval_open = False
        self._cancelling = False
        self._last_context_fraction = None
        self._last_status = "就绪，请选择示例或恢复实例"
        self._displayed_session_id = None
        self._seen_events = set()
        self._status_timer = None
        self._status_dirty = False
        self._last_status_refresh = 0
        self._activity_timer = None
        self._activity_started_at = None
        self._activity_phase = ""
        self._retry_until = None
        self._tool_records = {}
        self._selected_tool_id = None
        self._tool_table_width = None
        self._stream_group = None
        self._stream_text = ""
        self._stream_dirty = False
        self._run_output = RunOutputBuffer()
        self._run_output_dirty = False
        self._unsubscribe = self.harness.events.subscribe(self.receive_event)

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("◈  Ianvs 智能体", id="brand")
            yield Static("MVP v0.1", id="version")
            yield Static(id="backend-status", markup=False)
            yield Static("模型", id="model-label")
            yield Select([(f"{profile.name} · {profile.model}", name) for name, profile in self.settings.profiles.items()], value=self.settings.selected_profile, allow_blank=False, id="model-select")
        with Horizontal(id="workspace"):
            with VerticalScroll(id="sidebar", classes="panel"):
                if self._remote:
                    yield Button("连接后端", id="connect-backend")
                yield Static("示例与实例", classes="section-title")
                yield Input(placeholder="筛选示例名称或路径", id="example-filter")
                yield DataTable(id="example-table", cursor_type="row", zebra_stripes=True)
                yield Button("创建独立实例", id="create-instance", variant="primary")
                yield Static("恢复已有实例", classes="field-label")
                yield Select([], prompt="选择实例", id="instance-select")
                yield Static("尚未选择实例", id="instance-info", markup=False)
                yield Button("准备运行环境", id="prepare-environment")
                yield Button("运行当前示例", id="run-example", variant="success")
                yield Button("继续原任务 · F7", id="resume-task", variant="primary", disabled=True)
            with Vertical(id="main-panel", classes="panel"):
                yield Static("对话与执行记录", classes="section-title")
                with TabbedContent(initial="conversation", id="main-tabs"):
                    with TabPane("对话", id="conversation"):
                        yield ReflowRichLog(id="conversation-log", wrap=True, markup=False, highlight=False, min_width=1)
                        yield Static(id="stream-response", markup=False)
                    with TabPane("工具", id="tools"):
                        yield DataTable(id="tool-table", cursor_type="row", zebra_stripes=True)
                        yield Button("查看选中工具详情", id="tool-details-button", disabled=True)
                    with TabPane("修改", id="diff"):
                        yield RichLog(id="diff-log", wrap=False, markup=False)
                    with TabPane("用量", id="usage"):
                        with VerticalScroll():
                            yield Static(id="usage-detail", markup=False)
                    with TabPane("验证", id="verification"):
                        with VerticalScroll():
                            yield Static(id="verification-detail", markup=False)
                    with TabPane("日志", id="logs"):
                        yield ReflowRichLog(id="run-log", wrap=True, markup=False, highlight=False, max_lines=2000, min_width=1)
                        yield Static(id="run-progress", markup=False)
                with Horizontal(id="input-row"):
                    yield Input(placeholder="输入任务，或 /compact 压缩上下文", id="message-input")
                    yield Button("发送", id="send-message", variant="primary")
        with Vertical(id="summary-panel", classes="panel"):
            with Horizontal(id="run-heading"):
                yield Static("运行与验证", classes="section-title")
                yield Static("尚未运行", id="run-status", markup=False)
            with Horizontal(id="activity-row"):
                yield LoadingIndicator(id="activity-indicator")
                yield Static(self._last_status, id="task-status", markup=False)
                yield Static(id="activity-time", markup=False)
            yield Static("上下文与用量", id="usage-heading", classes="field-label")
            with Horizontal(id="usage-row"):
                yield Static("上下文占用\n未知", id="context-stat", classes="metric", markup=False)
                yield Static("输入 Token\n未知", id="input-stat", classes="metric", markup=False)
                yield Static("输出 Token\n未知", id="output-stat", classes="metric", markup=False)
                yield Static("费用\n未知", id="cost-stat", classes="metric", markup=False)
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="context-bar")
        yield Footer()

    def on_mount(self):
        self.query_one("#example-table", DataTable).add_column("示例名称", key="name")
        self.resize_tool_table()
        self.refresh_examples()
        self.refresh_instances()
        self.restore_conversation()
        self.refresh_status()
        self.query_one("#message-input", Input).focus()
        self._status_timer = self.set_interval(0.1, self.flush_status)
        self._activity_timer = self.set_interval(0.25, self.refresh_activity)
        self.set_class(self.size.width < 100, "compact-layout")
        self.set_class(self.size.height < 35, "short-layout")
        if self._remote:
            self.action_connect_backend()

    def on_resize(self, event: Resize):
        self.set_class(event.size.width < 100, "compact-layout")
        self.set_class(event.size.height < 35, "short-layout")
        self.call_after_refresh(self.resize_tool_table)

    async def on_unmount(self):
        if self._status_timer is not None:
            self._status_timer.stop()
        if self._activity_timer is not None:
            self._activity_timer.stop()
        self._unsubscribe()
        await self.close_harness()

    @asynccontextmanager
    async def run_test(self, *args, **kwargs):
        try:
            async with super().run_test(*args, **kwargs) as pilot:
                yield pilot
        finally:
            await self.close_harness()

    def backend_ready(self):
        return not self._remote or self.harness.connected

    @on(Button.Pressed, "#connect-backend")
    def connect_backend_button(self):
        self.action_connect_backend()

    def action_connect_backend(self):
        if not self._remote or self.harness.connected:
            return
        if self._connection_worker and self._connection_worker.is_running:
            return
        self._last_status = "正在连接执行后端"
        self.begin_activity(self._last_status)
        self._connection_worker = self.run_worker(self.connect_backend(), name="连接执行后端", group="connection", exit_on_error=False)
        self.refresh_status()

    async def connect_backend(self):
        await self.harness.start()
        self.examples = self.harness.catalog.list_examples()
        self.query_one("#model-select", Select).set_options([(f"{profile.name} · {profile.model}", name) for name, profile in self.settings.profiles.items()])
        self.query_one("#model-select", Select).value = self.settings.selected_profile
        self.refresh_examples(self.query_one("#example-filter", Input).value)
        self.refresh_instances()
        self.restore_conversation()
        self._last_status = "执行后端已连接，正式会话与实例信息已恢复"
        self.end_activity()
        self.refresh_status()

    def refresh_backend_identity(self):
        frontend = platform.system()
        label = "当前进程执行"
        details = f"界面平台：{frontend}"
        color = "#6ee7b7"
        if self._remote:
            state = "已连接" if self.harness.connected else "连接失败" if self.harness.connection_error else "等待连接"
            if self._connection_worker and self._connection_worker.is_running:
                state = "正在连接"
            label = f"{self.harness.backend_label} 后端 · {state}"
            info = self.harness.backend_info or {}
            details += "\n执行平台：" + str(info.get("platform") or info.get("os") or "等待后端报告")
            if info.get("wsl_distribution"):
                details += f"\nWSL 发行版：{info['wsl_distribution']}"
            if info.get("python_executable"):
                details += f"\n控制解释器：{info['python_executable']}"
            if info.get("pid") is not None:
                details += f"\n后端进程：{info['pid']}"
            if info.get("project_root"):
                details += f"\n后端源码：{info['project_root']}"
            if self.harness.connection_error:
                details += f"\n连接错误：{self.harness.connection_error}"
                self._last_status = f"执行后端连接失败：{self.harness.connection_error}"
            color = "#6ee7b7" if self.harness.connected else "#fbbf24"
            button = self.query_one("#connect-backend", Button)
            button.disabled = self.harness.connected or bool(self._connection_worker and self._connection_worker.is_running)
            button.label = "后端已连接" if self.harness.connected else "重新连接后端" if self.harness.connection_error else "连接后端"
        badge = self.query_one("#backend-status", Static)
        badge.update(Text(f"界面：{frontend}\n{label}", style=color))
        badge.tooltip = details

    def receive_event(self, event):
        self.post_message(HarnessEvent(event))

    def refresh_examples(self, filter_text=""):
        table = self.query_one("#example-table", DataTable)
        self.selected_example = None
        table.clear()
        needle = filter_text.casefold()
        for index, example in enumerate(self.examples):
            if needle and needle not in f"{example['name']} {example['benchmark_path']}".casefold():
                continue
            label = Text(example["name"])
            if example.get("config_status") == "invalid":
                label.append(" · 配置待修复", style="#fbbf24")
            table.add_row(label, key=str(index))
        table.border_subtitle = f"{table.row_count} 个示例"
        self.refresh_status()

    def refresh_instances(self):
        self._refreshing = True
        selector = self.query_one("#instance-select", Select)
        instances = self.harness.instances.list_instances()
        selector.set_options([(f"{instance.get('name', instance['instance_id'])} · {instance['instance_id']}", instance["instance_id"]) for instance in instances])
        current = self.harness.current_instance
        if current:
            selector.value = current["instance_id"]
        self._refreshing = False

    def restore_conversation(self):
        log = self.query_one("#conversation-log", RichLog)
        log.clear()
        self.query_one("#run-log", RichLog).clear()
        self._run_output = RunOutputBuffer()
        self._run_output_dirty = True
        self.query_one("#tool-table", DataTable).clear()
        self._tool_records.clear()
        self._selected_tool_id = None
        self.query_one("#tool-details-button", Button).disabled = True
        self.clear_stream()
        self._seen_events.clear()
        self._last_context_fraction = None
        self._displayed_session_id = self.harness.session["session_id"] if self.harness.session else None
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl") if self.harness.session else []
        messages = conversation_messages(self.harness.history(), events)
        if not messages:
            hint = "当前实例与会话已就绪，可以输入任务或运行示例。" if self.harness.current_instance else "选择示例并创建实例，或恢复已有实例。浏览与选择不会调用模型。"
            log.write(Text(hint, style="#94a3b8"))
        for role, content in messages:
            self.write_conversation(role, content)
        run = self.last_run()
        for event in events:
            if event.get("event_id"):
                self._seen_events.add(event["event_id"])
            self.update_tool_record(event, restored=True)
            self.write_internal_event(event)
            content = environment_text(event, run)
            if content:
                self.write_conversation("assistant", content)
        state = self.harness.state
        status = state.get("status")
        result = state.get("result")
        if not result and status in {"blocked", "failed", "cancelled", "interrupted", "completed"} and not (status == "completed" and self.harness.task.get("kind") == "conversation"):
            result = format_task_result(self.harness.task, state, status, state.get("status_message") or state.get("error") or "", run, record_ref=self.harness.session_relative_path + "/state.json")
        if result and result.get("content") and not any(content == result["content"] for _, content in messages):
            self.write_conversation("assistant", result["content"])

    def last_run(self):
        current = self.harness.current_instance
        return self.harness.runner.get_run(current["instance_id"], current["last_run_id"]) if current and current.get("last_run_id") else None

    def write_conversation(self, role, content):
        label = "用户" if role == "user" else "智能体"
        log = self.query_one("#conversation-log", RichLog)
        log.write(Text(label, style="bold #bfdbfe" if role == "user" else "bold #6ee7b7"))
        log.write(Text(content))
        log.write("")

    def clear_stream(self):
        self._stream_group = None
        self._stream_text = ""
        self._stream_dirty = False
        widget = self.query_one("#stream-response", Static)
        widget.update("")
        widget.display = False

    def begin_activity(self, phase):
        if self._activity_started_at is None:
            self._activity_started_at = time.monotonic()
        self._activity_phase = phase
        self._retry_until = None
        self.refresh_activity()

    def end_activity(self):
        self._activity_started_at = None
        self._activity_phase = ""
        self._retry_until = None
        self.refresh_activity()

    def refresh_activity(self):
        if not self.is_running:
            return
        active = self._activity_started_at is not None
        self.query_one("#activity-indicator", LoadingIndicator).styles.visibility = "visible" if active else "hidden"
        text = self._activity_phase if active else self._last_status
        if active and self._retry_until is not None:
            remaining = max(0, self._retry_until - time.monotonic())
            text += f" · {remaining:.0f} 秒后重试" if remaining > 0 else " · 准备重试"
        self.query_one("#task-status", Static).update(Text(text))
        elapsed = max(0, time.monotonic() - self._activity_started_at) if active else 0
        self.query_one("#activity-time", Static).update(f"累计 {elapsed:.0f} 秒" if active else "")
        for record in self._tool_records.values():
            started = record.get("started_at")
            if record.get("status") in {"running", "awaiting_approval"} and started is not None:
                second = int(max(0, time.monotonic() - started))
                if record.get("displayed_second") != second:
                    record["displayed_second"] = second
                    self.write_tool_row(record)
        if self._stream_dirty:
            widget = self.query_one("#stream-response", Static)
            widget.update(Text("智能体正在回复\n" + self._stream_text[-4000:]))
            widget.display = True
            self._stream_dirty = False
        if self._run_output_dirty:
            preview = self._run_output.preview()
            widget = self.query_one("#run-progress", Static)
            widget.update(preview)
            widget.display = bool(preview)
            self._run_output_dirty = False

    def update_activity(self, event):
        event_type = event["event_type"]
        payload = event.get("payload") or {}
        if event_type == "llm.started":
            self.clear_stream()
            self._stream_group = payload.get("request_group_id")
            purpose = payload.get("purpose")
            phase = "压缩上下文，等待模型响应" if purpose == "compaction" else "分析工具结果，等待模型响应" if purpose == "observe" else "等待模型响应"
            self.begin_activity(f"{phase} · {payload.get('model', self.settings.profile.model)}")
        elif event_type == "llm.delta":
            if payload.get("purpose") not in {"observe", "compaction"} and isinstance(payload.get("text"), str):
                if self._stream_group != payload.get("request_group_id"):
                    self.clear_stream()
                    self._stream_group = payload.get("request_group_id")
                self._stream_text += payload["text"]
                self._stream_dirty = True
                if self._activity_started_at is None:
                    self._activity_started_at = time.monotonic()
                self._activity_phase = "正在接收模型回复"
        elif event_type == "llm.progress":
            if self._activity_started_at is None:
                self._activity_started_at = time.monotonic() - float(payload.get("elapsed_seconds") or 0)
            label = "正在接收模型回复" if payload.get("content_characters") else "模型正在生成响应"
            elapsed = float(payload.get("elapsed_seconds") or 0)
            last = payload.get("last_chunk_seconds")
            activity = "等待首个数据片段" if last is None else f"最近数据 {max(0, elapsed - last):.1f} 秒前"
            self._activity_phase = f"{label} · 收到 {payload.get('chunk_count', 0)} 个片段 · {activity}"
        elif event_type == "llm.retry":
            self.clear_stream()
            self.begin_activity(f"等待模型重试 · 第 {payload.get('next_attempt', payload.get('attempt', 1))} 次请求")
            self._retry_until = time.monotonic() + float(payload.get("delay_seconds") or 0)
        elif event_type == "llm.failed":
            self.clear_stream()
            if payload.get("will_retry"):
                self.begin_activity("模型请求失败，等待重试")
            else:
                self._last_status = "模型请求已取消" if payload.get("status") == "cancelled" else short_text(payload.get("error") or "模型请求失败，错误已保存", 140)
                self.end_activity()
        elif event_type == "llm.completed":
            self.begin_activity("模型响应完成，正在处理结果")
        elif event_type == "tool.started":
            self.begin_activity("正在执行工具 · " + str(payload.get("name", "")))
        elif event_type == "approval.requested":
            self.begin_activity("等待操作审批 · " + str(payload.get("tool_name", "")))
        elif event_type in {"approval.decided", "approval.completed"}:
            self.begin_activity("审批已批准，正在执行工具" if payload.get("approved") else "审批已拒绝，正在记录结果")
        elif event_type in {"tool.completed", "tool.failed"}:
            if self.harness.busy or self._active_action is not None:
                self.begin_activity("工具已返回，正在处理结果")
            else:
                self.end_activity()
        elif event_type == "context.preparing":
            role = payload.get("role") or payload.get("purpose")
            self.begin_activity("正在准备压缩上下文" if role == "compaction" else "正在准备工具分析上下文" if role == "observe" else "正在准备模型上下文")
        elif event_type in {"context.prepared", "context.limit_recovery"}:
            self.begin_activity("上下文已准备，等待模型请求" if event_type == "context.prepared" else "上下文超限，正在压缩")
        elif event_type == "context.compacted":
            self.begin_activity("上下文已压缩，正在继续任务")
        elif event_type == "context.compaction_deferred":
            self.begin_activity("周期压缩请求失败，已保存证据，继续当前任务")
        elif event_type == "context.compaction_skipped":
            self.notify(payload.get("message") or "近期完整交互不足，无需压缩。", title="上下文")
            if self.harness.state.get("status") == "running":
                self.begin_activity("上下文无需压缩，继续当前任务")
            else:
                self.end_activity()
        elif event_type == "run.started":
            self.begin_activity("正在运行评测 · " + str(event.get("run_id") or payload.get("run_id") or ""))
        elif event_type == "run.stage_changed":
            self.begin_activity("正在评测 · " + STAGE_LABELS.get(payload.get("stage"), str(payload.get("stage") or "执行中")))
        elif event_type == "verification.started":
            self.begin_activity("正在独立验证本轮结果")
        elif event_type in {"environment.started", "environment.preparing"}:
            self.begin_activity("正在准备运行环境")
        elif event_type == "environment.step_started":
            self.begin_activity("正在准备环境 · " + str(payload.get("label") or payload.get("name", "")))
        elif event_type in {"environment.completed", "environment.failed"}:
            self.end_activity()
        elif event_type in {"environment.command_started", "environment.command_progress"}:
            self.begin_activity("正在执行环境命令 · " + str(payload.get("operation_id", "")))
        elif event_type in {"environment.inspected", "environment.command_completed", "environment.command_failed", "environment.credentials_configured", "environment.interrupted"}:
            if self.harness.busy or self._active_action is not None:
                self.begin_activity("环境证据已保存，正在继续当前任务")
            else:
                self.end_activity()
        elif event_type in {"agent.completed", "agent.blocked", "agent.cancelled", "agent.failed"}:
            self.clear_stream()
            self.end_activity()

    def update_tool_record(self, event, restored=False):
        event_type = event["event_type"]
        if event_type not in {"tool.started", "tool.completed", "tool.failed", "approval.requested", "approval.decided", "approval.completed"}:
            return
        payload = event.get("payload") or {}
        call_id = payload.get("call_id")
        if not call_id:
            return
        record = self._tool_records.setdefault(call_id, {"call_id": call_id})
        record.update(payload)
        record["name"] = payload.get("name") or payload.get("tool_name") or record.get("name", "未知工具")
        if event_type == "tool.started":
            record["status"] = "running"
            record["started_at"] = None if restored else time.monotonic()
        elif event_type == "approval.requested":
            record["status"] = "awaiting_approval"
        elif event_type in {"approval.decided", "approval.completed"}:
            record["status"] = "running" if payload.get("approved") else "denied"
        else:
            record["status"] = payload.get("status") or ("success" if event_type == "tool.completed" else "error")
            record["record_ref"] = self.harness.session_relative_path + "/tools.jsonl · " + str(payload.get("operation_id") or call_id)
        self.write_tool_row(record)

    def resize_tool_table(self):
        if not self.is_running:
            return
        width = max(34, self.query_one("#main-panel").content_size.width - 2)
        if width == self._tool_table_width:
            return
        self._tool_table_width = width
        table = self.query_one("#tool-table", DataTable)
        table.clear(columns=True)
        name_width = max(12, int((width - 6) * 0.4))
        status_width = 10
        summary_width = max(6, width - name_width - status_width - 6)
        table.add_column("工具 / 参数", key="name", width=name_width)
        table.add_column("状态 / 耗时", key="status", width=status_width)
        table.add_column("结果摘要", key="summary", width=summary_width)
        for record in self._tool_records.values():
            self.write_tool_row(record)

    def write_tool_row(self, record):
        table = self.query_one("#tool-table", DataTable)
        call_id = record["call_id"]
        duration = record.get("duration_seconds")
        timing = f"{duration:.2f} 秒" if isinstance(duration, (int, float)) else f"已用 {record['displayed_second']} 秒" if record.get("displayed_second") is not None else "执行中" if record.get("status") in {"running", "awaiting_approval"} else "耗时未知"
        display_status = tool_display_status(record)
        cells = [Text(record["name"] + "\n" + tool_parameters(record.get("arguments") or {}), overflow="ellipsis", no_wrap=True), Text(("error" if display_status == "error" else status_label(display_status)) + "\n" + timing, style="#fca5a5" if display_status == "error" else ""), Text(tool_summary(record), overflow="ellipsis", no_wrap=True)]
        if call_id in table.rows:
            for column, cell in zip(table.columns, cells):
                table.update_cell(call_id, column, cell)
        else:
            table.add_row(*cells, key=call_id, height=2)
        table.border_subtitle = f"{len(self._tool_records)} 次调用 · Enter 查看详情"

    def write_internal_event(self, event):
        if event["event_type"] in {"llm.delta", "agent.message", "assistant.message", "llm.usage"}:
            return
        payload = event.get("payload") or {}
        run_id = event.get("run_id") or payload.get("run_id") or "当前运行"
        log = self.query_one("#run-log", RichLog)
        if event["event_type"] == "run.output":
            channel = payload.get("channel") or "stdout"
            output = payload.get("text") or payload.get("output") or ""
            lines = self._run_output.feed(run_id, channel, output)
            if lines:
                content = Text(f"{run_id} · {channel}\n", style="#93c5fd")
                for index, line in enumerate(lines):
                    if index:
                        content.append("\n")
                    content.append(line)
                log.write(content)
            self._run_output_dirty = True
            return
        if event["event_type"] in {"run.completed", "run.failed", "run.cancelled", "run.pipeline_completed", "run.pipeline_failed"}:
            for key, line in self._run_output.finish(run_id):
                log.write(Text(f"{key[0]} · {key[1]}\n", style="#93c5fd") + line)
            self._run_output_dirty = True
        text, color = self.format_event(event)
        log.write(Text.from_ansi(text, style=color))

    @on(DataTable.RowHighlighted, "#tool-table")
    def highlight_tool(self, event: DataTable.RowHighlighted):
        self._selected_tool_id = event.row_key.value
        self.query_one("#tool-details-button", Button).disabled = False

    @on(DataTable.RowSelected, "#tool-table")
    def select_tool(self, event: DataTable.RowSelected):
        self._selected_tool_id = event.row_key.value
        self.show_tool_details()

    @on(Button.Pressed, "#tool-details-button")
    def show_tool_details(self):
        if self._selected_tool_id in self._tool_records:
            self.push_screen(ToolDetailsScreen(dict(self._tool_records[self._selected_tool_id])))

    def action_show_tools(self):
        self.query_one("#main-tabs", TabbedContent).active = "tools"

    def flush_status(self):
        if self._status_dirty or time.monotonic() - self._last_status_refresh >= 1:
            self.refresh_status()

    def refresh_status(self):
        if not self.is_running:
            return
        self._status_dirty = False
        self._last_status_refresh = time.monotonic()
        self.refresh_backend_identity()
        current = self.harness.current_instance
        busy = not self.backend_ready() or self.harness.busy or self._active_action is not None
        self.query_one("#create-instance", Button).disabled = busy or self.selected_example is None
        self.query_one("#prepare-environment", Button).disabled = busy or current is None
        self.query_one("#run-example", Button).disabled = busy or current is None
        self.query_one("#resume-task", Button).disabled = busy or current is None or not self.harness.task or self.harness.state.get("status") not in {"failed", "blocked", "cancelled", "interrupted"}
        self.query_one("#send-message", Button).disabled = busy or current is None
        self.query_one("#instance-select", Select).disabled = busy
        model_select = self.query_one("#model-select", Select)
        model_select.disabled = busy
        if not busy and model_select.value != self.settings.selected_profile:
            model_select.value = self.settings.selected_profile
        info = "尚未选择实例"
        if current:
            environment = current.get("environment") or {}
            state = environment.get("status", "尚未准备")
            info = f"实例：{current['instance_id']}\n状态：{status_label(current.get('status'))}\n环境：{status_label(state)}\n源码：{current['code_root']}"
            verification = current.get("verification") or {}
            invalidated = current.get("status") == "needs_verification" or verification.get("stale") or verification.get("status") == "needs_verification"
            if invalidated:
                info += "\n实例内容或环境变化，需要重新验证"
            elif verification.get("passed") is False:
                info += "\n最近独立验证：未通过"
            elif current.get("needs_verification"):
                info += "\n当前任务需要独立验证"
            elif isinstance(verification, dict) and verification.get("passed"):
                info += "\n最近独立验证：通过"
            if not busy and current.get("last_run_id"):
                run = self.harness.runner.get_run(current["instance_id"], current["last_run_id"])
                state = "需要重新验证" if invalidated else "独立验证通过" if verification.get("passed") else "独立验证未通过" if verification.get("passed") is False else status_label(run["status"])
                self.query_one("#run-status", Static).update(Text(f"{state} · {run['run_id']}"))
        self.query_one("#instance-info", Static).update(Text(info))
        if self._remote and self.harness.connection_error and self._activity_started_at is not None:
            self.clear_stream()
            self.end_activity()
        self.refresh_activity()
        summary = self.harness.usage_summary()
        fraction = summary.get("context_fraction", self._last_context_fraction)
        context = "未知" if fraction is None else f"{fraction * 100:.1f}%（估算）"
        self.query_one("#context-stat", Static).update(f"上下文占用\n{context}")
        self.query_one("#context-bar", ProgressBar).update(progress=0 if fraction is None else min(100, fraction * 100))
        self.query_one("#input-stat", Static).update(f"输入 Token\n{format_number(summary.get('input_tokens'))}")
        self.query_one("#output-stat", Static).update(f"输出 Token\n{format_number(summary.get('output_tokens'))}")
        cost = summary.get("cost")
        self.query_one("#cost-stat", Static).update(f"费用\n{'未知' if cost is None else f'{cost:.6f}'}")
        if self.query_one("#main-tabs", TabbedContent).active == "usage":
            self.refresh_usage(summary)

    def start_action(self, awaitable_factory, description):
        if not self.backend_ready():
            self.notify("请连接执行后端后继续操作。", title="后端尚未连接", severity="warning")
            return
        if self.harness.busy or self._active_action is not None:
            self.notify("当前任务正在执行，可以取消当前任务后继续。", title="任务进行中", severity="warning")
            return
        self._last_status = description
        self.begin_activity(description)
        self._active_action = self.run_worker(self.execute_action(awaitable_factory), name=description, group="harness", exit_on_error=False)
        self.refresh_status()

    async def execute_action(self, awaitable_factory):
        result = await awaitable_factory()
        self.refresh_instances()
        self.refresh_status()
        return result

    def on_worker_state_changed(self, event: Worker.StateChanged):
        if not self.is_running:
            return
        worker = event.worker
        if worker is self._connection_worker and worker.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            self._connection_worker = None
            self.end_activity()
            if worker.state == WorkerState.ERROR:
                self._last_status = f"执行后端连接失败：{worker.error}"
                self.query_one("#run-log", RichLog).write(Text(self._last_status, style="#fca5a5"))
                self.notify(str(worker.error), title="后端连接失败", severity="error", timeout=10, markup=False)
            self.refresh_status()
            return
        if worker is self._active_action and worker.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            self._active_action = None
            self.clear_stream()
            self.end_activity()
            if worker.state == WorkerState.ERROR:
                self._last_status = f"操作失败：{worker.error}"
                self.query_one("#run-log", RichLog).write(Text(self._last_status, style="#fca5a5"))
                self.notify(str(worker.error), title="操作失败", severity="error", timeout=10, markup=False)
            elif worker.state == WorkerState.CANCELLED:
                self._last_status = "任务已取消，已有记录已经保存"
            else:
                task_status = self.harness.state.get("status")
                if worker.name == "正在准备独立运行环境":
                    environment = (self.harness.current_instance or {}).get("environment") or {}
                    self._last_status = environment.get("summary") or "声明的运行环境已准备，可以运行当前示例"
                elif task_status == "blocked":
                    self._last_status = "智能体任务阻塞，原因与已有证据已经保存"
                elif task_status == "completed" and self.harness.task.get("kind") == "conversation":
                    self._last_status = "对话完成，可以继续输入任务"
                elif task_status == "completed" and self.harness.task.get("kind") == "maintenance":
                    progress = self.harness.state.get("maintenance_progress") or {}
                    self._last_status = "安装与运维任务完成，真实环境检查通过" if progress.get("phase") == "completed" else "环境任务结束，检查证据尚未完整"
                elif task_status == "completed":
                    verification = self.harness.state.get("verification") or {}
                    verified = verification.get("passed") is True and not verification.get("stale") and verification.get("status") != "needs_verification"
                    self._last_status = "任务完成，本轮独立验证通过" if verified else "任务结束，请查看本轮独立验证报告"
                else:
                    self._last_status = "操作完成，可以继续输入任务"
            self.refresh_status()
        elif worker.state == WorkerState.ERROR:
            self._last_status = f"操作失败：{worker.error}"
            self.notify(str(worker.error), title="操作失败", severity="error", timeout=10, markup=False)

    @on(Input.Changed, "#example-filter")
    def filter_examples(self, event: Input.Changed):
        self.refresh_examples(event.value)

    @on(DataTable.RowHighlighted, "#example-table")
    def select_example(self, event: DataTable.RowHighlighted):
        self.selected_example = self.examples[int(event.row_key.value)]
        self.refresh_status()

    @on(DataTable.RowSelected, "#example-table")
    def choose_example(self, event: DataTable.RowSelected):
        self.selected_example = self.examples[int(event.row_key.value)]
        self.action_create_instance()

    @on(Select.Changed, "#instance-select")
    def choose_instance(self, event: Select.Changed):
        if self._refreshing or event.value is Select.BLANK or self.harness.busy:
            return
        if self.harness.current_instance and self.harness.current_instance["instance_id"] == event.value:
            return
        self.start_action(lambda: self.harness.restore_instance(event.value), "正在恢复实例与持久会话")

    @on(Select.Changed, "#model-select")
    def choose_model(self, event: Select.Changed):
        if event.value is Select.BLANK or event.value == self.settings.selected_profile:
            return
        if self.harness.busy or self._active_action is not None:
            event.select.value = self.settings.selected_profile
            self.notify("当前任务执行期间无法切换模型。", title="任务进行中", severity="warning")
            return
        self.start_action(lambda: self.switch_model(event.value), "正在切换后端模型配置")

    async def switch_model(self, profile_name):
        result = self.harness.switch_model(profile_name)
        if inspect.isawaitable(result):
            await result
        self._last_status = f"已选择模型：{self.settings.profile.model}"
        self.refresh_status()

    @on(Button.Pressed, "#create-instance")
    def create_instance_button(self):
        self.action_create_instance()

    def action_create_instance(self):
        if self.selected_example is None:
            self.notify("请选择需要运行的示例。", title="选择示例", severity="warning")
            return
        path = self.selected_example["benchmark_path"]
        self.start_action(lambda: self.harness.create_instance(path), "正在创建独立实例")

    @on(Button.Pressed, "#prepare-environment")
    def prepare_environment_button(self):
        self.action_prepare_environment()

    def require_instance(self):
        if not self.backend_ready():
            self.notify("请连接执行后端后继续操作。", title="后端尚未连接", severity="warning")
            return False
        if self.harness.current_instance is None:
            self.notify("请创建或恢复一个实例。", title="需要实例", severity="warning")
            return False
        return True

    def action_prepare_environment(self):
        if self.require_instance():
            self.start_action(self.harness.prepare_instance, "正在准备独立运行环境")

    @on(Button.Pressed, "#run-example")
    def run_example_button(self):
        self.action_run_example()

    def action_run_example(self):
        if self.require_instance():
            self.query_one("#main-tabs", TabbedContent).active = "conversation"
            self.start_action(self.harness.run_current, "正在运行当前示例")

    @on(Button.Pressed, "#resume-task")
    def resume_task_button(self):
        self.action_resume_task()

    def action_resume_task(self):
        if not self.require_instance():
            return False
        if self.harness.busy or self._active_action is not None:
            self.notify("当前任务正在执行，输入内容已保留。", title="任务进行中", severity="warning")
            return False
        if not self.harness.task or self.harness.state.get("status") not in {"failed", "blocked", "cancelled", "interrupted"}:
            self.notify("当前没有可以继续的未完成任务。", title="继续原任务", severity="warning")
            return False
        self.query_one("#main-tabs", TabbedContent).active = "conversation"
        self.write_conversation("user", "/resume")
        self.start_action(self.harness.resume_current, "正在恢复原任务并核查已有证据")
        return True

    @on(Input.Submitted, "#message-input")
    def input_submitted(self, event: Input.Submitted):
        self.submit_message(event.value)

    @on(Button.Pressed, "#send-message")
    def send_message_button(self):
        self.submit_message(self.query_one("#message-input", Input).value)

    def submit_message(self, text):
        text = text.strip()
        if not text:
            return
        if text == "/compact":
            self.action_compact()
        elif text == "/run":
            self.action_run_example()
        elif text == "/diff":
            self.action_show_diff()
        elif text == "/usage":
            self.action_show_usage()
        elif text == "/verify":
            self.action_show_verification()
        elif text == "/cancel":
            self.action_cancel_task()
        elif text in {"/resume", "/继续"}:
            if not self.action_resume_task():
                return
        elif text == "/help":
            self.query_one("#conversation-log", RichLog).write(Text("快捷指令：/run 运行；/diff 查看修改；/usage 查看用量；/verify 查看验证；/compact 压缩；/cancel 取消；/resume 或 /继续 恢复原任务。"))
        elif text.startswith("/"):
            self.notify("未知指令。输入 /help 查看可用指令。", title="指令提示", severity="warning")
            return
        elif self.require_instance():
            if self.harness.busy or self._active_action is not None:
                self.notify("当前任务正在执行，输入内容已保留。", title="任务进行中", severity="warning")
                return
            self.query_one("#main-tabs", TabbedContent).active = "conversation"
            self.write_conversation("user", text)
            self.start_action(lambda: self.harness.send_message(text), "智能体正在处理任务")
        else:
            return
        self.query_one("#message-input", Input).value = ""

    def action_compact(self):
        if self.require_instance():
            if self.harness.busy or self._active_action is not None:
                self.run_worker(self.harness.compact(), name="提交上下文压缩请求", group="context-control", exit_on_error=False)
            else:
                self.start_action(self.harness.compact, "正在压缩上下文并保存检查点")

    def action_cancel_task(self):
        if not self.backend_ready():
            self.notify("执行后端尚未连接，可以重新连接或退出界面。", title="连接状态", severity="warning")
            return
        self.run_worker(self.cancel_action(), name="取消当前任务", group="control", exit_on_error=False)

    async def cancel_action(self):
        self._cancelling = True
        self.begin_activity("正在取消当前任务并保存记录")
        try:
            self._approvals.clear()
            if isinstance(self.screen, ApprovalScreen):
                await self.screen.dismiss(False)
            await self.harness.cancel()
            if self._active_action:
                self._active_action.cancel()
            self._last_status = "取消请求已处理，已有记录已经保存"
        finally:
            self._cancelling = False
            self.clear_stream()
            self.end_activity()
            self.refresh_status()

    @on(TabbedContent.TabActivated, "#main-tabs")
    def refresh_activated_view(self, event: TabbedContent.TabActivated):
        for log in event.pane.query(ReflowRichLog):
            log.call_after_refresh(log.reflow)
        if event.pane.id == "diff":
            self.action_show_diff()
        elif event.pane.id == "usage":
            self.refresh_usage()
        elif event.pane.id == "verification":
            self.action_show_verification()

    def action_show_diff(self):
        log = self.query_one("#diff-log", RichLog)
        log.clear()
        diff = self.harness.diff()
        log.write(Syntax(diff, "diff", theme="monokai", word_wrap=False) if diff else Text("当前实例尚无代码修改。", style="#94a3b8"))
        self.query_one("#main-tabs", TabbedContent).active = "diff"

    def refresh_usage(self, summary=None):
        summary = self.harness.usage_summary() if summary is None else summary
        lines = ["当前任务", f"任务编号：{summary.get('current_task_id') or '尚未创建任务'}"]
        for label, key in [("任务输入 Token", "task_input_tokens"), ("任务输出 Token", "task_output_tokens"), ("任务模型调用次数", "task_model_calls"), ("任务工具调用次数", "task_tool_calls")]:
            lines.append(f"{label}：{format_number(summary.get(key))}")
        lines.extend(["", "当前会话累计"])
        for label, key in [("累计输入 Token", "input_tokens"), ("累计输出 Token", "output_tokens"), ("缓存读取 Token", "cache_read_tokens"), ("模型调用次数", "model_calls"), ("模型成功次数", "model_calls_succeeded"), ("模型失败次数", "model_calls_failed"), ("模型重试次数", "model_retries"), ("工具调用次数", "tool_calls"), ("工具失败次数", "tool_calls_failed"), ("上下文压缩次数", "compaction_calls"), ("压缩输入 Token（已知部分）", "compaction_input_tokens_known"), ("压缩输出 Token（已知部分）", "compaction_output_tokens_known"), ("缺少实际用量的请求", "missing_usage_calls")]:
            lines.append(f"{label}：{format_number(summary.get(key))}")
        cost = summary.get("cost")
        lines.append(f"费用：{'未知，尚无价格或实际费用记录' if cost is None else f'{cost:.6f}'}")
        lines.append("费用来源：" + {"reported": "服务返回", "estimated": "价格表估算"}.get(summary.get("cost_source"), "未知"))
        if summary.get("missing_usage_calls"):
            lines.append(f"已知输入 Token：{format_number(summary.get('input_tokens_known'))}")
            lines.append(f"已知输出 Token：{format_number(summary.get('output_tokens_known'))}")
        last = summary.get("last_request")
        if last:
            lines.extend(["", "最近模型请求", f"模型：{last.get('model', '未知')}", f"请求阶段：{status_label(last.get('purpose'))}"])
            for label, key in [("实际输入 Token", "input_tokens"), ("实际输出 Token", "output_tokens"), ("缓存读取 Token", "cache_read_tokens"), ("输入 Token 估算", "estimated_input_tokens"), ("输出 Token 估算", "estimated_output_tokens")]:
                lines.append(f"{label}：{format_number(last.get(key))}")
            if last.get("error"):
                lines.append(f"错误证据：{last['error']}")
        lines.extend(["", "缓存读取 Token 包含在输入 Token 内，累计用量不重复相加。", "上下文占用属于请求大小估算，累计 Token 来自实际用量记录。"])
        self.query_one("#usage-detail", Static).update(Text("\n".join(lines)))

    def action_show_usage(self):
        self.refresh_usage()
        self.query_one("#main-tabs", TabbedContent).active = "usage"

    def action_show_verification(self):
        report = self.harness.verification_report()
        lines = ["独立验证报告"]
        if not report:
            lines.extend(["", "当前实例尚无独立验证记录。"])
        else:
            invalidated = report.get("stale") or report.get("status") == "needs_verification"
            if invalidated:
                lines.extend(["当前判定：需要重新验证", "实例内容或环境变化后，先前运行的判定已失效。", ""])
            if report.get("reason"):
                lines.append(str(report["reason"]))
            for label, key in [("运行完成", "execution_completed"), ("结果完整", "results_complete"), ("检查通过", "checks_passed"), ("目标达成", "goal_achieved"), ("修复完成", "repair_completed"), ("最终通过", "passed")]:
                value = report.get(key)
                display_label = f"历史{label}" if report.get("stale") else label
                lines.append(f"{display_label}：{'通过' if value is True else '未通过' if value is False else '尚未确认'}")
            if report.get("run_id"):
                lines.append(f"本轮运行：{report['run_id']}")
                current = self.harness.current_instance
                run = self.harness.runner.get_run(current["instance_id"], report["run_id"]) if current and current.get("last_run_id") == report["run_id"] else None
                launch = (run or {}).get("execution_configuration") or {}
                if launch:
                    lines.append(f"本轮执行入口：{launch.get('entrypoint', '未知')}")
                    enabled = launch.get("mock_runtime_enabled")
                    lines.append("Mock 运行开关：" + ("已启用" if enabled is True else "未启用" if enabled is False else "未知"))
            if report.get("original_goal_ref"):
                lines.extend(["", f"原目标版本：{report.get('goal_id', '未知')}", f"原目标来源：{report['original_goal_ref']}"])
                for source in report.get("unresolved_goal_sources", []):
                    lines.append(f"来源待确认：{source.get('field', '')} · {source.get('reason', '')}")
                lines.extend(str(item) for item in report.get("goal_check_limitations", []))
            lines.extend(["", "判定依据"])
            lines.extend(str(reason) for reason in report.get("reasons", []))
            for check in report.get("checks", []):
                if isinstance(check, dict):
                    name = check.get("name", check.get("id", "检查"))
                    source = "静态检查 · " if check.get("source_kind") == "static_validation" else ""
                    lines.append(f"{source}{CHECK_LABELS.get(name, name)}：{status_label(check.get('status'))}")
                    if check.get("reason"):
                        lines.append(str(check["reason"]))
                    if check.get("file"):
                        lines.append(f"文件：{check['file']}")
                    if check.get("status") in {"FAIL", "ERROR", "WARNING", "SKIP"} and check.get("message"):
                        lines.append(f"原始检查输出：{check['message']}")
                        lines.extend(str(detail) for detail in check.get("details", []))
                else:
                    lines.append(str(check))
        self.query_one("#verification-detail", Static).update(Text("\n".join(lines)))
        self.query_one("#main-tabs", TabbedContent).active = "verification"

    def format_event(self, event):
        event_type = event["event_type"]
        payload = event.get("payload") or {}
        label = EVENT_LABELS.get(event_type, "状态更新")
        details = []
        if payload.get("tool_name") or payload.get("name"):
            details.append(str(payload.get("tool_name") or payload["name"]))
        if payload.get("call_id"):
            details.append(str(payload["call_id"]))
        if payload.get("model"):
            details.append(str(payload["model"]))
        if payload.get("purpose"):
            details.append(status_label(payload["purpose"]))
        if payload.get("stage"):
            details.append(STAGE_LABELS.get(payload["stage"], str(payload["stage"])))
        if event.get("run_id") or payload.get("run_id"):
            details.append(str(event.get("run_id") or payload["run_id"]))
        text = label + (" · " + " · ".join(details) if details else "")
        for key in ("message", "error", "reason", "content", "text", "output"):
            if payload.get(key):
                text += f"\n{payload[key]}"
        if event_type in {"tool.started", "tool.completed", "tool.failed"}:
            text = label + " · " + str(payload.get("name", "")) + " · " + short_text(tool_parameters(payload.get("arguments") or {}))
            if event_type != "tool.started":
                text += "\n" + tool_summary(payload)
        color = "#fca5a5" if event_type.endswith(("failed", "blocked")) else "#6ee7b7" if event_type.endswith(("completed", "compacted", "applied")) else "#93c5fd"
        return text, color

    def on_harness_event(self, message: HarnessEvent):
        if not self.is_running:
            return
        event = message.event
        if event["event_type"] in {"session.started", "session.resumed"} and event.get("session_id") != self._displayed_session_id:
            self.restore_conversation()
        identity = event.get("event_id")
        if identity and identity in self._seen_events:
            return
        if identity:
            self._seen_events.add(identity)
        event_type = event["event_type"]
        payload = event.get("payload") or {}
        if event_type in {"llm.delta", "llm.progress"}:
            self.update_activity(event)
            return
        if event_type == "run.output":
            self.write_internal_event(event)
            return
        text, color = self.format_event(event)
        if event_type in {"agent.message", "assistant.message"}:
            content = assistant_text(payload)
            if content:
                self.write_conversation("assistant", content)
            self.clear_stream()
        else:
            self.write_internal_event(event)
            content = environment_text(event, self.last_run()) if event_type.startswith("environment.") else ""
            if content:
                self.write_conversation("assistant", content)
        self.update_tool_record(event)
        if event_type not in {"llm.usage", "context.preparing", "context.prepared", "context.compaction_queued", "context.compaction_skipped", "run.output", "agent.message", "assistant.message"}:
            self._last_status = text.split("\n", 1)[0]
        self.update_activity(event)
        if event_type in {"run.started", "run.stage_changed", "run.completed", "run.failed", "run.cancelled", "run.pipeline_completed", "run.pipeline_failed"}:
            self.query_one("#run-status", Static).update(Text(text.split("\n", 1)[0]))
        elif event_type == "verification.completed":
            passed = payload.get("passed")
            self.query_one("#run-status", Static).update(Text("独立验证通过" if passed else "独立验证未通过", style="#6ee7b7" if passed else "#fbbf24"))
        if payload.get("context_fraction") is not None:
            self._last_context_fraction = payload["context_fraction"]
        if event_type == "approval.requested":
            self._approvals.append(payload)
            self.open_next_approval()
        active_tab = self.query_one("#main-tabs", TabbedContent).active
        if event_type == "patch.applied" and active_tab == "diff":
            self.action_show_diff()
        if event_type == "verification.completed" and active_tab == "verification":
            self.action_show_verification()
        if event_type in {"approval.requested", "session.started", "session.resumed", "agent.completed", "agent.blocked", "agent.cancelled", "agent.failed", "run.completed", "run.failed", "run.cancelled", "verification.completed"}:
            self.refresh_status()
        else:
            self._status_dirty = True

    def open_next_approval(self):
        if self._cancelling or self._approval_open or not self._approvals:
            return
        self._approval_open = True
        request = self._approvals.pop(0)
        self.push_screen(ApprovalScreen(request), lambda approved: self.resolve_approval(request["approval_id"], approved))

    def resolve_approval(self, approval_id, approved):
        self._approval_open = False
        if self._cancelling:
            return
        if not self.backend_ready():
            self._approvals.clear()
            return
        self.run_worker(self.harness.approve(approval_id, approved), name="记录操作审批", group="approval", exit_on_error=False)
        self.open_next_approval()

    def action_request_exit(self):
        if self.backend_ready() and (self.harness.busy or self._active_action):
            self.push_screen(ExitScreen(), self.exit_decision)
        else:
            self.run_worker(self.close_session(), name="保存并退出", group="control", exit_on_error=False)

    def exit_decision(self, confirmed):
        if confirmed:
            self.run_worker(self.close_session(), name="保存并退出", group="control", exit_on_error=False)

    async def close_session(self):
        await self.close_harness()
        self.exit()

    async def close_harness(self):
        if self._harness_closed:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self.harness.close())
        await asyncio.shield(self._close_task)
        self._harness_closed = True


def run_app(settings, harness=None):
    app = IanvsAgentApp(settings, harness)
    try:
        app.run()
        if app.return_code:
            raise SystemExit(app.return_code)
    finally:
        if not app._harness_closed:
            asyncio.run(app.close_harness())
