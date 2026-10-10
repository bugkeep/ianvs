# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json

from rich.syntax import Syntax
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ianvs_agent.app.records import tool_display_status, tool_outcome, tool_summary


class ApprovalScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "reject", "拒绝操作")]

    def __init__(self, request):
        super().__init__()
        self.request = request

    def operation_view(self):
        operation = dict(self.request["resolved_operation"])
        versions = operation.get("file_versions")
        if isinstance(versions, dict):
            # 审批保留完整版本记录，界面通过正式引用展示其依据。
            operation["file_versions"] = {"count": len(versions), **{key: operation[key] for key in ("snapshot_fingerprint", "source_revision", "evidence_ref") if key in operation}}
        return operation

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static("操作审批", classes="dialog-title")
            description = "请检查下方修改内容。批准后将修改当前独立实例中的文件。" if self.request.get("diff") else self.request.get("description", "请检查具体操作和参数。")
            yield Static(Text(description), id="approval-description")
            yield Static(Text(f"工具：{self.request['tool_name']}\n审批编号：{self.request['approval_id']}"))
            with VerticalScroll(id="approval-details"):
                if self.request.get("diff"):
                    yield Static(Syntax(self.request["diff"], "diff", theme="monokai", word_wrap=True))
                else:
                    yield Static(Syntax(json.dumps(self.request["arguments"], ensure_ascii=False, indent=2), "json", theme="monokai", word_wrap=True))
                if self.request.get("resolved_operation") is not None:
                    yield Static("实际操作与版本依据", classes="field-label")
                    yield Static(Syntax(json.dumps(self.operation_view(), ensure_ascii=False, indent=2), "json", theme="monokai", word_wrap=True), id="approval-operation")
            yield Static("审批仅适用于此处展示的操作与参数。", classes="muted")
            with Horizontal(classes="dialog-actions"):
                yield Button("拒绝操作", id="reject", variant="error")
                yield Button("批准操作", id="approve", variant="success")

    @on(Button.Pressed, "#approve")
    def approve(self):
        self.dismiss(True)

    @on(Button.Pressed, "#reject")
    def reject(self):
        self.dismiss(False)

    def action_reject(self):
        self.dismiss(False)


class ExitScreen(ModalScreen[bool]):
    BINDINGS = [("escape", "continue_session", "继续会话")]

    def compose(self) -> ComposeResult:
        with Vertical(id="exit-dialog"):
            yield Static("结束当前会话", classes="dialog-title")
            yield Static("当前任务仍在执行。退出将取消任务，并保存已有会话与运行记录。")
            with Horizontal(classes="dialog-actions"):
                yield Button("继续会话", id="continue-session", variant="primary")
                yield Button("取消任务并退出", id="exit-session", variant="error")

    @on(Button.Pressed, "#continue-session")
    def continue_session(self):
        self.dismiss(False)

    @on(Button.Pressed, "#exit-session")
    def exit_session(self):
        self.dismiss(True)

    def action_continue_session(self):
        self.dismiss(False)


class ToolDetailsScreen(ModalScreen[None]):
    BINDINGS = [("escape", "close_details", "关闭详情")]

    def __init__(self, record):
        super().__init__()
        self.record = record

    def compose(self) -> ComposeResult:
        record = self.record
        with Vertical(id="tool-dialog"):
            yield Static("工具执行详情", classes="dialog-title")
            yield Static(Text(f"工具：{record.get('name', '未知工具')}\n调用编号：{record['call_id']}\n执行编号：{record.get('operation_id') or '尚未取得'}"))
            with VerticalScroll(id="tool-details"):
                yield Static(Text(f"工具调用状态：{record.get('status', 'unknown')}\n结果显示状态：{tool_display_status(record) or 'unknown'}\n{tool_summary(record)}"), id="tool-result-status")
                if tool_outcome(record):
                    yield Static(Syntax(json.dumps(tool_outcome(record), ensure_ascii=False, indent=2), "json", theme="monokai", word_wrap=True), id="tool-outcome")
                yield Static("参数", classes="field-label")
                yield Static(Syntax(json.dumps(record.get("arguments", {}), ensure_ascii=False, indent=2), "json", theme="monokai", word_wrap=True))
                if record.get("error"):
                    yield Static("错误", classes="field-label")
                    yield Static(Text(str(record["error"]), style="#fca5a5"), id="tool-error")
                if record.get("data") is not None:
                    output = json.dumps(record["data"], ensure_ascii=False, indent=2)
                    yield Static("结果预览", classes="field-label")
                    yield Static(Syntax(output[:6000], "json", theme="monokai", word_wrap=True), id="tool-output-preview")
                    if len(output) > 6000 or record.get("truncated"):
                        yield Static("预览仅展示部分结果，完整内容请查看正式记录。", classes="muted")
                yield Static("正式记录引用", classes="field-label")
                references = [record.get("record_ref"), record.get("full_output_ref"), tool_outcome(record).get("full_run_ref")]
                yield Static(Text("\n".join(dict.fromkeys(str(reference) for reference in references if reference)) or "尚未取得完成记录"), id="tool-output-reference")
            with Horizontal(classes="dialog-actions"):
                yield Button("关闭详情", id="close-tool-details", variant="primary")

    @on(Button.Pressed, "#close-tool-details")
    def close_details_button(self):
        self.dismiss(None)

    def action_close_details(self):
        self.dismiss(None)
