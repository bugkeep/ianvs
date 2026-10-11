# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import hashlib
import inspect
import json
import re
import time
import uuid

from jsonschema import Draft202012Validator, ValidationError

class ToolRegistry:
    def __init__(self, harness):
        self.harness = harness
        self.tools = {}
        self.pending = {}

    def register(self, name, description, properties, required, handler, modifies=False, approval_context=None):
        schema = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
        Draft202012Validator.check_schema(schema)
        self.tools[name] = {"schema": schema, "description": description, "handler": handler, "modifies": modifies, "approval_context": approval_context}

    def definitions(self):
        return [{"type": "function", "function": {"name": name, "description": item["description"], "parameters": item["schema"]}} for name, item in self.tools.items()]

    def is_read_only(self, name, arguments=None):
        if name not in self.tools:
            return False
        modifies = self.tools[name]["modifies"]
        return not modifies(arguments) if callable(modifies) and arguments is not None else modifies is False

    async def _resolved_approval(self, item, arguments):
        resolver = item.get("approval_context")
        if resolver is None:
            return None
        result = resolver(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return copy.deepcopy(result)

    async def execute(self, call):
        name = call["name"]
        arguments = call["arguments"]
        started = time.monotonic()
        await self.harness._run_sync(lambda: self.harness.usage.check_budget(self.harness.settings, reserve_tool_calls=1))
        operation_id = await self.harness._run_sync(self.harness.store.allocate_id, "TOOL")
        call = {**call, "operation_id": operation_id}
        events = self.harness.events
        try:
            await self.harness._run_sync(events.emit, "tool.started", {"call_id": call["id"], "operation_id": operation_id, "name": name, "arguments": arguments})
            if name not in self.tools:
                return await self.harness._run_sync(self._failure, call, "未登记的工具", started)
            item = self.tools[name]
            Draft202012Validator(item["schema"]).validate(arguments)
            if self.harness.state.get("status") == "running" and not self.is_read_only(name, arguments):
                await self.harness._run_sync(self.harness.completion.ensure_action_allowed)
            resolved = None
            if not self.is_read_only(name, arguments):
                resolved = await self._resolved_approval(item, arguments)
                if resolved is not None:
                    call = {**call, "approval_context": resolved}
                approved = await self._approval(call)
                if not approved:
                    return await self.harness._run_sync(self._failure, call, "用户拒绝此操作", started, True)
                if resolved != await self._resolved_approval(item, arguments):
                    raise PermissionError("审批后的操作依据或文件版本发生变化，需要重新生成操作并审批")
            if resolved is not None:
                result = await item["handler"](**arguments, _approved=resolved)
            else:
                result = await item["handler"](**arguments)
        except asyncio.CancelledError:
            recorded = await self.harness._run_sync(lambda: any(record.get("operation_id") == operation_id for record in self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")))
            if not recorded:
                await self.harness._run_sync(self._failure, call, "工具操作已取消，未取得完成证据", started, False, "cancelled")
            raise
        except (ValidationError, PermissionError, FileNotFoundError, ValueError, RuntimeError, TimeoutError, re.error) as error:
            return await self.harness._run_sync(self._failure, call, str(error), started)
        elapsed = time.monotonic() - started
        record = {"operation_id": operation_id, "call_id": call["id"], "task_id": self.harness.task.get("task_id"), "name": name, "arguments": arguments, "status": "success", "data": result, "error": None, "duration_seconds": elapsed}
        if call.get("approval_context") is not None:
            record["approval_context"] = call["approval_context"]
        return await self.harness._run_sync(self._complete, record)

    def _complete(self, record):
        preview = self._preview(record)
        self._save(record)
        if self.harness.usage:
            self.harness.usage.record_tool("success", record["duration_seconds"])
        self.harness.events.emit("tool.completed", preview)
        return preview

    def _failure(self, call, error, started, denied=False, status=None):
        record = {"operation_id": call["operation_id"], "call_id": call["id"], "name": call["name"], "arguments": call["arguments"], "status": status or ("denied" if denied else "error"), "data": None, "error": error, "duration_seconds": time.monotonic() - started}
        if call.get("approval_context") is not None:
            record["approval_context"] = call["approval_context"]
        self._save(record)
        if self.harness.usage:
            self.harness.usage.record_tool(record["status"], record["duration_seconds"])
        preview = self._preview(record)
        self.harness.events.emit("tool.failed", preview)
        return preview

    def _save(self, record):
        relative = self.harness.session_relative_path + "/tools.jsonl"
        self.harness.store.append(relative, record)

    def _preview(self, record):
        public = {key: value for key, value in record.items() if key != "approval_context"}
        outcome = self._outcome(record)
        if outcome:
            public["outcome"] = outcome
        if record.get("approval_context") is not None:
            public["approval_bound"] = True
        text = json.dumps(record["data"], ensure_ascii=False, indent=2)
        limit = self.harness.settings.tool_output_chars
        if len(text) <= limit:
            return {**public, "truncated": False}
        relative = record.get("full_output_ref")
        if relative is None:
            output_id = self.harness.store.allocate_id("OUTPUT")
            relative = self.harness.session_relative_path + f"/tool-output/{output_id}.json"
            self.harness.store.write(relative, record["data"])
        elif not self.harness.store.path(relative).is_file():
            raise FileNotFoundError("已登记的工具全文证据缺失：" + relative)
        record["full_output_ref"] = relative
        return {**public, "data": text[:limit], "truncated": True, "total_chars": len(text), "full_output_ref": relative, "continuation": "使用 read_evidence 读取完整结果或指定区间"}

    @staticmethod
    def _outcome(record):
        data = record.get("data")
        if not isinstance(data, dict):
            return {}
        name = record.get("name")
        if name == "run_example":
            run = data.get("run") or {}
            verification = data.get("verification") or {}
        elif name == "verify_run":
            run = {}
            verification = data
        else:
            return {}
        # 长输出保留独立的运行与验证概要，全文继续使用正式证据引用。
        return {
            "run_id": run.get("run_id") or verification.get("run_id"),
            "run_status": run.get("status"),
            "exit_code": run.get("exit_code"),
            "verification_passed": verification.get("passed"),
            "verification_stale": bool(verification.get("stale") or verification.get("status") == "needs_verification"),
            "repair_completed": verification.get("repair_completed"),
            "full_run_ref": run.get("full_run_ref"),
        }

    async def _approval(self, call):
        arguments = call["arguments"]
        identity = hashlib.sha256(json.dumps(call, sort_keys=True).encode()).hexdigest()
        approval_id = "approval-" + uuid.uuid4().hex[:12]
        policy = self.harness.settings.approval_policy
        record = {"approval_id": approval_id, "call_id": call["id"], "tool_name": call["name"], "arguments": copy.deepcopy(arguments), "identity": identity, "description": self.tools[call["name"]]["description"], "diff": arguments.get("patch", "")}
        if call.get("approval_context") is not None:
            record["resolved_operation"] = copy.deepcopy(call["approval_context"])
        if policy == "deny":
            self.harness.events.emit("approval.decided", {**record, "approved": False, "source": "fixed_policy"})
            return False
        if policy == "auto_scoped":
            self.harness.events.emit("approval.decided", {**record, "approved": True, "source": "fixed_policy"})
            return True
        if policy != "manual":
            raise ValueError("未识别的审批策略")
        future = asyncio.get_running_loop().create_future()
        self.pending[approval_id] = {"future": future, "record": record}
        self.harness.events.emit("approval.requested", record)
        try:
            approved = await future
        finally:
            self.pending.pop(approval_id, None)
        if hashlib.sha256(json.dumps(call, sort_keys=True).encode()).hexdigest() != identity:
            raise PermissionError("审批后工具参数发生变化，需要重新审批")
        return approved

    def approve(self, approval_id, approved):
        if approval_id not in self.pending:
            raise ValueError("审批请求已失效")
        item = self.pending[approval_id]
        self.harness.events.emit("approval.decided", {**item["record"], "approved": bool(approved), "source": "user"})
        item["future"].set_result(bool(approved))

    def cancel_approvals(self):
        for item in tuple(self.pending.values()):
            if not item["future"].done():
                item["future"].cancel()
