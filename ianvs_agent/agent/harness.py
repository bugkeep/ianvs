# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ianvs_agent.agent.events import EventBus
from ianvs_agent.context.builder import ContextManager
from ianvs_agent.llm.provider import Provider
from ianvs_agent.llm.usage import UsageTracker
from ianvs_agent.message_types import format_task_result, observation_statement, tool_evidence_references
from ianvs_agent.runtime import ExampleCatalog, EnvironmentManager, InstanceManager, Runner, Verifier
from ianvs_agent.storage.local_store import LocalStore
from ianvs_agent.tools.filesystem import register_tools
from ianvs_agent.tools.registry import ToolRegistry


OBSERVATION_CORRECTION = "上一条 Observe 响应未通过 JSON 格式或字段检查。请重新核对同一轮已完成工具结果，只返回更短且完整的 JSON object，包含 facts、hypotheses、next_goal、decision；不要重复工具、复述完整源码或补造未证实事实。"


def now():
    return datetime.now(timezone.utc).isoformat()


class AgentHarness:
    def __init__(self, settings):
        self.settings = settings
        self.store = LocalStore(settings.state_root)
        self.events = EventBus(self.store)
        self.catalog = ExampleCatalog(settings.project_root)
        self.instances = InstanceManager(settings, self.store, self.events)
        self.environment = EnvironmentManager(settings, self.store, self.events)
        self.runner = Runner(settings, self.store, self.events)
        self.verifier = Verifier(settings, self.store, self.events)
        self.current_instance = None
        self.session = None
        self.session_relative_path = None
        self.task = {}
        self.state = {}
        self.usage = None
        self.provider = None
        self.context = None
        self.busy = False
        self._active_task = None
        self._cancel_requested = False
        self._compact_requested = False
        self._execution_started = None
        self._execution_task_id = None
        self._providers = []
        self._session_save_guard = asyncio.Lock()
        self._session_write_lock = threading.RLock()
        self.tools = ToolRegistry(self)
        self.registry = self.tools
        register_tools(self, self.tools)
        latest = self.store.read("last_session.json", {})
        if latest:
            self.select_instance(latest["instance_id"])

    def _require_instance(self):
        if self.current_instance is None:
            raise ValueError("请创建或恢复一个实例")

    def select_instance(self, instance_id):
        if self.busy:
            raise RuntimeError("任务执行期间无法切换实例")
        return self._select_instance(instance_id)

    def _select_instance(self, instance_id):
        self.current_instance = self.instances.get(instance_id)
        pointer = self.store.read(f"instances/{instance_id}/latest_session.json", {})
        if pointer:
            self.session_relative_path = pointer["path"]
            snapshot = self.store.read(self.session_relative_path + "/session_snapshot.json")
            self.session = snapshot["session"] if snapshot is not None else self.store.read(self.session_relative_path + "/session.json")
            if self.session is None:
                raise ValueError("实例会话指针缺少对应记录")
            if self.session.get("instance_id") != instance_id:
                raise ValueError("会话记录身份与所选实例不一致")
            saved_profile = self.session.get("model_profile")
            if saved_profile in self.settings.profiles:
                self.settings.selected_profile = saved_profile
            self.task = snapshot["task"] if snapshot is not None else self.store.read(self.session_relative_path + "/task.json", {})
            self.state = snapshot["state"] if snapshot is not None else self.store.read(self.session_relative_path + "/state.json", {})
            self._messages = snapshot["history"] if snapshot is not None else self.store.read(self.session_relative_path + "/history.json", [])
            self._repair_open_tool_pairs()
            if self.state.get("status") in {"running", "waiting_approval"}:
                self.state["status"] = "interrupted"
                self.state["interruption_reason"] = "上次会话在任务完成前退出，正式记录已恢复"
        else:
            session_id = self.store.allocate_id("session")
            self.session_relative_path = f"instances/{instance_id}/sessions/{session_id}"
            self.session = {"schema_version": 1, "session_id": session_id, "instance_id": instance_id, "created_at": now(), "model_profile": self.settings.selected_profile}
            self.task, self.state, self._messages = {}, {}, []
            self.store.write(f"instances/{instance_id}/latest_session.json", {"path": self.session_relative_path})
        self.events.bind(instance_id=instance_id, session_id=self.session["session_id"], task_id=self.task.get("task_id"), environment_id=(self.current_instance.get("environment") or {}).get("environment_id"), run_id=None)
        self._connect_provider()
        self._save_session()
        self.store.write("last_session.json", {"instance_id": instance_id, "path": self.session_relative_path})
        self.events.emit("session.started", {"restored": bool(pointer), "model": self.settings.profile.model})

    def _repair_open_tool_pairs(self):
        pending = {}
        for message in self._messages:
            if message["role"] == "assistant":
                pending = {call["id"]: call for call in message.get("tool_calls", [])}
            elif message["role"] == "tool":
                pending.pop(message["tool_call_id"], None)
        records = self.store.read_events(self.session_relative_path + "/tools.jsonl")
        open_round = self.state.get("open_tool_round") or {}
        records = records[open_round["tools_start_index"]:] if open_round else []
        for call_id, call in pending.items():
            arguments = json.loads(call["function"]["arguments"])
            matches = [record for record in records if record["call_id"] == call_id and record["name"] == call["function"]["name"] and record["arguments"] == arguments]
            result = self.tools._preview(matches[-1]) if matches else {"call_id": call_id, "name": call["function"]["name"], "status": "interrupted", "error": "会话中断，未取得本次工具完成证据；需要核查当前状态后重新安排动作"}
            self._messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False)})
        self.state.pop("open_tool_round", None)

    def _connect_provider(self):
        if self.provider:
            self._providers.append(self.provider)
        self.usage = UsageTracker(self.store, self.session_relative_path, self.events)
        self.provider = Provider(self.settings.profile, self.events, self.usage)
        self.context = ContextManager(self.settings, self.store, self.events, self.provider)

    def switch_model(self, profile_name):
        if self.busy:
            raise RuntimeError("任务执行期间无法切换模型")
        if profile_name not in self.settings.profiles:
            raise ValueError("模型配置不存在")
        self.settings.selected_profile = profile_name
        if self.session:
            self.session["model_profile"] = profile_name
            self._connect_provider()
            self._save_session()
        self.events.emit("model.selected", {"model": self.settings.profile.model, "profile": profile_name})

    async def create_instance(self, benchmark_path):
        return await self._action(self._create(benchmark_path))

    async def _create(self, benchmark_path):
        instance = await asyncio.to_thread(self.instances.create, benchmark_path)
        await asyncio.to_thread(self._select_instance, instance["instance_id"])
        return self.current_instance

    async def restore_instance(self, instance_id):
        return await self._action(self._restore(instance_id))

    async def _restore(self, instance_id):
        await asyncio.to_thread(self._select_instance, instance_id)
        return self.current_instance

    async def prepare_instance(self):
        self._require_instance()
        return await self._action(self._prepare())

    async def _prepare(self):
        try:
            return await self.environment.prepare(self.current_instance)
        finally:
            self.current_instance = self.instances.get(self.current_instance["instance_id"])
            self.events.bind(environment_id=(self.current_instance.get("environment") or {}).get("environment_id"))
            await self._save_session_async()

    def _ensure_task(self, goal, kind="conversation"):
        status = self.state.get("status")
        finished_conversation = self.task.get("kind") in {"conversation", "maintenance"} and status in {"completed", "cancelled", "failed", "blocked", "interrupted"}
        finished_run = self.task.get("kind") == "run" and kind != "run" and status in {"completed", "failed", "cancelled", "blocked", "interrupted"}
        if self.task and status != "completed" and not finished_conversation and not finished_run:
            return
        acceptance = copy.deepcopy(self.task.get("acceptance", {})) if finished_run or (kind == "repair" and self.task.get("kind") == "maintenance") else {}
        if self.task:
            self.store.write(self.session_relative_path + f"/tasks/{self.task['task_id']}.json", {"task": self.task, "state": self.state})
        task_id = self.store.allocate_id("task")
        self.task = {"schema_version": 1, "task_id": task_id, "kind": kind, "instance_id": self.current_instance["instance_id"], "goal": goal, "benchmark_path": self.current_instance["benchmark_path"], "code_root": self.current_instance["code_root"], "readme_path": self.current_instance.get("readme_path"), "acceptance": {"original_benchmark_path": self.current_instance["benchmark_path"], **acceptance}, "created_at": now()}
        self.state = {"status": "pending", "facts": [], "hypotheses": [], "changes": [], "verification": {}, "repair_attempts": 0}
        self.current_instance = self.instances.get(self.current_instance["instance_id"])
        previous_run = self.current_instance.get("last_run_id")
        if previous_run:
            run = self.store.read(f"instances/{self.current_instance['instance_id']}/runs/{previous_run}/run.json")
            if run:
                self.state["last_run_id"] = previous_run
                self.state["last_run"] = self.run_summary(run)
                if run["status"] == "failed":
                    self.state["failure"] = self._failure_summary(run)
        self.state["verification"] = self.current_instance.get("verification") or {}
        self.events.bind(task_id=task_id)
        self._save_session()

    async def run_current(self):
        self._require_instance()
        if self.busy:
            raise RuntimeError("当前任务正在执行")
        self._ensure_task("运行所选 Example 的原始评测任务，独立检查 TestCase、结果、指标与验收条件", kind="run")
        return await self._action(self._run_current(), task_budget=True)

    async def _run_current(self):
        result = await self._run()
        passed = result["verification"].get("passed") is True
        if self.task.get("kind") == "repair":
            passed = self._repair_is_complete()
        await self._finish("completed" if passed else "failed", "当前运行已通过独立验证" if passed else "当前运行未通过独立验证，运行证据已保存")
        return result

    @staticmethod
    def run_summary(run):
        result = {key: value for key, value in run.items() if key not in {"source_manifest", "config_snapshot", "events", "environment_snapshot"}}
        result["full_run_ref"] = f"instances/{run['instance_id']}/runs/{run['run_id']}/run.json"
        for field in ("testcases", "artifacts", "metrics"):
            value = result.get(field)
            if isinstance(value, (list, dict)):
                result[field + "_count"] = len(value)
                result[field + "_truncated"] = len(value) > 20
                if isinstance(value, dict):
                    result[field] = dict(list(value.items())[:20])
                else:
                    result[field] = value[:20]
        return result

    def _failure_summary(self, run):
        evidence = []
        for key in ("stderr_ref", "stdout_ref"):
            reference = run.get(key)
            if reference and self.store.path(reference).exists():
                text = self.store.path(reference).read_text(encoding="utf-8", errors="replace")
                evidence.append({"evidence_ref": reference, "tail": text[-6000:], "total_chars": len(text), "truncated": len(text) > 6000})
        return {"run_id": run["run_id"], "status": run["status"], "exit_code": run.get("exit_code"), "evidence": evidence}

    async def _run(self):
        self.state["repair_attempts"] = self.state.get("repair_attempts", 0) + 1
        await self._save_session_async()
        run = await self.runner.run(self.current_instance)
        self.current_instance = self.instances.get(self.current_instance["instance_id"])
        self.state["last_run_id"] = run["run_id"]
        self.state["last_run"] = self.run_summary(run)
        self.state["failure"] = None
        if run["status"] == "cancelled" or self._cancel_requested:
            self.state["verification"] = {"passed": False, "status": "cancelled", "run_id": run["run_id"], "reason": "当前运行取消，未取得完整验证证据"}
            await self._save_session_async()
            raise asyncio.CancelledError
        if run["status"] != "completed":
            self.state["failure"] = self._failure_summary(run)
        verification = await self.verifier.verify(self.current_instance, run, self.task.get("acceptance"))
        self.state["verification"] = verification
        self.current_instance = self.instances.get(self.current_instance["instance_id"])
        await self._save_session_async()
        return {"run": self.state["last_run"], "verification": verification, "failure": self.state["failure"]}

    async def send_message(self, text):
        self._require_instance()
        if self.busy:
            raise RuntimeError("当前任务正在执行")
        if not text.strip():
            raise ValueError("任务内容不能为空")
        if text.strip().rstrip("。.!！").lower() in {"继续", "继续任务", "继续当前任务", "继续原任务", "resume", "continue", "/resume"} and self.task and self.state.get("status") in {"failed", "blocked", "cancelled", "interrupted"}:
            return await self.resume_current(text.strip())
        requested_repair = any(word in text.lower() for word in ("修复", "repair", "fix "))
        self._ensure_task(text.strip(), kind="repair" if requested_repair else "conversation")
        if requested_repair:
            self.task["kind"] = "repair"
            self.task["acceptance"]["repair_task_id"] = self.task["task_id"]
            self.task["acceptance"]["repair_started_at"] = self.task["created_at"]
        self._messages.append({"role": "user", "content": text.strip()})
        self._save_session()
        return await self._action(self._loop(), task_budget=True)

    async def resume_current(self, text=None):
        self._require_instance()
        if self.busy:
            raise RuntimeError("当前任务正在执行")
        if not self.task or self.state.get("status") not in {"failed", "blocked", "cancelled", "interrupted"}:
            raise ValueError("当前没有可以继续的未完成任务")
        # 只增加续作动作，原目标、验收、工具记录和累计用量均沿用。
        self.usage.check_budget(self.settings)
        if self.task.get("kind") == "conversation":
            events = await self._run_sync(self.store.read_events, self.session_relative_path + "/events.jsonl")
            operations = {event["payload"].get("operation_id") for event in events if event.get("task_id") == self.task["task_id"] and event.get("event_type") in {"tool.completed", "tool.failed"}}
            await self._run_sync(self._update_task_progress, [{"operation_id": operation} for operation in operations if operation])
        self.state["resume_count"] = self.state.get("resume_count", 0) + 1
        self._messages.append({"role": "user", "content": text or "继续原任务，使用已有正式证据核查当前状态并完成剩余动作；保持原目标、验收和累计用量记录。"})
        await self._save_session_async()
        return await self._action(self._loop(), task_budget=True)

    def _repair_is_complete(self):
        instance = self.instances.get(self.current_instance["instance_id"])
        verification = instance.get("verification") or {}
        verified_task = (verification.get("acceptance") or {}).get("repair_task_id")
        complete = (
            verification.get("passed") is True
            and verification.get("repair_completed") is True
            and not verification.get("stale")
            and not instance.get("needs_verification")
            and verification.get("run_id") == instance.get("last_run_id")
            and verification.get("run_id") == self.state.get("last_run_id")
            and verified_task == self.task.get("task_id")
        )
        if complete:
            self.current_instance = instance
            self.state["verification"] = verification
        return complete

    def _requires_observation(self, calls):
        return self.task.get("kind") != "conversation" or any(not self.tools.is_read_only(call["name"], call.get("arguments")) for call in calls)

    async def _loop(self):
        self.state["status"] = "running"
        self.state.pop("error", None)
        self.state.pop("result", None)
        await self._save_session_async()
        while not self._cancel_requested:
            changes = await self._run_sync(self.instances.check_changes, self.current_instance["instance_id"], False)
            self.current_instance = self.instances.get(self.current_instance["instance_id"])
            if self.current_instance.get("needs_verification"):
                self.state["verification"] = self.current_instance.get("verification") or {"passed": False, "status": "needs_verification"}
            if changes.get("source_updates"):
                self.state["source_updates"] = changes["source_updates"]
            if self._compact_requested:
                await self._compact()
            self.usage.check_budget(self.settings)
            if self.task.get("kind") == "repair" and await self._run_sync(self._repair_is_complete):
                return await self._finish("completed", "原始任务已经通过本轮独立验证")
            if self.task.get("kind") == "maintenance" and await self._run_sync(self._maintenance_is_complete):
                self.state["maintenance_progress"]["phase"] = "completed"
                return await self._finish("completed", "当前安装与运维步骤已通过真实环境检查")
            response = await self.context.complete(self.task, self.state, self._messages, self.tools.definitions(), role="plan", purpose="plan")
            await self._record_assistant(response, "plan")
            calls = response["tool_calls"]
            if calls:
                results = []
                for index, call in enumerate(calls):
                    try:
                        self.usage.check_budget(self.settings)
                        result = await self.tools.execute(call)
                    except BaseException:
                        self._repair_open_tool_pairs()
                        await self._save_session_async()
                        raise
                    self._messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
                    results.append(result)
                    if index == len(calls) - 1:
                        self.state.pop("open_tool_round", None)
                    await self._save_session_async()
                await self._run_sync(self._update_task_progress, results)
                await self._save_session_async()
                if self._repair_is_complete():
                    return await self._finish("completed", "原始任务已经通过本轮独立验证")
                if self.task.get("kind") == "maintenance" and await self._run_sync(self._maintenance_is_complete):
                    self.state["maintenance_progress"]["phase"] = "completed"
                    return await self._finish("completed", "当前安装与运维步骤已通过真实环境检查")
                if not self._requires_observation(calls):
                    self.state.pop("observation_decision", None)
                    self.state["next_goal"] = "继续完成当前用户授权目标；根据本轮真实工具结果安排尚未完成的动作，普通问题已有充分证据时直接答复"
                    continue
            else:
                if self._repair_is_complete():
                    return await self._finish("completed", "原始任务已经通过本轮独立验证")
                if self.task.get("kind") == "conversation":
                    return await self._finish("completed", response["content"] or "对话任务已完成")
                if self.task.get("kind") == "maintenance":
                    if await self._run_sync(self._maintenance_is_complete):
                        self.state["maintenance_progress"]["phase"] = "completed"
                        return await self._finish("completed", response["content"] or "当前安装与运维步骤已通过真实环境检查")
            observation = await self._observe(allow_text=not calls)
            if self._compact_requested:
                await self._compact()
            if self._repair_is_complete():
                return await self._finish("completed", "原始任务已经通过本轮独立验证")
            if self.task.get("kind") == "maintenance" and await self._run_sync(self._maintenance_is_complete):
                self.state["maintenance_progress"]["phase"] = "completed"
                return await self._finish("completed", "当前安装与运维步骤已通过真实环境检查")
            if observation.get("decision") == "blocked":
                return await self._finish("blocked", observation.get("next_goal") or "当前任务缺少必要条件")
        raise asyncio.CancelledError

    async def _observe(self, previous=None, allow_text=False):
        if previous is not None:
            reference = self.session_relative_path + f"/requests/{previous['usage']['request_group_id']}.json"
            saved = await self._run_sync(self.store.read, reference, {})
            if previous["usage"].get("task_id") != self.task["task_id"] or saved.get("purpose") != "observe" or saved.get("response") != previous:
                raise PermissionError("Observe 恢复需要当前任务已正式保存的原始响应")
        for attempt in range(2):
            observed = previous if attempt == 0 and previous is not None else await self.context.complete(self.task, self.state, self._messages, [], role="observe", purpose="observe", correction=OBSERVATION_CORRECTION if attempt else None, retry_reason="observation_format" if attempt else None, max_attempts=1 if attempt else None, allow_text=allow_text)
            if observed["tool_calls"]:
                raise ValueError("Observe 请求返回了未授权的工具调用；原始响应已保存，工具未执行")
            try:
                observation = self._observation(observed["content"])
            except ValueError as error:
                reference = self.session_relative_path + f"/requests/{observed['usage']['request_group_id']}.json"
                await self._run_sync(self.events.emit, "observation.failed", {"request_id": observed["request_id"], "response_ref": reference, "error_type": type(error).__name__, "error": str(error), "attempt": attempt + 1, "will_retry": attempt == 0})
                if attempt:
                    raise ValueError("Observe 格式纠正后仍未通过检查；原始响应与任务记录已保存，可通过 /resume 继续") from error
                continue
            await self._record_assistant(observed, "observe")
            self._update_observation(observation)
            self.state["next_goal"] = observation.get("next_goal", "")
            self.state["observation_decision"] = observation.get("decision")
            await self._save_session_async()
            return observation

    def _update_task_progress(self, results):
        self._update_repair_progress(results)
        if self.task.get("kind") not in {"conversation", "maintenance"}:
            return
        operations = {result["operation_id"] for result in results if result.get("operation_id")}
        records = [{**record, "arguments": record.get("arguments") if isinstance(record.get("arguments"), dict) else {}} for record in self.store.read_events(self.session_relative_path + "/tools.jsonl") if record.get("operation_id") in operations]
        previous = self.state.get("maintenance_progress") or {}
        mutations = [record for record in records if record["name"] in self.tools.tools and not self.tools.is_read_only(record["name"], record.get("arguments"))]
        pending_plans = [record for record in records if record["name"] == "install_dependencies" and record.get("arguments", {}).get("action") == "plan" and isinstance(record.get("data"), dict) and record["data"].get("requires_install") is True]
        installation_planned = any(record["name"] == "install_dependencies" and record.get("arguments", {}).get("action") == "plan" for record in records)
        if not mutations and not installation_planned and self.task.get("kind") != "maintenance":
            return
        self.task["kind"] = "maintenance"
        progress = {**previous, "phase": previous.get("phase", "checks_pending"), "tool_rounds": previous.get("tool_rounds", 0) + 1, "updated_at": now()}
        actions = list(previous.get("recent_actions", []))
        for record in records:
            actions.append({"name": record["name"], "status": record["status"], "evidence_ref": "tool:" + record["operation_id"], "error": record.get("error")})
            if record["name"] == "install_dependencies" and record.get("arguments", {}).get("action") == "plan":
                progress.update(install_pending=True, phase="checks_pending")
                progress.pop("environment_check", None)
            if record["name"] == "inspect_environment":
                progress["phase"] = "checks_pending"
                progress.pop("environment_check", None)
            if record in mutations:
                progress.update(phase="checks_pending", last_mutation_operation_id=record["operation_id"])
                progress.pop("environment_check", None)
            if record in pending_plans:
                progress.update(install_pending=True, pending_plan_id=record["data"].get("plan_id"))
            data = record.get("data")
            check = data.get("postcheck", data) if isinstance(data, dict) else {}
            reference = check.get("evidence_ref") if isinstance(check, dict) else None
            prefix = f"instances/{self.current_instance['instance_id']}/environment-checks/"
            if record["status"] == "success" and isinstance(reference, str) and reference.startswith(prefix):
                formal = self.store.read(reference, {})
                if formal.get("source_kind") == "observed_environment" and formal.get("task_id") == self.task["task_id"]:
                    progress["environment_check"] = {key: formal.get(key) for key in ("check_id", "instance_id", "task_id", "environment_id", "generation", "status", "passed", "checked_at", "evidence_ref", "dependency_fingerprint", "requirements_files", "file_versions")}
                    progress["phase"] = "checks_pending" if formal.get("passed") is True else "checks_failed"
                    if record["name"] == "install_dependencies" and record.get("arguments", {}).get("action") == "apply" and formal.get("passed") is True:
                        progress["install_pending"] = False
        progress["recent_actions"] = actions[-8:]
        self.state["maintenance_progress"] = progress
        if self._maintenance_is_complete():
            progress["phase"] = "completed"

    def _maintenance_is_complete(self):
        if self.task.get("kind") != "maintenance":
            return False
        progress = self.state.get("maintenance_progress") or {}
        reference = (progress.get("environment_check") or {}).get("evidence_ref")
        if not reference or progress.get("install_pending"):
            return False
        check = self.store.read(reference, {})
        instance = self.instances.get(self.current_instance["instance_id"])
        environment = instance.get("environment") or {}
        environment_id = environment.get("environment_id")
        registered = self.store.read(f"environments/{environment_id}/environment.json", {}) if environment_id else {}
        venv = self.store.path(f"environments/{environment_id}/venv") if environment_id else None
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python") if venv else None
        valid = (bool(environment_id) and environment.get("status") == "ready"
                 and environment.get("instance_id") == instance["instance_id"]
                 and all(registered.get(key) == environment.get(key) for key in ("environment_id", "instance_id", "generation", "status"))
                 and venv.is_dir() and python.is_file()
                 and Path(environment.get("venv_root", "")) == venv
                 and Path(environment.get("python_executable", "")) == python
                 and check.get("passed") is True and check.get("status") == "completed"
                 and check.get("source_kind") == "observed_environment"
                 and check.get("task_id") == self.task.get("task_id")
                 and check.get("instance_id") == instance["instance_id"]
                 and check.get("environment_id") == environment.get("environment_id")
                 and check.get("generation") == environment.get("generation"))
        # 真实检查对应的文件版本变化后必须重新检查，不能沿用历史通过。
        root = Path(instance["code_root"])
        for relative, version in (check.get("file_versions") or {}).items():
            path = (root / relative).resolve()
            valid = valid and path.is_relative_to(root.resolve()) and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == version
        if valid:
            self.current_instance = instance
        else:
            progress["phase"] = "checks_failed" if check.get("passed") is False else "checks_pending"
        return valid

    def _update_repair_progress(self, results):
        if self.task.get("kind") != "repair":
            return
        operations = {result["operation_id"] for result in results if result.get("operation_id")}
        records = [record for record in self.store.read_events(self.session_relative_path + "/tools.jsonl") if record.get("operation_id") in operations]
        previous = self.state.get("repair_progress") or {}
        changed = any(record["status"] == "success" and not self.tools.is_read_only(record["name"], record.get("arguments")) for record in records)
        files = {item["path"]: item for item in previous.get("files_read", [])}
        frozen = self.task.get("acceptance", {}).get("file_identities", {})
        actions = list(previous.get("recent_actions", []))
        for record in records:
            actions.append({"name": record["name"], "status": record["status"], "evidence_ref": "tool:" + record["operation_id"], "error": record.get("error")})
            data = record.get("data")
            if record["name"] == "read_file" and record["status"] == "success" and isinstance(data, dict):
                item = {key: data[key] for key in ("path", "file_version", "start_line", "end_line", "total_lines", "truncated")}
                item["evidence_ref"] = "tool:" + record["operation_id"]
                if item["path"] in frozen:
                    item["matches_frozen_identity"] = item["file_version"] == frozen[item["path"]]
                files.pop(item["path"], None)
                files[item["path"]] = item
        phase = previous.get("phase", "investigating")
        if any(record["name"] == "apply_patch" and record["status"] == "success" for record in records):
            phase = "verification_pending"
        if any(record["name"] == "run_example" and record["status"] == "success" for record in records):
            phase = "completed" if self._repair_is_complete() else "verification_failed"
        self.state["repair_progress"] = {"phase": phase, "tool_rounds": previous.get("tool_rounds", 0) + 1, "rounds_without_change": 0 if changed else previous.get("rounds_without_change", 0) + 1, "files_read": list(files.values())[-24:], "recent_actions": actions[-8:], "last_run_id": self.state.get("last_run_id"), "updated_at": now()}

    @staticmethod
    def _observation(content):
        value = content.strip()
        if value.startswith("```json") and value.endswith("```"):
            value = value[7:-3].strip()
        elif value.startswith("```") and value.endswith("```"):
            value = value[3:-3].strip()
        result = json.loads(value)
        if not isinstance(result, dict) or result.get("decision") not in {"continue", "completed", "blocked"}:
            raise ValueError("Observe 响应缺少有效 decision")
        if not isinstance(result.get("facts", []), list) or not isinstance(result.get("hypotheses", []), list):
            raise ValueError("Observe 的 facts 与 hypotheses 必须为列表")
        return result

    async def _record_assistant(self, response, role):
        raw = response["raw_message"]
        self._messages.append({**raw, "_ianvs_stage": role})
        if response["tool_calls"]:
            self.state["open_tool_round"] = {"assistant_index": len(self._messages) - 1, "request_id": response["request_id"], "tools_start_index": len(self.store.read_events(self.session_relative_path + "/tools.jsonl"))}
        await self._save_session_async()
        if response["content"] and role != "observe":
            self.events.emit("agent.message", {"role": role, "content": response["content"]})

    def _update_observation(self, observation):
        records = self.store.read_events(self.session_relative_path + "/tools.jsonl")
        references = {"tool:" + record["operation_id"] for record in records if record.get("operation_id")}
        references.update("tool:" + record["call_id"] for record in records)
        instance_reference = f"instances/{self.current_instance['instance_id']}/instance.json"
        if self.store.path(instance_reference).is_file():
            references.add(instance_reference)
        for record in records:
            if record.get("full_output_ref"):
                references.add(record["full_output_ref"])
            data = record.get("data")
            references.update(tool_evidence_references(data, self.current_instance["instance_id"]))
        facts, unverified = [], list(observation.get("hypotheses", []))
        for fact in observation.get("facts", []):
            reference = fact.get("evidence_ref") if isinstance(fact, dict) else None
            permitted = reference in references if isinstance(reference, str) else False
            if not permitted and isinstance(reference, str) and reference.startswith(f"instances/{self.current_instance['instance_id']}/runs/"):
                target = self.store.path(reference)
                runs_root = self.store.path(f"instances/{self.current_instance['instance_id']}/runs")
                permitted = target.is_relative_to(runs_root) and target.is_file()
            if not permitted and isinstance(reference, str) and reference.startswith(self.session_relative_path + "/tool-output/"):
                target = self.store.path(reference)
                permitted = target.is_file()
            if permitted:
                facts.append({**fact, "source": "observe", "evidence_status": "registered_reference"})
            else:
                unverified.append({"value": fact, "status": "unverified", "reason": "没有引用已登记证据"})
        self.state["facts"] = self._merge_observations(self.state.get("facts", []), facts)
        self.state["hypotheses"] = self._merge_observations(self.state.get("hypotheses", []), unverified)

    @staticmethod
    def _merge_observations(existing, incoming):
        merged = {}
        for item in [*existing, *incoming]:
            if isinstance(item, dict):
                value = observation_statement(item)
                if not value:
                    value = {key: entry for key, entry in item.items() if key not in {"source", "evidence_status"}}
                identity = (item.get("evidence_ref"), json.dumps(value, ensure_ascii=False, sort_keys=True))
            else:
                identity = (None, json.dumps(item, ensure_ascii=False, sort_keys=True))
            merged[identity] = item
        return list(merged.values())

    async def _finish(self, status, message):
        self._account_execution()
        self.state["status"] = status
        self.state["status_message"] = message
        instance = await self._run_sync(self.instances.get, self.current_instance["instance_id"]) if self.current_instance and status != "cancelled" else {}
        run_id = instance.get("last_run_id")
        run = await self._run_sync(self.store.read, f"instances/{instance['instance_id']}/runs/{run_id}/run.json", {}) if run_id else {}
        reference = self.session_relative_path + "/state.json" if self.session_relative_path else None
        result = format_task_result(self.task, self.state, status, message, run, reference)
        self.state["result"] = result
        await self._save_session_async()
        if status != "completed" or self.task.get("kind") != "conversation":
            self.events.emit("agent.message", {"role": "result", "content": result["content"]})
        event_type = {"completed": "agent.completed", "blocked": "agent.blocked", "failed": "agent.failed", "cancelled": "agent.cancelled"}[status]
        self.events.emit(event_type, {"message": message, "content": result["content"], "result": result, "verification": self.state.get("verification", {}), **({"error": message} if status == "failed" else {})})
        return result

    async def compact(self):
        self._require_instance()
        if self.busy:
            self._compact_requested = True
            self.events.emit("context.compaction_queued", {"message": "压缩将在当前完整工具交互结束后执行"})
            return {"status": "queued"}
        return await self._action(self._compact())

    async def _compact(self):
        self._compact_requested = False
        result = await self.context.compact(self.task, self.state, self._messages, self.tools.definitions(), manual=True)
        if result.get("status") != "skipped":
            await self._save_session_async()
        return result

    async def approve(self, approval_id, approved):
        self.tools.approve(approval_id, approved)

    def _account_execution(self):
        if self._execution_started is not None and self.task.get("task_id") == self._execution_task_id:
            current = time.monotonic()
            self.state["active_execution_seconds"] = self.state.get("active_execution_seconds", 0) + current - self._execution_started
            self._execution_started = current

    async def _action(self, awaitable, *, task_budget=False):
        if self.busy:
            awaitable.close()
            raise RuntimeError("当前任务正在执行")
        self.busy = True
        self._cancel_requested = False
        self._active_task = asyncio.current_task()
        self._execution_started = time.monotonic() if task_budget and self.task else None
        self._execution_task_id = self.task.get("task_id") if task_budget else None
        try:
            return await awaitable
        except asyncio.CancelledError:
            if self.state.get("open_tool_round"):
                self._repair_open_tool_pairs()
            await self._finish("cancelled", "任务已取消，已有证据已保存")
            raise
        except Exception as error:
            await self._record_action_failure(error)
            raise
        finally:
            self._account_execution()
            self._execution_started = self._execution_task_id = None
            try:
                await self._save_session_async()
            finally:
                self.busy = False
                self._active_task = None

    async def _record_action_failure(self, error):
        if self.state.get("open_tool_round"):
            self._repair_open_tool_pairs()
        self.state["error"] = str(error) or type(error).__name__
        if getattr(error, "request_failure", None):
            self.state["request_failure"] = {key: error.request_failure.get(key) for key in ("request_group_id", "purpose", "error_type", "failure_stage", "duration_seconds", "chunk_count", "last_chunk_seconds", "deadline_seconds", "idle_timeout_seconds", "attempt_ref")}
        await self._finish("failed", self.state["error"])

    async def cancel(self):
        self._cancel_requested = True
        self.tools.cancel_approvals()
        commands = getattr(self, "environment_commands", None)
        try:
            try:
                if commands is not None:
                    await commands.cancel()
            finally:
                await self.environment.cancel(self.current_instance)
        finally:
            try:
                await self.runner.cancel()
            finally:
                if self._active_task and self._active_task is not asyncio.current_task():
                    self._active_task.cancel()
                await self._save_session_async()

    def _session_snapshot(self):
        self._account_execution()
        self.session["updated_at"] = now()
        self.session["task_id"] = self.task.get("task_id")
        return copy.deepcopy({"schema_version": 1, "session": self.session, "task": self.task, "state": self.state, "history": self._messages})

    def _persist_session(self, relative, snapshot):
        with self._session_write_lock:
            self.store.write(relative + "/session_snapshot.json", snapshot)
            for filename, field in [("session.json", "session"), ("task.json", "task"), ("state.json", "state"), ("history.json", "history")]:
                self.store.write(relative + "/" + filename, snapshot[field])
            if snapshot["task"]:
                self.store.write(relative + f"/tasks/{snapshot['task']['task_id']}.json", {"task": snapshot["task"], "state": snapshot["state"]})

    async def _save_session_async(self):
        if not self.session_relative_path:
            return
        async with self._session_save_guard:
            await self._run_sync(self._persist_session, self.session_relative_path, self._session_snapshot())

    @staticmethod
    async def _run_sync(function, *arguments):
        pending = asyncio.create_task(asyncio.to_thread(function, *arguments))
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(pending)
                break
            except asyncio.CancelledError:
                if pending.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _save_session(self):
        if not self.session_relative_path:
            return
        self._persist_session(self.session_relative_path, self._session_snapshot())

    def history(self):
        return list(self._messages) if self.session else []

    def diff(self):
        if not self.current_instance:
            return ""
        root = self.store.path(f"instances/{self.current_instance['instance_id']}/changes")
        if not root.exists():
            return ""
        return "\n".join(item["diff"] for file in sorted(root.glob("*/change.json")) for item in self.store.read(file.relative_to(self.store.root).as_posix())["changes"])

    def usage_summary(self):
        return self.usage.summary() if self.usage else {"input_tokens": 0, "output_tokens": 0, "model_calls": 0, "tool_calls": 0, "context_fraction": None, "cost": None}

    def verification_report(self):
        if not self.current_instance:
            return {}
        return self.instances.get(self.current_instance["instance_id"]).get("verification") or {}

    async def close(self):
        await self.cancel()
        if self._active_task and self._active_task is not asyncio.current_task():
            await asyncio.gather(self._active_task, return_exceptions=True)
        await self._save_session_async()
        for provider in [*self._providers, self.provider]:
            if provider:
                await provider.close()
