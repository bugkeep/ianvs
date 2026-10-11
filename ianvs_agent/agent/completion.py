# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
from pathlib import Path

from ianvs_agent.runtime.instances import file_manifest, utc_now


class CompletionManager:
    def __init__(self, harness):
        self.harness = harness

    def contract(self):
        task = self.harness.task
        value = task.get("completion_contract")
        if value is None:
            return None
        saved = self.harness.store.read(value["evidence_ref"])
        if saved != value or value["task_id"] != task["task_id"] or value["original_goal"] != task["goal"]:
            raise ValueError("完成清单与正式任务记录不一致")
        return value

    def declare(self, requirements, answer_text=None):
        harness = self.harness
        task = harness.task
        if not task:
            raise ValueError("完成清单需要当前任务")
        if not requirements or any(item["kind"] not in {"answer", "environment", "benchmark", "code"} for item in requirements):
            raise ValueError("完成清单需要有效成果类型")
        if len({item["id"] for item in requirements}) != len(requirements):
            raise ValueError("完成清单的成果 ID 不能重复")
        current = self.contract()
        if current:
            existing = {item["id"]: item for item in current["requirements"]}
            proposed = {item["id"]: item for item in requirements}
            if any(proposed.get(identity) != item for identity, item in existing.items()):
                raise PermissionError("原完成清单不能删除或修改，新增用户要求只能追加成果")
            if proposed != existing and not harness.state.get("completion_revision_pending"):
                raise PermissionError("追加成果需要用户明确的新要求")
        if task.get("kind") in {"repair", "run"} and not any(item["kind"] == "benchmark" for item in requirements):
            raise ValueError("原修复或运行任务必须保留原 Example 的运行与独立验证要求")
        if any(item["kind"] == "answer" for item in requirements) and len(requirements) != 1:
            raise ValueError("answer 用于纯问答；有实际动作时声明对应的 environment、benchmark 或 code 成果")
        if requirements[0]["kind"] == "answer" and self._has_actions():
            raise ValueError("当前任务已有行动记录或待执行行动，不能改为纯问答完成")
        if answer_text is not None and (requirements[0]["kind"] != "answer" or not answer_text.strip()):
            raise ValueError("answer_text 仅用于纯问答的完整答复")
        records = harness.store.read_events(harness.session_relative_path + "/events.jsonl")
        source = next((event for event in reversed(records) if event.get("task_id") == task["task_id"] and event["event_type"] == "tool.started" and event["payload"].get("name") == "declare_completion_contract"), None)
        if source is None:
            raise ValueError("完成清单需要正式的声明工具事件")
        if current is None or current["requirements"] != requirements or harness.state.get("completion_revision_pending"):
            version = (current or {}).get("version", 0) + 1
            reference = harness.session_relative_path + f"/completion/{task['task_id']}/contract-{version}.json"
            value = {"schema_version": 1, "version": version, "task_id": task["task_id"], "instance_id": task["instance_id"], "original_goal": task["goal"], "goal_source": copy.deepcopy(task.get("goal_source")), "user_updates": copy.deepcopy(task.get("goal_updates", [])), "requirements": copy.deepcopy(requirements), "created_at": utc_now(), "source_event": {key: source[key] for key in ("event_id", "sequence", "timestamp")}, "source_operation_id": source["payload"]["operation_id"], "source_assistant": copy.deepcopy(harness.state.get("open_tool_round")), "previous_ref": (current or {}).get("evidence_ref"), "evidence_ref": reference}
            harness.store.write(reference, value)
            task["completion_contract"] = value
            current = value
        harness.state.pop("completion_revision_pending", None)
        if any(item["kind"] == "benchmark" for item in current["requirements"]):
            task["acceptance"].update(repair_task_id=task["task_id"], repair_started_at=task["created_at"])
        if answer_text is not None:
            harness.state["completion_answer"] = {"text": answer_text, "source_event_id": source["event_id"], "contract_version": current["version"]}
        return copy.deepcopy(current)

    def _task_tools(self):
        harness = self.harness
        events = harness.store.read_events(harness.session_relative_path + "/events.jsonl")
        operations = {event["payload"].get("operation_id") for event in events if event.get("task_id") == harness.task["task_id"] and event["event_type"] == "tool.started"}
        return [record for record in harness.store.read_events(harness.session_relative_path + "/tools.jsonl") if record.get("operation_id") in operations]

    def _has_actions(self):
        harness = self.harness
        records = self._task_tools()
        opened = harness.state.get("open_tool_round") or {}
        index = opened.get("assistant_index")
        if isinstance(index, int) and index < len(harness.history()):
            records += [{"name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])} for call in harness.history()[index].get("tool_calls", [])]
        return any(record["name"] != "declare_completion_contract" and (record["name"] == "install_dependencies" or not harness.tools.is_read_only(record["name"], record.get("arguments"))) for record in records)

    def ensure_action_allowed(self):
        contract = self.contract()
        if not contract or self.harness.state.get("completion_revision_pending"):
            raise PermissionError("执行行动前请调用 declare_completion_contract，完整声明当前用户目标的全部成果")
        if any(item["kind"] == "answer" for item in contract["requirements"]):
            raise PermissionError("纯问答完成清单不能授权修改或运行；新增行动需要用户新要求")

    def bind_function_check(self, requirement_id, record):
        contract = self.contract()
        requirement = next((item for item in (contract or {}).get("requirements", []) if item["id"] == requirement_id and item["kind"] == "code"), None)
        if requirement is None:
            raise ValueError("功能检查必须关联当前完成清单中的 code 成果")
        if record.get("task_id") != self.harness.task["task_id"]:
            raise ValueError("功能检查不属于当前任务")
        self.harness.state.setdefault("completion_function_checks", {}).setdefault(requirement_id, []).append(record["evidence_ref"])

    def validate_function_requirement(self, requirement_id):
        contract = self.contract()
        if not any(item["id"] == requirement_id and item["kind"] == "code" for item in (contract or {}).get("requirements", [])):
            raise ValueError("completion_requirement_id 必须是当前完成清单中的 code 成果")

    @staticmethod
    def is_install_plan(record):
        arguments = record.get("arguments") or {}
        data = record.get("data") or {}
        return bool(record.get("name") == "install_dependencies" and isinstance(arguments, dict) and arguments.get("action") == "plan"
                    and record.get("status") == "success" and isinstance(data, dict) and data.get("requires_install") is True
                    and data.get("plan_id") and data.get("evidence_ref"))

    def installation_pending(self):
        pending_plan_id = None
        for record in self._task_tools():
            if record["name"] == "install_dependencies":
                arguments = record.get("arguments") or {}
                if self.is_install_plan(record):
                    pending_plan_id = record["data"]["plan_id"]
                elif (isinstance(arguments, dict) and arguments.get("action") == "apply" and arguments.get("plan_id") == pending_plan_id
                      and record["status"] == "success" and (record.get("data") or {}).get("postcheck", {}).get("passed") is True):
                    pending_plan_id = None
        return pending_plan_id is not None

    def _environment(self, instance):
        harness = self.harness
        root = Path(instance["code_root"])
        environment = instance.get("environment") or {}
        environment_id = environment.get("environment_id")
        if self.installation_pending():
            return None
        if not environment_id or environment.get("status") != "ready" or instance.get("environment_changed"):
            return None
        registered = harness.store.read(f"environments/{environment_id}/environment.json", {})
        venv = harness.store.path(f"environments/{environment_id}/venv")
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not venv.is_dir() or not python.is_file() or Path(environment.get("venv_root", "")) != venv or Path(environment.get("python_executable", "")) != python:
            return None
        if any(registered.get(key) != environment.get(key) for key in ("environment_id", "instance_id", "generation", "status")):
            return None
        for path in sorted(harness.store.path(f"instances/{instance['instance_id']}/environment-checks").glob("*/check.json"), reverse=True):
            reference = path.relative_to(harness.store.root).as_posix()
            check = harness.store.read(reference)
            if check.get("task_id") != harness.task["task_id"] or check.get("instance_id") != instance["instance_id"] or check.get("environment_id") != environment_id or check.get("generation") != environment.get("generation"):
                continue
            if check.get("passed") is not True or check.get("status") != "completed" or check.get("source_kind") != "observed_environment":
                return None
            for relative, version in (check.get("file_versions") or {}).items():
                target = (root / relative).resolve()
                if not target.is_relative_to(root.resolve()) or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != version:
                    return None
            return reference
        return None

    def _benchmark(self, instance):
        harness = self.harness
        if not harness._repair_is_complete():
            return None
        run_id = instance.get("last_run_id")
        run = harness.runner.get_run(instance["instance_id"], run_id)
        verification = harness.store.read(f"instances/{instance['instance_id']}/runs/{run_id}/verification.json", {})
        binding = harness.task.get("original_goal") or {}
        if run.get("original_goal_ref") != binding.get("evidence_ref") or run.get("goal_digest") != binding.get("goal_digest") or verification.get("task_original_goal") != binding:
            return None
        if run.get("source_manifest") != file_manifest(Path(instance["code_root"])):
            return None
        return f"instances/{instance['instance_id']}/runs/{run_id}/verification.json"

    def _code(self, instance, requirement, manifest):
        harness = self.harness
        environment = instance.get("environment") or {}
        if instance.get("environment_changed") or environment.get("status") not in {"ready", "created"}:
            return None
        if harness.state.get("completion_function_check_pending", {}).get(requirement["id"]):
            return None
        modifications = []
        root = Path(instance["code_root"])
        initial_values = {}
        for path in sorted(harness.store.path(f"instances/{instance['instance_id']}/changes").glob("*/change.json")):
            record = harness.store.read(path.relative_to(harness.store.root).as_posix())
            if record.get("task_id") != harness.task["task_id"]:
                continue
            effective = []
            for item in record.get("changes", []):
                initial_values.setdefault(item["path"], item.get("before"))
                target = (root / item["path"]).resolve()
                if not target.is_relative_to(root.resolve()):
                    continue
                content = target.read_text(encoding="utf-8") if target.is_file() else None
                if item.get("diff") and content == item.get("after") and content != initial_values[item["path"]]:
                    effective.append(item["path"])
            if record.get("task_id") == harness.task["task_id"] and effective:
                modifications.append(path.relative_to(harness.store.root).as_posix())
        if not modifications:
            return None
        revision = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        references = harness.state.get("completion_function_checks", {}).get(requirement["id"], [])
        for reference in reversed(references):
            prefix = f"instances/{instance['instance_id']}/environment-commands/"
            if not reference.startswith(prefix):
                continue
            record = harness.store.read(reference, {})
            if record.get("task_id") == harness.task["task_id"] and record.get("status") != "completed":
                return None
            if (record.get("task_id") == harness.task["task_id"] and record.get("status") == "completed" and record.get("exit_code") == 0
                    and record.get("command_executed") is True and record.get("source_revision") == revision
                    and record.get("environment_id") == environment.get("environment_id") and record.get("generation") == environment.get("generation")
                    and record.get("python_executable") == environment.get("python_executable") and record.get("dependency_fingerprint") == environment.get("dependency_fingerprint")
                    and record.get("log_ref") and harness.store.path(record["log_ref"]).is_file()):
                return {"check_ref": reference, "patch_refs": sorted(modifications)}
        return None

    def evaluate(self):
        harness = self.harness
        contract = self.contract()
        scope = {"task_id": harness.task.get("task_id"), "instance_id": harness.task.get("instance_id"), "original_goal": copy.deepcopy(harness.task.get("original_goal"))}
        progress = {"task_id": harness.task.get("task_id"), "contract_version": (contract or {}).get("version"), "checked_at": utc_now(), "evidence_scope": scope, "requirements": [], "ready": False}
        if contract is None or harness.state.get("completion_revision_pending"):
            progress["missing"] = ["声明原始目标及用户补充要求的完整成果清单"]
        else:
            instance = harness.instances.get(harness.current_instance["instance_id"])
            manifest = None
            for item in contract["requirements"]:
                reference = None
                if item["kind"] == "environment":
                    reference = self._environment(instance)
                elif item["kind"] == "benchmark":
                    reference = self._benchmark(instance)
                elif item["kind"] == "code":
                    if manifest is None:
                        manifest = file_manifest(Path(instance["code_root"]))
                    reference = self._code(instance, item, manifest)
                elif item["kind"] == "answer":
                    answer = harness.state.get("completion_answer") or {}
                    if answer.get("text") and answer.get("contract_version") == contract["version"] and not self._has_actions():
                        reference = answer["source_event_id"]
                reasons = {
                    "environment": "需要当前 task_id 的 inspect_environment 通过记录或安装后的通过检查，对应当前实例环境、环境代次和声明文件版本；待执行安装计划及环境变更仍需处理。",
                    "benchmark": "需要通过本任务 run_example 执行原 Example，并取得归属当前 task_id、当前 original_goal 及源码版本的独立验证通过记录。历史任务的成功仅作背景。",
                    "code": "需要本任务当前仍有效的 Diff，以及关联本成果 completion_requirement_id、当前源码和环境版本的真实功能检查通过记录与日志；功能覆盖范围仍由 Observe 核对。",
                    "answer": "需要当前清单版本的 answer_text 完整答复，且本任务没有已执行或待执行的行动。",
                }
                progress["requirements"].append({**item, "status": "passed" if reference else "pending", "evidence": reference, "pending_reason": None if reference else reasons[item["kind"]]})
            progress["missing"] = [item["id"] + "：" + item["pending_reason"] for item in progress["requirements"] if item["status"] != "passed"]
            progress["ready"] = not progress["missing"]
        harness.state["completion_progress"] = progress
        return progress


def register_completion(harness, registry):
    async def declare_completion_contract(requirements, answer_text=None):
        value = await harness._run_sync(harness.completion.declare, requirements, answer_text)
        await harness._save_session_async()
        return value

    requirement = {"type": "object", "properties": {"id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,63}$"}, "kind": {"enum": ["answer", "environment", "benchmark", "code"]}, "description": {"type": "string", "minLength": 1, "description": "描述用户要求完成的事项；历史 Run ID、历史成功结果通过正式证据展示，不能作为事项描述或本任务完成声明。"}}, "required": ["id", "kind", "description"], "additionalProperties": False}
    registry.register("declare_completion_contract", "依据原始用户目标一次声明全部成果：answer纯问答、environment安装检查、benchmark原Example运行独立验证、code功能修改与对应真实功能检查。允许同时声明多项行动成果；已有成果不可删除修改。纯问答在本调用的answer_text给出完整答复，无需另发分类请求。", {"requirements": {"type": "array", "minItems": 1, "maxItems": 32, "items": requirement}, "answer_text": {"type": "string", "minLength": 1}}, ["requirements"], declare_completion_contract)
