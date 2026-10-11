# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import asyncio
import difflib
import json
import os
import shutil
import sys
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from ianvs_agent.agent.harness import AgentHarness, OBSERVATION_CORRECTION
from ianvs_agent.config import load_settings
from ianvs_agent.message_types import format_task_result, tool_evidence_references


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "examples/llm_simple_qa/benchmarkingjob.yaml"


class HarnessTaskTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        artifacts = Path(os.environ.get("IANVS_HARNESS_TEST_ROOT", PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/harness-tasks"))
        self.root = artifacts / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        project = self.root / "source"
        shutil.copytree(PROJECT_ROOT / "examples/llm_simple_qa", project / "examples/llm_simple_qa")
        settings = replace(load_settings(PROJECT_ROOT, load_credentials=False), project_root=project, state_root=self.root / "state", python_executable=sys.executable, approval_policy="auto_scoped")
        self.harness = AgentHarness(settings)
        await self.harness.create_instance(BENCHMARK)

    async def asyncTearDown(self):
        await self.harness.close()

    async def test_original_goal_is_reloaded_for_context_and_readable_after_restart(self):
        self.harness._ensure_task("修复当前 Example，保留原始目标", kind="repair")
        acceptance = copy.deepcopy(self.harness.task["acceptance"])
        binding = copy.deepcopy(self.harness.task["original_goal"])
        instance = self.harness.current_instance
        goal = self.harness.store.read(binding["evidence_ref"])
        self.assertEqual(goal["status"], "complete", goal["unresolved"])
        path = Path(instance["code_root"]) / goal["documents"]["testenv"]["path"]
        path.write_text("testenv:\n  metrics: []\n", encoding="utf-8")
        self.harness._bind_original_goal()
        self.assertEqual(self.harness.task["original_goal"], binding)
        self.assertEqual(self.harness.task["acceptance"], acceptance)
        await self.harness._save_session_async()
        identity = instance["instance_id"]
        settings = self.harness.settings
        await self.harness.close()
        self.harness = AgentHarness(settings)
        self.harness.select_instance(identity)
        self.assertEqual(self.harness.task["original_goal"], binding)
        self.assertEqual(self.harness.task["acceptance"], acceptance)
        messages = self.harness.context._pinned(self.harness.task, self.harness.state, "observe")
        pinned = next(message for message in messages if message["content"].startswith("程序读取的原目标与创建来源"))
        source = json.loads(pinned["content"].split("\n", 1)[1])
        self.assertEqual(source["requirements"], goal["requirements"])
        self.assertEqual(source["evidence_ref"], binding["evidence_ref"])
        result = await self.harness.tools.execute({"id": "read-original-goal", "name": "read_evidence", "arguments": {"reference": binding["evidence_ref"], "end_line": 2000}})
        self.assertEqual(result["status"], "success", result)
        full = self.harness.store.read(result["full_output_ref"]) if result["truncated"] else result["data"]
        self.assertIn(goal["goal_id"], full["content"])
        self.harness._update_observation({"facts": [{"value": "原目标来自创建来源记录", "evidence_ref": binding["evidence_ref"]}], "hypotheses": []})
        self.assertEqual(self.harness.state["facts"][-1]["evidence_ref"], binding["evidence_ref"])
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_runner_rejects_different_task_goal_before_execution(self):
        self.harness._ensure_task("运行当前 Example", kind="run")
        binding = {**self.harness.task["original_goal"], "goal_id": "different-goal"}
        with self.assertRaisesRegex(ValueError, "原目标版本"):
            await self.harness.runner.run(self.harness.current_instance, original_goal=binding)
        instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        run = self.harness.runner.get_run(instance["instance_id"], instance["last_run_id"])
        self.assertEqual(run["status"], "failed")
        self.assertIsNone(run["pid"])
        self.assertIsNone(run["source_manifest"])

    async def test_failed_run_starts_repair_with_user_goal_and_frozen_acceptance(self):
        self.harness._ensure_task("运行当前 Example", kind="run")
        self.harness.task["acceptance"]["metric_thresholds"] = {"acc": {"min": 1.0}}
        with self.assertRaisesRegex(RuntimeError, "环境尚未准备"):
            await self.harness.run_current()
        old_task, old_state = copy.deepcopy((self.harness.task, self.harness.state))
        instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        old_run = self.harness.run_summary(self.harness.runner.get_run(instance["instance_id"], instance["last_run_id"]))
        goal = "修复当前 Example 的运行错误，保持原始数据、模型和指标，提交补丁后重新验证"
        self.harness._ensure_task(goal, kind="repair")
        self.assertNotEqual(self.harness.task["task_id"], old_task["task_id"])
        self.assertEqual(self.harness.task["kind"], "repair")
        self.assertEqual(self.harness.task["goal"], goal)
        self.assertEqual(self.harness.task["acceptance"], old_task["acceptance"])
        self.assertEqual(self.harness.state["last_run"], old_run)
        self.assertEqual(self.harness.state["failure"]["run_id"], old_run["run_id"])
        archived = self.harness.store.read(self.harness.session_relative_path + f"/tasks/{old_task['task_id']}.json")
        self.assertEqual(archived, {"task": old_task, "state": old_state})
        prepared = await self.harness.context.prepare(self.harness.task, self.harness.state, self.harness.history(), self.harness.tools.definitions())
        self.assertIn(goal, prepared["messages"][1]["content"])
        self.assertIn("环境尚未准备", prepared["messages"][2]["content"])
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)

    async def test_resuming_repair_preserves_original_task_acceptance_and_usage(self):
        goal = "修复当前 Example 的 Python 实现，运行原任务并接受独立验证"
        self.harness._ensure_task(goal, kind="repair")
        self.harness.task["acceptance"]["metric_thresholds"] = {"acc": {"min": 1.0}}
        result = await self.harness.tools.execute({"id": "read-benchmark", "name": "read_file", "arguments": {"path": BENCHMARK}})
        self.assertEqual(result["status"], "success")
        await self.harness._finish("blocked", "当前尚未准备实例环境，已保存文件证据")
        original = copy.deepcopy(self.harness.task)
        usage = self.harness.usage_summary()
        self.harness._ensure_task("继续当前修复", kind="repair")
        self.assertEqual(self.harness.task, original)
        self.assertEqual(self.harness.usage_summary(), usage)
        self.assertEqual(self.harness.usage_summary()["tool_calls"], 1)
        with self.assertRaisesRegex(RuntimeError, "环境尚未准备"):
            await self.harness.run_current()
        self.assertEqual(self.harness.task, original)
        self.assertEqual(self.harness.task["goal"], goal)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    @unittest.skipUnless(os.environ.get("IANVS_HARNESS_COMPLETED_RUN_CAPTURE"), "需要已归档的真实 completed Run 任务与状态")
    async def test_completed_real_run_starts_repair_with_frozen_acceptance(self):
        capture = Path(os.environ["IANVS_HARNESS_COMPLETED_RUN_CAPTURE"]).resolve()
        original = capture.read_bytes()
        archived = json.loads(original)
        task, state = copy.deepcopy((archived["task"], archived["state"]))
        self.assertEqual(task["kind"], "run")
        self.assertEqual(state["status"], "completed")
        source = capture.parents[5]
        base = Path("instances") / task["instance_id"]
        self.assertEqual(capture.relative_to(source).parts[:3], ("instances", task["instance_id"], "sessions"))
        run_id = state["last_run_id"]
        run_root = source / base / "runs" / run_id
        run_bytes = (run_root / "run.json").read_bytes()
        verification_bytes = (run_root / "verification.json").read_bytes()
        run = json.loads(run_bytes)
        verification = json.loads(verification_bytes)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["exit_code"], 0)
        self.assertEqual(run["run_id"], run_id)
        self.assertEqual(run["instance_id"], task["instance_id"])
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["run_id"], run_id)
        # completed 状态与 Run 完整保留；测试合同的附加约束来自该健康 Run 的正式证据。
        metrics = next(iter(run["metrics"].values()))
        task["acceptance"]["metric_thresholds"] = {name: {"min": value} for name, value in metrics.items() if isinstance(value, (int, float))}
        task["acceptance"]["file_identities"] = {task["benchmark_path"]: run["source_manifest"][task["benchmark_path"]]}
        self.assertTrue(task["acceptance"]["metric_thresholds"])
        expected = copy.deepcopy(task["acceptance"])
        await self.harness.close()
        isolated = self.root / "completed-state"
        destination = isolated / base
        destination.mkdir(parents=True)
        shutil.copy2(source / "counters.json", isolated / "counters.json")
        shutil.copy2(source / base / "instance.json", destination / "instance.json")
        shutil.copytree(source / base / "runs", destination / "runs")
        environment_id = json.loads((destination / "instance.json").read_text(encoding="utf-8")).get("environment", {}).get("environment_id")
        if environment_id:
            environment_destination = isolated / "environments" / environment_id
            environment_destination.mkdir(parents=True)
            shutil.copy2(source / "environments" / environment_id / "environment.json", environment_destination / "environment.json")
        settings = replace(self.harness.settings, project_root=PROJECT_ROOT, state_root=isolated)
        self.harness = AgentHarness(settings)
        await self.harness.restore_instance(task["instance_id"])
        self.harness.task, self.harness.state = task, state
        self.harness._save_session()
        old_task, old_state = copy.deepcopy((task, state))
        goal = "修复当前 Example 的新故障，保留已冻结的原始验收"
        self.harness._ensure_task(goal, kind="repair")
        self.assertNotEqual(self.harness.task["task_id"], old_task["task_id"])
        self.assertEqual(self.harness.task["kind"], "repair")
        self.assertEqual(self.harness.task["goal"], goal)
        self.assertEqual(self.harness.task["acceptance"], expected)
        saved = self.harness.store.read(self.harness.session_relative_path + f"/tasks/{old_task['task_id']}.json")
        self.assertEqual(saved, {"task": old_task, "state": old_state})
        self.assertFalse(self.harness._repair_is_complete())
        self.assertEqual(capture.read_bytes(), original)
        self.assertEqual((run_root / "run.json").read_bytes(), run_bytes)
        self.assertEqual((run_root / "verification.json").read_bytes(), verification_bytes)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)
        self.harness.store.write("completed-run-test-source.json", {"capture": str(capture), "run_ref": str(run_root / "run.json"), "verification_ref": str(run_root / "verification.json"), "captured_status": archived["state"]["status"], "source_acceptance_keys": sorted(archived["task"]["acceptance"]), "test_contract_added_from_real_run": ["metric_thresholds", "file_identities"], "model_calls": 0})

    async def test_repair_progress_records_actual_file_identity_and_applied_patch(self):
        self.harness._ensure_task("核查实例 README 并补充维护记录", kind="repair")
        path = "examples/llm_simple_qa/README.md"
        read = await self.harness.tools.execute({"id": "read-readme", "name": "read_file", "arguments": {"path": path, "end_line": 2000}})
        self.assertEqual(read["status"], "success")
        version = read["data"]["file_version"]
        self.harness.task["acceptance"]["file_identities"] = {path: version}
        self.harness._update_repair_progress([read])
        progress = self.harness.state["repair_progress"]
        self.assertEqual(progress["phase"], "investigating")
        self.assertTrue(progress["files_read"][0]["matches_frozen_identity"])
        self.assertFalse(progress["files_read"][0]["truncated"])
        self.assertEqual(progress["files_read"][0]["evidence_ref"], "tool:" + read["operation_id"])
        target = Path(self.harness.current_instance["code_root"]) / path
        before = target.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in before else "\n"
        after = before + newline + "本实例维护时保留独立验证记录。" + newline
        patch = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="a/" + path, tofile="b/" + path))
        applied = await self.harness.tools.execute({"id": "patch-readme", "name": "apply_patch", "arguments": {"patch": patch}})
        self.assertEqual(applied["status"], "success")
        self.harness._update_repair_progress([applied])
        self.assertEqual(self.harness.state["repair_progress"]["phase"], "verification_pending")
        self.assertEqual(self.harness.state["repair_progress"]["rounds_without_change"], 0)
        self.assertFalse(self.harness.state["verification"]["passed"])
        self.assertEqual(target.read_bytes().decode("utf-8"), after)
        self.assertEqual((self.harness.settings.project_root / path).read_bytes().decode("utf-8"), before)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_install_goal_remains_authorized_after_actual_read(self):
        goal = "安装当前 Example 声明的依赖，补齐缺失文件并检查结果"
        self.harness._ensure_task(goal)
        result = await self.harness.tools.execute({"id": "inspect-install-goal", "name": "read_file", "arguments": {"path": BENCHMARK}})
        self.harness._update_task_progress([result])
        self.assertEqual(self.harness.task["kind"], "conversation")
        self.assertFalse(self.harness._requires_observation([{"name": "read_file"}]))
        prepared = await self.harness.context.prepare(self.harness.task, self.harness.state, self.harness.history(), self.harness.tools.definitions())
        self.assertIn(goal, prepared["messages"][-1]["content"])
        self.assertIn("安装并检查结果", prepared["messages"][-1]["content"])
        self.assertNotIn("不要自动检查环境", prepared["messages"][-1]["content"])
        self.assertIn("只读诊断和向用户说明发现无需修改审批", self.harness.context._system("observe"))
        self.assertIn("每项使用 value", self.harness.context._system("observe"))
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_actual_patch_promotes_maintenance_without_claiming_repair(self):
        self.harness._ensure_task("在独立实例 README 补充准备说明")
        path = "examples/llm_simple_qa/README.md"
        read = await self.harness.tools.execute({"id": "maintenance-read", "name": "read_file", "arguments": {"path": path, "end_line": 2000}})
        target = Path(self.harness.current_instance["code_root"]) / path
        before = target.read_bytes().decode("utf-8")
        after = before + "\n本实例安装后需检查声明依赖。\n"
        patch = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="a/" + path, tofile="b/" + path))
        applied = await self.harness.tools.execute({"id": "maintenance-patch", "name": "apply_patch", "arguments": {"patch": patch}})
        self.harness._update_task_progress([applied])
        self.assertEqual(applied["status"], "success")
        self.assertEqual(self.harness.task["kind"], "maintenance")
        self.assertEqual(self.harness.state["maintenance_progress"]["phase"], "checks_pending")
        self.assertFalse(self.harness._maintenance_is_complete())
        self.assertFalse(self.harness._repair_is_complete())
        result = format_task_result(self.harness.task, self.harness.state, "blocked", "尚未检查当前环境")
        self.assertIn("尚未取得修改后的完整检查结果", result["content"])
        self.assertIn("当前不能宣称修复完成", result["content"])
        self.assertEqual((self.harness.settings.project_root / path).read_bytes().decode("utf-8"), before)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_actual_tools_continue_after_statistics_threshold_and_restore(self):
        self.harness._ensure_task("准备当前实例的依赖并验证")
        for number in range(2):
            result = await self.harness.tools.execute({"id": "resume-budget-" + str(number), "name": "read_file", "arguments": {"path": BENCHMARK}})
            self.assertEqual(result["status"], "success")
        await self.harness._finish("failed", "当前安装步骤尚未完成，保留读取证据")
        original_task, original_history = copy.deepcopy((self.harness.task, self.harness.history()))
        usage = self.harness.usage_summary()
        self.harness.settings.max_tool_calls = 1
        self.harness.settings.max_input_tokens = 500000
        result = await self.harness._action(self.harness.tools.execute({"id": "read-after-statistics-threshold", "name": "read_file", "arguments": {"path": BENCHMARK}}), task_budget=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.harness.task, original_task)
        self.assertEqual(self.harness.history(), original_history)
        self.assertEqual(self.harness.usage_summary()["task_tool_calls"], usage["task_tool_calls"] + 1)
        await self.harness.close()
        self.harness = AgentHarness(self.harness.settings)
        result = await self.harness._action(self.harness.tools.execute({"id": "read-after-statistics-restore", "name": "read_file", "arguments": {"path": BENCHMARK}}), task_budget=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.harness.task, original_task)
        self.assertEqual(self.harness.usage_summary()["task_tool_calls"], usage["task_tool_calls"] + 2)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        self.assertEqual(self.harness.settings.max_input_tokens, 500000)
        self.assertIsNone(self.harness.provider.client)

    async def test_active_time_survives_restore_without_counting_idle_time(self):
        self.harness._ensure_task("核查当前实例配置")
        await self.harness._action(self.harness.tools.execute({"id": "actual-timed-read", "name": "read_file", "arguments": {"path": BENCHMARK}}), task_budget=True)
        elapsed = self.harness.state["active_execution_seconds"]
        self.assertGreater(elapsed, 0)
        original_task = copy.deepcopy(self.harness.task)
        await self.harness.close()
        await asyncio.sleep(0.02)
        self.harness = AgentHarness(self.harness.settings)
        self.assertEqual(self.harness.task, original_task)
        self.assertEqual(self.harness.state["active_execution_seconds"], elapsed)
        self.assertEqual(self.harness.usage_summary()["tool_calls"], 1)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)

    async def test_actual_tool_continues_after_recorded_active_time_threshold(self):
        self.harness._ensure_task("继续检查当前实例的依赖")
        await self.harness._action(self.harness.tools.execute({"id": "read-before-time-budget", "name": "read_file", "arguments": {"path": BENCHMARK}}), task_budget=True)
        await self.harness._finish("failed", "当前步骤需要继续")
        original = copy.deepcopy(self.harness.task)
        consumed = self.harness.state["active_execution_seconds"]
        self.harness.settings.max_task_seconds = consumed / 2
        result = await self.harness._action(self.harness.tools.execute({"id": "read-after-time-threshold", "name": "read_file", "arguments": {"path": BENCHMARK}}), task_budget=True)
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.harness.task, original)
        self.assertGreater(self.harness.state["active_execution_seconds"], consumed)
        self.assertEqual(self.harness.settings.max_input_tokens, 500000)
        self.assertEqual(self.harness.usage_summary()["tool_calls"], 2)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)

    async def test_actual_runs_continue_after_attempt_statistics_threshold(self):
        self.harness.settings.max_repair_attempts = 1
        run_ids = []
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "环境尚未准备"):
                await self.harness.run_current()
            run_id = self.harness.instances.get(self.harness.current_instance["instance_id"])["last_run_id"]
            run = self.harness.runner.get_run(self.harness.current_instance["instance_id"], run_id)
            self.assertEqual(run["status"], "failed")
            self.assertEqual(self.harness.state["status"], "failed")
            run_ids.append(run_id)
        self.assertEqual(len(set(run_ids)), 2)
        self.assertEqual(self.harness.state["repair_attempts"], 2)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)

    @unittest.skipUnless(os.environ.get("IANVS_HARNESS_VERIFIED_RESUME_CAPTURE") and os.environ.get("IANVS_HARNESS_RESUME_ARCHIVE"), "需要真实复验通过的状态目录与原阻塞会话归档")
    async def test_verified_repair_resume_finishes_without_another_model_request(self):
        source = Path(os.environ["IANVS_HARNESS_VERIFIED_RESUME_CAPTURE"])
        archive = Path(os.environ["IANVS_HARNESS_RESUME_ARCHIVE"])
        archived = {name: (archive / name).read_bytes() for name in ("session_snapshot.json", "task.json", "state.json", "history.json", "events.jsonl", "usage.jsonl")}
        snapshot = json.loads(archived["session_snapshot.json"])
        task, state = snapshot["task"], snapshot["state"]
        self.assertEqual(task["kind"], "repair")
        self.assertIn(state["status"], {"blocked", "failed", "interrupted"})
        instance_id, session_id, run_id = task["instance_id"], snapshot["session"]["session_id"], state["last_run_id"]
        base = Path("instances") / instance_id
        relative = base / "sessions" / session_id
        instance = json.loads((source / base / "instance.json").read_bytes())
        verification = instance["verification"]
        self.assertTrue(verification["passed"])
        self.assertTrue(verification["repair_completed"])
        self.assertEqual(verification["acceptance"]["repair_task_id"], task["task_id"])
        self.assertEqual(instance["last_run_id"], run_id)
        await self.harness.close()
        isolated = self.root / "verified-resume-state"
        shutil.copytree(archive, isolated / relative)
        records = [Path("counters.json"), base / "instance.json", base / "latest_session.json", base / "runs" / run_id / "run.json", base / "runs" / run_id / "verification.json", Path("environments") / instance["environment"]["environment_id"] / "environment.json"]
        for record in records:
            destination = isolated / record
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / record, destination)
        self.assertTrue(all((archive / name).read_bytes() == content for name, content in archived.items()))
        settings = replace(self.harness.settings, project_root=PROJECT_ROOT, state_root=isolated)
        self.harness = AgentHarness(settings)
        await self.harness.restore_instance(instance_id)
        self.assertEqual(self.harness.task, task)
        self.assertEqual(self.harness.state["status"], state["status"])
        self.assertEqual(self.harness.state["active_execution_seconds"], state["active_execution_seconds"])
        run = self.harness.runner.get_run(instance_id, run_id)
        verification = await self.harness.verifier.verify(self.harness.current_instance, run, copy.deepcopy(task["acceptance"]))
        self.assertTrue(verification["passed"], verification["reasons"])
        self.assertTrue(verification["repair_completed"])
        self.assertFalse(verification["stale"])
        facts = copy.deepcopy(self.harness.state.get("facts"))
        usage = self.harness.usage_summary()
        self.harness.settings.max_model_calls = 1
        self.harness.settings.max_tool_calls = 1
        self.harness.settings.max_input_tokens = 1
        self.harness.settings.max_output_tokens = 1
        self.harness.settings.max_repair_attempts = 1
        self.harness.settings.max_task_seconds = max(0.000001, self.harness.state["active_execution_seconds"] / 2)
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        llm_started = sum(event["event_type"] == "llm.started" for event in events)
        result = await self.harness.resume_current()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.harness.task, task)
        self.assertEqual(self.harness.state.get("facts"), facts)
        self.assertEqual(self.harness.state["resume_count"], state.get("resume_count", 0) + 1)
        self.assertTrue(self.harness.state["verification"]["repair_completed"])
        self.assertEqual(self.harness.state["verification"], verification)
        self.assertEqual(self.harness.usage_summary(), usage)
        self.assertEqual((isolated / relative / "usage.jsonl").read_bytes(), archived["usage.jsonl"])
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        self.assertEqual(sum(event["event_type"] == "llm.started" for event in events), llm_started)
        self.assertEqual(events[-1]["event_type"], "agent.completed")
        self.assertEqual(self.harness.settings.max_input_tokens, 1)
        self.assertIsNone(self.harness.provider.client)
        self.assertTrue(all((archive / name).read_bytes() == content for name, content in archived.items()))
        self.harness.store.write("verified-resume-test-source.json", {"source_state": str(source), "source_archive": str(archive), "task_id": task["task_id"], "run_id": run_id, "original_status": state["status"], "final_status": result["status"], "verified_at": verification.get("verified_at"), "original_active_execution_seconds": state["active_execution_seconds"], "final_active_execution_seconds": self.harness.state["active_execution_seconds"], "model_calls_added": 0, "usage_preserved": True, "max_input_tokens": settings.max_input_tokens})

    async def test_failed_install_plan_requires_real_completion_evidence(self):
        self.harness._ensure_task("安装当前 Example 的声明依赖并检查")
        goal = self.harness.task["goal"]
        result = await self.harness.tools.execute({"id": "missing-install-declaration", "name": "install_dependencies", "arguments": {"action": "plan", "requirements_files": ["missing-dependency-declaration.txt"]}})
        self.assertEqual(result["status"], "error")
        self.assertIn("missing-dependency-declaration", result["error"])
        self.harness._update_task_progress([result])
        self.assertEqual(self.harness.task["kind"], "maintenance")
        self.assertEqual(self.harness.task["goal"], goal)
        self.assertTrue(self.harness.state["maintenance_progress"]["install_pending"])
        self.assertFalse(self.harness._maintenance_is_complete())
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_invalid_argument_type_keeps_schema_failure_and_task_progress(self):
        self.harness._ensure_task("安装当前 Example 的依赖并检查")
        goal = self.harness.task["goal"]
        call = {"id": "invalid-install-argument-type", "name": "install_dependencies", "arguments": []}
        result = await self.harness.tools.execute(call)
        self.assertEqual(result["status"], "error")
        self.assertIn("object", result["error"])
        self.assertTrue(self.harness._requires_observation([call]))
        self.harness._update_task_progress([result])
        self.assertEqual(self.harness.task["goal"], goal)
        self.assertEqual(self.harness.task["kind"], "maintenance")
        self.assertFalse(self.harness._maintenance_is_complete())
        records = self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")
        record = next(item for item in records if item["operation_id"] == result["operation_id"])
        self.assertEqual(record["arguments"], [])
        self.assertEqual(record["status"], "error")
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bwrap"), "需要真实 Linux 隔离进程与独立 Python 环境")
    async def test_maintenance_completion_requires_current_real_environment_check(self):
        wheel_root = os.environ.get("IANVS_ENVIRONMENT_WHEEL_ROOT")
        if not wheel_root:
            self.skipTest("需要真实 packaging26.3 wheel 缓存 IANVS_ENVIRONMENT_WHEEL_ROOT")
        self.harness._ensure_task("创建并检查独立 Python 环境和标准库导入，不运行评测")
        path = "environment-check-requirements.txt"
        read = await self.harness.tools.execute({"id": "read-real-check-file", "name": "read_file", "arguments": {"path": path}})
        self.assertEqual(read["status"], "success")
        self.assertFalse(read["data"]["exists"])
        self.assertEqual(read["data"]["file_version"], "absent")
        content = "# 此次检查仅使用 Python 标准库。\n"
        patch = "".join(difflib.unified_diff([], content.splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + path))
        applied = await self.harness.tools.execute({"id": "add-real-check-file", "name": "apply_patch", "arguments": {"patch": patch}})
        self.assertEqual(applied["status"], "success", applied.get("error"))
        self.harness._update_task_progress([applied])
        self.assertEqual(self.harness.task["kind"], "maintenance")
        self.assertFalse(self.harness._maintenance_is_complete())
        await self.harness.environment.prepare(self.harness.current_instance, install=False)
        self.harness.current_instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        checked = await self.harness.tools.execute({"id": "real-environment-check", "name": "inspect_environment", "arguments": {"requirements_files": [path], "modules": ["json"]}})
        self.assertEqual(checked["status"], "success", checked.get("error"))
        self.harness._update_task_progress([checked])
        self.assertTrue(checked["data"]["passed"])
        self.assertFalse(self.harness._maintenance_is_complete())
        root = Path(self.harness.current_instance["code_root"])
        wheels = root / "wheels"
        wheels.mkdir()
        shutil.copy2(Path(wheel_root) / "packaging-26.3-py3-none-any.whl", wheels)
        declaration = "environment-install-requirements.txt"
        declared = "--no-index\n--find-links ./wheels\npackaging==26.3\n"
        (root / declaration).write_text(declared, encoding="utf-8")
        plan = await self.harness.tools.execute({"id": "real-offline-plan", "name": "install_dependencies", "arguments": {"action": "plan", "requirements_files": [declaration], "modules": ["packaging"]}})
        self.assertEqual(plan["status"], "success", plan.get("error"))
        self.harness._update_task_progress([plan])
        installed = await self.harness.tools.execute({"id": "real-offline-install", "name": "install_dependencies", "arguments": {"action": "apply", "plan_id": plan["data"]["plan_id"]}})
        self.assertEqual(installed["status"], "success", installed.get("error"))
        self.harness._update_task_progress([installed])
        self.assertTrue(self.harness._maintenance_is_complete())
        checked = await self.harness.tools.execute({"id": "ready-environment-check", "name": "inspect_environment", "arguments": {"requirements_files": [path], "modules": ["json"]}})
        self.assertEqual(checked["status"], "success", checked.get("error"))
        self.harness._update_task_progress([checked])
        self.assertTrue(self.harness._maintenance_is_complete())
        progress = self.harness.state["maintenance_progress"]
        self.assertEqual(progress["phase"], "completed")
        formal = self.harness.store.read(progress["environment_check"]["evidence_ref"])
        self.assertTrue(formal["passed"])
        self.assertTrue(formal["pip_check"]["passed"])
        self.assertTrue(formal["imports"][0]["passed"])
        self.assertEqual(formal["task_id"], self.harness.task["task_id"])
        records = self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")
        actual = next(record for record in records if record["operation_id"] == checked["operation_id"])
        self.assertIn(formal["evidence_ref"], tool_evidence_references(actual["data"], self.harness.current_instance["instance_id"]))
        self.assertNotIn(formal["evidence_ref"], tool_evidence_references(actual["data"], "instance-other"))
        environment = self.harness.instances.get(self.harness.current_instance["instance_id"])["environment"]
        self.assertIn(formal["evidence_ref"], tool_evidence_references(environment, self.harness.current_instance["instance_id"]))
        self.harness._update_observation({"facts": [{"value": "本次真实 pip check 与 json 导入检查通过", "evidence_ref": formal["evidence_ref"]}], "hypotheses": []})
        self.assertEqual(self.harness.state["facts"][-1]["evidence_ref"], formal["evidence_ref"])
        self.assertEqual(self.harness.state["facts"][-1]["evidence_status"], "registered_reference")
        prepared = await self.harness.context.prepare(self.harness.task, self.harness.state, self.harness.history(), [], role="observe")
        self.assertIn(formal["evidence_ref"], prepared["messages"][-1]["content"])
        self.assertFalse(self.harness._repair_is_complete())
        result = format_task_result(self.harness.task, self.harness.state, "completed", "当前环境检查已完成")
        self.assertIn("环境检查：已通过", result["content"])
        self.assertIn("当前不能宣称修复完成", result["content"])
        failed = await self.harness.tools.execute({"id": "failed-environment-check", "name": "inspect_environment", "arguments": {"requirements_files": ["../outside-check-declaration.txt"]}})
        self.assertEqual(failed["status"], "error")
        self.harness._update_task_progress([failed])
        self.assertFalse(self.harness._maintenance_is_complete())
        self.assertNotIn("environment_check", self.harness.state["maintenance_progress"])
        self.assertEqual(self.harness.state["maintenance_progress"]["phase"], "checks_pending")
        checked = await self.harness.tools.execute({"id": "real-environment-recheck", "name": "inspect_environment", "arguments": {"requirements_files": [path], "modules": ["json"]}})
        self.assertEqual(checked["status"], "success", checked.get("error"))
        self.harness._update_task_progress([checked])
        self.assertTrue(self.harness._maintenance_is_complete())
        progress = self.harness.state["maintenance_progress"]
        target = Path(self.harness.current_instance["code_root"]) / path
        target.write_text(content + "# 声明文件已经变化，需要重新检查。\n", encoding="utf-8")
        self.assertFalse(self.harness._maintenance_is_complete())
        self.assertEqual(progress["phase"], "checks_pending")
        (root / declaration).write_text(declared + "# 环境声明已变化。\n", encoding="utf-8")
        failed = await self.harness.tools.execute({"id": "actual-missing-module-check", "name": "inspect_environment", "arguments": {"requirements_files": [path], "modules": ["ianvs_harness_module_that_does_not_exist"]}})
        self.assertEqual(failed["status"], "success", failed.get("error"))
        self.assertFalse(failed["data"]["passed"])
        self.harness._update_task_progress([failed])
        self.assertEqual(self.harness.current_instance["environment"]["status"], "modified")
        checked = await self.harness.tools.execute({"id": "modified-environment-check", "name": "inspect_environment", "arguments": {"requirements_files": [path], "modules": ["json"]}})
        self.assertEqual(checked["status"], "success", checked.get("error"))
        self.assertTrue(checked["data"]["passed"])
        self.harness._update_task_progress([checked])
        self.assertEqual(self.harness.current_instance["environment"]["status"], "modified")
        self.assertFalse(self.harness._maintenance_is_complete())
        plan = await self.harness.tools.execute({"id": "modified-offline-plan", "name": "install_dependencies", "arguments": {"action": "plan", "requirements_files": [declaration], "modules": ["packaging"]}})
        self.assertEqual(plan["status"], "success", plan.get("error"))
        self.harness._update_task_progress([plan])
        installed = await self.harness.tools.execute({"id": "modified-offline-install", "name": "install_dependencies", "arguments": {"action": "apply", "plan_id": plan["data"]["plan_id"]}})
        self.assertEqual(installed["status"], "success", installed.get("error"))
        self.harness._update_task_progress([installed])
        self.assertEqual(self.harness.current_instance["environment"]["status"], "ready")
        self.assertTrue(self.harness._maintenance_is_complete())
        task = copy.deepcopy(self.harness.task)
        usage = self.harness.usage_summary()
        usage_path = self.harness.store.path(self.harness.session_relative_path + "/usage.jsonl")
        usage_bytes = usage_path.read_bytes()
        check_ref = self.harness.state["maintenance_progress"]["environment_check"]["evidence_ref"]
        check_bytes = self.harness.store.path(check_ref).read_bytes()
        result = await self.harness._action(self.harness._loop(), task_budget=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.harness.task, task)
        self.assertEqual(self.harness.state["maintenance_progress"]["phase"], "completed")
        self.assertTrue(self.harness._maintenance_is_complete())
        self.assertFalse(self.harness._repair_is_complete())
        self.assertEqual(self.harness.usage_summary(), usage)
        self.assertEqual(usage_path.read_bytes(), usage_bytes)
        self.assertEqual(self.harness.store.path(check_ref).read_bytes(), check_bytes)
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        self.assertEqual(events[-1]["event_type"], "agent.completed")
        self.assertFalse(any(event["event_type"] == "llm.started" for event in events))
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)

    @unittest.skipUnless(os.environ.get("IANVS_OBSERVATION_CAPTURE") and os.environ.get("IANVS_OBSERVATION_SESSION_CAPTURE"), "需要真实未闭合 Observe 响应及同任务归档")
    async def test_captured_observation_correction_preparation_preserves_actual_usage(self):
        source = Path(os.environ["IANVS_OBSERVATION_CAPTURE"])
        session = Path(os.environ["IANVS_OBSERVATION_SESSION_CAPTURE"])
        originals = {path: path.read_bytes() for path in (source, source.parent.parent / (source.parent.name + ".json"), session / "session_snapshot.json", session / "usage.jsonl")}
        captured = json.loads(originals[source])
        response = captured["response"]
        snapshot = json.loads(originals[session / "session_snapshot.json"])
        self.assertEqual(captured["purpose"], "observe")
        self.assertEqual(response["finish_reason"], "stop")
        self.assertLess(response["usage"]["output_tokens"], captured["request"]["max_tokens"])
        chunks = "".join(choice["delta"].get("content", "") for chunk in captured["raw_chunks"] for choice in chunk.get("choices", []) if choice["index"] == 0)
        self.assertEqual(chunks, response["content"])
        with self.assertRaises(json.JSONDecodeError):
            self.harness._observation(response["content"])
        self.harness.task, self.harness.state, self.harness._messages = copy.deepcopy((snapshot["task"], snapshot["state"], snapshot["history"]))
        self.harness.session = copy.deepcopy(snapshot["session"])
        self.harness.session_relative_path = f"instances/{self.harness.task['instance_id']}/sessions/{self.harness.session['session_id']}"
        self.harness.events.bind(task_id=self.harness.task["task_id"], instance_id=self.harness.task["instance_id"], session_id=self.harness.session["session_id"])
        self.harness._connect_provider()
        self.harness.store.path(self.harness.usage.path).parent.mkdir(parents=True, exist_ok=True)
        self.harness.store.path(self.harness.usage.path).write_bytes(originals[session / "usage.jsonl"])
        reference = self.harness.session_relative_path + f"/requests/{response['usage']['request_group_id']}.json"
        self.harness.store.write(reference, json.loads(originals[source.parent.parent / (source.parent.name + ".json")]))
        original = copy.deepcopy((self.harness.task, self.harness.state, self.harness.history()))
        usage = self.harness.usage_summary()
        self.harness.settings.max_input_tokens = 1
        self.harness.settings.max_output_tokens = 1
        self.harness.settings.max_model_calls = 1
        self.harness.settings.max_tool_calls = 1
        self.harness.usage.check_budget(self.harness.settings)
        prepared = await self.harness.context.prepare(self.harness.task, self.harness.state, self.harness.history(), [], role="observe", correction=OBSERVATION_CORRECTION)
        self.assertEqual(prepared["tools"], [])
        self.assertEqual(prepared["messages"][-1]["content"], OBSERVATION_CORRECTION)
        self.assertEqual(prepared["estimated_tokens"], self.harness.context.estimator.request(prepared["messages"], []))
        self.assertEqual(json.loads(prepared["messages"][1]["content"].split("\n", 1)[1])["goal"], self.harness.task["goal"])
        self.assertEqual(self.harness.store.read(self.harness.session_relative_path + f"/tool-output/TASK-{self.harness.task['task_id']}.json"), self.harness.task)
        self.assertEqual(self.harness.store.read(reference)["response"], response)
        self.assertEqual((self.harness.task, self.harness.state, self.harness.history()), original)
        self.assertEqual(self.harness.usage_summary()["model_calls"], usage["model_calls"])
        self.assertEqual(self.harness.usage_summary()["task_effective_input_tokens"], usage["task_effective_input_tokens"])
        self.assertIsNone(self.harness.provider.client)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        self.harness.store.write("captured-observation-check.json", {"source": str(source), "session_capture": str(session), "request_id": response["request_id"], "task_id": self.harness.task["task_id"], "prior_input_tokens": usage["task_effective_input_tokens"], "new_model_calls": 0, "estimated_correction_input_tokens": prepared["estimated_tokens"], "original_active_execution_seconds": self.harness.state["active_execution_seconds"]})

    @unittest.skipUnless(os.environ.get("IANVS_MAINTENANCE_CAPTURE"), "需要真实安装请求会话 IANVS_MAINTENANCE_CAPTURE")
    async def test_captured_install_request_keeps_chinese_facts_and_authorized_actions(self):
        source = Path(os.environ["IANVS_MAINTENANCE_CAPTURE"])
        original = (source / "session_snapshot.json").read_bytes()
        snapshot = json.loads(original)
        task, state, history = copy.deepcopy((snapshot["task"], snapshot["state"], snapshot["history"]))
        chinese = [item for item in state["facts"] if isinstance(item, dict) and "事实" in item]
        self.assertTrue(chinese, "必须使用已发生中文字段遗漏的真实记录")
        result = format_task_result(task, state, state["status"], state.get("error", ""))
        self.assertTrue(all(any(fact["text"] == item["事实"] and fact["evidence_ref"] == item["evidence_ref"] for fact in result["facts"]) for item in chinese))
        self.assertEqual(result["verification"], state["verification"])
        prepared = await self.harness.context.prepare(task, state, history, self.harness.tools.definitions())
        self.assertIn(task["goal"], prepared["messages"][-1]["content"])
        self.assertIn("安装并检查结果", prepared["messages"][-1]["content"])
        self.assertNotIn("不要自动检查环境", prepared["messages"][-1]["content"])
        usages = [json.loads(line) for line in (source / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
        failure = next(item for item in reversed(usages) if item.get("failure_stage") == "absolute_deadline")
        self.assertEqual(failure["deadline_seconds"], 120)
        self.assertGreater(failure["chunk_count"], 0)
        self.assertFalse(failure["context_limit"])
        self.assertEqual((source / "session_snapshot.json").read_bytes(), original)
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)
        self.harness.store.write("captured-maintenance-check.json", {"source": str(source), "task_id": task["task_id"], "chinese_facts": len(chinese), "rendered_facts": len(result["facts"]), "original_deadline_seconds": failure["deadline_seconds"], "stream_chunks_before_failure": failure["chunk_count"], "model_calls": 0})


if __name__ == "__main__":
    unittest.main()
