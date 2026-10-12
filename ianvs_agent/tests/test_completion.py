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

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import load_settings
from ianvs_agent.message_types import format_task_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        artifacts = Path(os.environ.get("IANVS_COMPLETION_TEST_ROOT", PROJECT_ROOT / "workspace/completion-tests"))
        self.root = artifacts / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        source = self.root / "source"
        shutil.copytree(PROJECT_ROOT / "examples/llm_simple_qa", source / "examples/llm_simple_qa", ignore=shutil.ignore_patterns("__pycache__"))
        settings = replace(load_settings(PROJECT_ROOT, load_credentials=False), project_root=source, state_root=self.root / "state", python_executable=sys.executable, approval_policy="auto_scoped")
        self.harness = AgentHarness(settings)
        await self.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
        self.harness._ensure_task("安装当前实例环境并运行原 Example，修改指定功能并验证")
        self.sequence = 0

    async def asyncTearDown(self):
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        await self.harness.close()

    async def tool(self, name, **arguments):
        self.sequence += 1
        result = await self.harness.tools.execute({"id": f"completion-{self.sequence}", "name": name, "arguments": arguments})
        self.assertEqual(result["status"], "success", result)
        self.harness._update_task_progress([result])
        return result

    async def declare(self, *kinds, answer=None):
        requirements = [{"id": kind, "kind": kind, "description": {"answer": "说明当前问题", "environment": "当前实例离线安装 packaging 并检查", "benchmark": "运行原 Example 并独立验证", "code": "新增 feature.value 并真实检查返回值"}[kind]} for kind in kinds]
        arguments = {"requirements": requirements}
        if answer is not None:
            arguments["answer_text"] = answer
        await self.tool("declare_completion_contract", **arguments)
        return requirements

    async def patch(self, name, content):
        await self.tool("read_file", path=name)
        target = Path(self.harness.current_instance["code_root"]) / name
        before = target.read_text(encoding="utf-8") if target.exists() else ""
        patch = "".join(difflib.unified_diff(before.splitlines(keepends=True), content.splitlines(keepends=True), fromfile="a/" + name if target.exists() else "/dev/null", tofile="b/" + name))
        return await self.tool("apply_patch", patch=patch)

    def require_linux(self):
        if os.name != "posix" or not shutil.which("bwrap"):
            self.skipTest("真实隔离命令需要 Linux bwrap")

    async def install(self):
        self.require_linux()
        wheel_root = os.environ.get("IANVS_ENVIRONMENT_WHEEL_ROOT")
        if not wheel_root:
            self.skipTest("需要本地真实 packaging26.3 wheel")
        code = Path(self.harness.current_instance["code_root"])
        (code / "wheels").mkdir(exist_ok=True)
        shutil.copy2(Path(wheel_root) / "packaging-26.3-py3-none-any.whl", code / "wheels")
        declaration = "completion-requirements.txt"
        await self.patch(declaration, "--no-index\n--find-links ./wheels\npackaging==26.3\n")
        await self.harness.environment.prepare(self.harness.current_instance, install=False)
        self.harness.current_instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        plan = await self.tool("install_dependencies", action="plan", requirements_files=[declaration], modules=["packaging"])
        await self.tool("install_dependencies", action="apply", plan_id=plan["data"]["plan_id"])

    async def test_contract_is_immutable_and_restored_from_formal_records(self):
        requirements = await self.declare("environment", "benchmark", "code")
        task = copy.deepcopy(self.harness.task)
        rejected = await self.harness.tools.execute({"id": "remove-benchmark", "name": "declare_completion_contract", "arguments": {"requirements": requirements[:1]}})
        self.assertEqual(rejected["status"], "error")
        self.assertIn("不能删除", rejected["error"])
        self.assertEqual(self.harness.task, task)
        await self.harness._save_session_async()
        settings = self.harness.settings
        await self.harness.close()
        self.harness = AgentHarness(settings)
        self.assertEqual(self.harness.task, task)
        self.assertEqual(self.harness.completion.contract(), task["completion_contract"])
        messages = self.harness.context._pinned(self.harness.task, self.harness.state, "observe")
        pinned = "\n".join(item["content"] for item in messages)
        self.assertIn(task["goal"], pinned)
        self.assertIn("completion_contract", pinned)
        self.assertIn("新增 feature.value", pinned)
        self.assertIn("source_event", pinned)

    async def test_answer_declaration_returns_answer_without_provider(self):
        await self.declare("answer", answer="当前实例有独立源码目录。")
        result = await self.harness._complete_goal()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["content"], "当前实例有独立源码目录。")
        self.assertIsNone(self.harness.provider.client)
        messages = [event for event in self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl") if event["event_type"] == "agent.message"]
        self.assertEqual(messages[-1]["payload"]["content"], result["content"])

    async def test_action_without_contract_does_not_change_files(self):
        self.harness.state["status"] = "running"
        result = await self.harness.tools.execute({"id": "undeclared-run", "name": "run_example", "arguments": {}})
        self.assertEqual(result["status"], "error")
        self.assertIn("declare_completion_contract", result["error"])
        self.assertIsNone(self.harness.current_instance.get("last_run_id"))
        rejected = await self.harness.tools.execute({"id": "answer-after-action", "name": "declare_completion_contract", "arguments": {"requirements": [{"id": "answer", "kind": "answer", "description": "说明问题"}], "answer_text": "处理完成"}})
        self.assertEqual(rejected["status"], "error")
        self.assertIsNone(self.harness.completion.contract())

    async def test_repair_minimum_does_not_hide_other_requirements(self):
        self.harness.task["kind"] = "repair"
        requirements = await self.declare("environment", "benchmark", "code")
        self.assertEqual(self.harness.completion.contract()["requirements"], requirements)
        self.assertEqual(self.harness.task["acceptance"]["repair_task_id"], self.harness.task["task_id"])

    async def test_legacy_repair_requires_declaration_and_keeps_original_goal(self):
        self.harness.task.pop("completion_mode")
        self.harness.task["kind"] = "repair"
        original = copy.deepcopy(self.harness.task)
        # 旧任务的数据结构兼容检查，不制造 Run、验证通过或模型响应记录。
        self.assertIsNone(await self.harness._complete_goal({"decision": "completed"}))
        self.assertFalse(self.harness.state["completion_progress"]["ready"])
        self.assertIn("完整成果清单", self.harness.state["completion_progress"]["missing"][0])
        requirements = await self.declare("environment", "benchmark", "code")
        contract = self.harness.completion.contract()
        self.assertEqual(contract["original_goal"], original["goal"])
        self.assertEqual(contract["requirements"], requirements)
        self.assertEqual(self.harness.task["original_goal"], original["original_goal"])
        self.assertEqual(self.harness.task["acceptance"]["original_benchmark_path"], original["acceptance"]["original_benchmark_path"])

    async def test_run_button_keeps_undeclared_legacy_task_and_requests_full_goal(self):
        self.harness.task.pop("completion_mode")
        credential_name = "IANVS_F2_LEGACY_TEST_KEY_" + uuid.uuid4().hex
        self.harness.settings.profile.api_key_env = credential_name
        for kind in ("repair", "maintenance", "conversation"):
            with self.subTest(kind=kind):
                self.harness.task["kind"] = kind
                self.harness.state["status"] = "pending"
                original = copy.deepcopy(self.harness.task)
                with self.assertRaisesRegex(ValueError, "缺少模型凭据环境变量 " + credential_name):
                    await self.harness.run_current()
                self.assertEqual(self.harness.task, original)
                self.assertEqual(self.harness.state["status"], "failed")
                self.assertIsNone(self.harness.current_instance.get("last_run_id"))
                self.assertFalse(self.harness.state["completion_progress"]["ready"])
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        self.assertFalse(any(event["event_type"] in {"agent.completed", "run.started", "llm.started"} for event in events))

    @unittest.skipUnless(os.environ.get("IANVS_COMPLETION_PASSED_VERIFICATION"), "需要真实已保存的独立验证通过记录")
    async def test_pending_benchmark_does_not_display_historical_success_as_current(self):
        source = Path(os.environ["IANVS_COMPLETION_PASSED_VERIFICATION"])
        original = source.read_bytes()
        verification = json.loads(original)
        self.assertIs(verification["passed"], True)
        self.assertTrue(verification["run_id"])
        await self.declare("benchmark")
        self.harness.state["verification"] = verification
        self.assertFalse(self.harness.completion.evaluate()["ready"])
        for status in ("failed", "blocked"):
            content = format_task_result(self.harness.task, self.harness.state, status, "继续完成本任务")["content"]
            self.assertIn("待完成", content)
            self.assertIn("原始评测独立验证：尚未通过", content)
            self.assertNotIn("原始评测独立验证：通过", content)
            self.assertIn(f"历史运行 {verification['run_id']} 曾通过", content)
        self.assertEqual(source.read_bytes(), original)

    @unittest.skipUnless(os.environ.get("IANVS_COMPLETION_PASSED_VERIFICATION"), "需要既有真实成功 Run 与独立验证记录")
    async def test_current_task_pending_overrides_real_historical_success_in_context(self):
        verification_source = Path(os.environ["IANVS_COMPLETION_PASSED_VERIFICATION"])
        run_source = verification_source.parent / "run.json"
        originals = {path: path.read_bytes() for path in (verification_source, run_source)}
        verification = json.loads(originals[verification_source])
        run = json.loads(originals[run_source])
        self.assertIs(verification["passed"], True)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(verification["run_id"], run["run_id"])
        historical_claim = run["run_id"] + " 已成功运行并通过独立验证"
        await self.tool("declare_completion_contract", requirements=[{"id": kind, "kind": kind, "description": historical_claim} for kind in ("environment", "benchmark", "code")])
        contract = copy.deepcopy(self.harness.completion.contract())
        self.harness.state["verification"] = verification
        self.harness.state["last_run"] = self.harness.run_summary(run)
        self.assertIsNone(await self.harness._complete_goal({"decision": "completed"}))
        progress = self.harness.state["completion_progress"]
        self.assertFalse(progress["ready"])
        self.assertEqual(progress["evidence_scope"]["task_id"], self.harness.task["task_id"])
        self.assertEqual(progress["evidence_scope"]["original_goal"], self.harness.task["original_goal"])
        self.assertNotIn(historical_claim, "\n".join(progress["missing"]))
        self.assertIn("inspect_environment", progress["requirements"][0]["pending_reason"])
        self.assertIn("run_example", progress["requirements"][1]["pending_reason"])
        self.assertIn("completion_requirement_id", progress["requirements"][2]["pending_reason"])
        for role in ("plan", "observe"):
            prepared = await self.harness.context.prepare(self.harness.task, self.harness.state, self.harness.history(), [], role=role)
            tail = prepared["messages"][-1]["content"]
            focus = json.loads(tail.rsplit("\n", 1)[-1])
            self.assertEqual(focus["task_id"], self.harness.task["task_id"])
            self.assertFalse(focus["ready"])
            self.assertEqual([item["status"] for item in focus["requirements"]], ["pending"] * 3)
            self.assertTrue(all(item["pending_reason"] and item["evidence"] is None for item in focus["requirements"]))
            self.assertTrue(all("description" not in item for item in focus["requirements"]))
            self.assertEqual(focus["next_goal"], self.harness.state["next_goal"])
            self.assertIn("run_example", focus["next_goal"])
            self.assertNotIn(historical_claim, tail)
            self.assertIn("历史 verification.passed 仅作背景", tail)
            self.assertIs(json.loads(prepared["messages"][2]["content"].split("\n", 1)[-1])["verification"]["passed"], True)
        compact = self.harness.context._compaction_messages(self.harness.task, self.harness.state, self.harness.history(), None)
        formal = json.loads(compact[1]["content"].split("\n", 1)[-1])
        self.assertEqual(formal["state"]["completion_progress"], progress)
        self.assertEqual(formal["task"]["completion_contract"], contract)
        self.assertEqual(self.harness.completion.contract(), contract)
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    async def test_existing_contract_only_appends_for_recorded_user_update(self):
        requirements = await self.declare("environment")
        extra = {"id": "benchmark", "kind": "benchmark", "description": "额外运行原 Example 并验证"}
        rejected = await self.harness.tools.execute({"id": "unrequested-extra", "name": "declare_completion_contract", "arguments": {"requirements": requirements + [extra]}})
        self.assertEqual(rejected["status"], "error")
        original = copy.deepcopy(self.harness.completion.contract())
        event = self.harness.events.emit("task.user_instruction", {"text": "安装后再运行原 Example"})
        self.harness.task["goal_updates"] = [{"text": "安装后再运行原 Example", "event_id": event["event_id"]}]
        self.harness.state["completion_revision_pending"] = True
        await self.tool("declare_completion_contract", requirements=requirements + [extra])
        current = self.harness.completion.contract()
        self.assertEqual(current["version"], 2)
        self.assertEqual(current["original_goal"], original["original_goal"])
        self.assertEqual(current["previous_ref"], original["evidence_ref"])
        self.assertEqual(self.harness.store.read(original["evidence_ref"]), original)
        self.assertEqual(current["user_updates"][0]["event_id"], event["event_id"])

    async def test_explicit_resume_with_suffix_preserves_goal_and_allows_additions(self):
        requirements = await self.declare("environment", "benchmark")
        original = copy.deepcopy(self.harness.task)
        contract = copy.deepcopy(self.harness.completion.contract())
        usage_path = self.harness.store.path(self.harness.session_relative_path + "/usage.jsonl")
        usage_bytes = usage_path.read_bytes()
        credential_name = "IANVS_RESUME_SUFFIX_TEST_KEY_" + uuid.uuid4().hex
        self.harness.settings.profile.api_key_env = credential_name
        cases = (
            ("conversation", "failed", "继续当前任务，完成剩余要求。"),
            ("conversation", "blocked", "继续原任务；另外增加功能检查。"),
            ("conversation", "interrupted", "continue, finish remaining checks."),
            ("maintenance", "failed", "/resume：继续完成原评测。"),
            ("maintenance", "blocked", "继续任务，另外增加功能检查。"),
            ("maintenance", "interrupted", "继续，保留原要求并完善功能。"),
        )
        for kind, status, instruction in cases:
            with self.subTest(kind=kind, status=status):
                self.harness.task["kind"] = kind
                self.harness.state["status"] = status
                with self.assertRaisesRegex(ValueError, "缺少模型凭据环境变量 " + credential_name):
                    await self.harness.send_message(instruction)
                for field in ("task_id", "goal", "acceptance", "original_goal", "completion_contract"):
                    self.assertEqual(self.harness.task[field], original[field])
                self.assertEqual(self.harness.completion.contract(), contract)
                self.assertTrue(self.harness.state["completion_revision_pending"])
                self.assertFalse(self.harness.state["completion_progress"]["ready"])
                self.assertEqual(self.harness.task["goal_updates"][-1]["text"], instruction)
                source = self.harness.task["goal_updates"][-1]["event_id"]
                events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
                event = next(item for item in events if item["event_id"] == source)
                self.assertEqual(event["task_id"], original["task_id"])
                self.assertEqual(event["payload"]["text"], instruction)
                self.assertTrue(event["payload"]["resume"])
                self.assertEqual(usage_path.read_bytes(), usage_bytes)
        prior_updates = copy.deepcopy(self.harness.task["goal_updates"])
        with self.assertRaisesRegex(ValueError, "缺少模型凭据环境变量 " + credential_name):
            await self.harness.send_message("继续当前任务！")
        self.assertEqual(self.harness.task["goal_updates"], prior_updates)
        self.assertEqual(self.harness.completion.contract(), contract)
        self.assertEqual(usage_path.read_bytes(), usage_bytes)
        rejected = await self.harness.tools.execute({"id": "resume-cannot-drop", "name": "declare_completion_contract", "arguments": {"requirements": requirements[:1]}})
        self.assertEqual(rejected["status"], "error")
        self.assertIn("不能删除", rejected["error"])
        extra = {"id": "code", "kind": "code", "description": "增加用户续作指令要求的功能检查"}
        await self.tool("declare_completion_contract", requirements=requirements + [extra])
        current = self.harness.completion.contract()
        self.assertEqual(current["version"], 2)
        self.assertEqual(current["requirements"], requirements + [extra])
        self.assertEqual(current["original_goal"], original["goal"])
        self.assertEqual(current["user_updates"], self.harness.task["goal_updates"])
        self.assertEqual(self.harness.store.read(contract["evidence_ref"]), contract)
        self.assertFalse(self.harness.state.get("completion_revision_pending"))

    async def test_non_resume_instruction_starts_new_task_without_prefix_guessing(self):
        await self.declare("environment")
        original = copy.deepcopy(self.harness.task)
        self.harness.state["status"] = "blocked"
        instruction = "继续学习 Python 的依赖管理是什么意思？"
        credential_name = "IANVS_NEW_TASK_PREFIX_TEST_KEY_" + uuid.uuid4().hex
        self.harness.settings.profile.api_key_env = credential_name
        with self.assertRaisesRegex(ValueError, "缺少模型凭据环境变量 " + credential_name):
            await self.harness.send_message(instruction)
        self.assertNotEqual(self.harness.task["task_id"], original["task_id"])
        self.assertEqual(self.harness.task["goal"], instruction)
        self.assertIsNone(self.harness.completion.contract())
        archived = self.harness.store.read(self.harness.session_relative_path + f"/tasks/{original['task_id']}.json")
        self.assertEqual(archived["task"], original)

    async def test_environment_success_cannot_complete_install_and_run(self):
        await self.declare("environment", "benchmark")
        await self.install()
        progress = self.harness.completion.evaluate()
        self.assertEqual([item["status"] for item in progress["requirements"]], ["passed", "pending"])
        self.assertFalse(progress["ready"])
        # 直接校验程序对结构化完成请求的证据要求，不模拟模型推理。
        result = await self.harness._complete_goal({"decision": "completed", "next_goal": "完成"})
        self.assertIsNone(result)
        self.assertEqual(self.harness.state["observation_decision"], "continue")
        self.assertIn("原 Example", self.harness.state["next_goal"])
        content = format_task_result(self.harness.task, self.harness.state, "failed", "仍有未完成成果")["content"]
        self.assertIn("已验证", content)
        self.assertIn("待完成", content)
        self.assertIsNone(self.harness.current_instance.get("last_run_id"))

    async def test_environment_only_requires_observe_and_pending_plan_invalidates(self):
        await self.declare("environment")
        await self.install()
        self.assertTrue(self.harness.completion.evaluate()["ready"])
        self.harness.task["kind"] = "conversation"
        self.assertTrue(self.harness._requires_observation([{"name": "inspect_environment", "arguments": {}}]))
        self.assertIsNone(await self.harness._complete_goal())
        self.assertIsNone(await self.harness._complete_goal({"decision": "continue"}))
        result = await self.harness._complete_goal({"decision": "completed", "next_goal": "环境已准备并检查"})
        self.assertEqual(result["status"], "completed")
        self.harness.state["maintenance_progress"]["phase"] = "checks_pending"
        content = format_task_result(self.harness.task, self.harness.state, "completed", "环境已准备并检查")["content"]
        self.assertIn("已验证", content)
        self.assertNotIn("尚未取得", content)
        self.assertNotIn("原始评测独立验证", content)
        await self.tool("install_dependencies", action="plan", requirements_files=["completion-requirements.txt"], modules=["packaging"])
        self.assertFalse(self.harness.completion.evaluate()["ready"])
        await self.tool("prepare_instance")
        await self.tool("inspect_environment", modules=["packaging"])
        self.assertTrue(self.harness.state["maintenance_progress"]["install_pending"])
        self.assertFalse(self.harness.completion.evaluate()["ready"])

    async def test_failed_plan_does_not_block_subsequent_real_preparation_and_check(self):
        await self.declare("environment")
        await self.install()
        failed = await self.harness.tools.execute({"id": "real-failed-install-plan", "name": "install_dependencies", "arguments": {"action": "plan", "requirements_files": ["missing-environment-requirements.txt"]}})
        self.assertEqual(failed["status"], "error")
        self.harness._update_task_progress([failed])
        instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        failed_ref = instance["environment_operation_ref"]
        plan = self.harness.store.read(failed_ref)
        self.assertEqual(plan["status"], "invalid")
        self.assertFalse(plan["dependencies"]["passed"])
        self.assertTrue(self.harness.store.path(plan["parse_operation_ref"]).with_name("worker-output").joinpath("observation.json").is_file())
        failed_bytes = self.harness.store.path(failed_ref).read_bytes()
        prepared = await self.tool("prepare_instance")
        self.assertEqual(prepared["data"]["status"], "ready")
        checked = await self.tool("inspect_environment", modules=["packaging"])
        self.assertTrue(checked["data"]["passed"])
        self.assertFalse(self.harness.state["maintenance_progress"]["install_pending"])
        self.assertTrue(self.harness._maintenance_is_complete())
        self.assertTrue(self.harness.completion.evaluate()["ready"])
        self.assertIsNone(await self.harness._complete_goal({"decision": "continue"}))
        completed = await self.harness._complete_goal({"decision": "completed", "next_goal": "环境准备与完整检查已通过"})
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(self.harness.store.path(failed_ref).read_bytes(), failed_bytes)
        recorded = next(item for item in self.harness.completion._task_tools() if item["operation_id"] == failed["operation_id"])
        self.assertEqual(recorded["status"], "error")

    async def test_applying_earlier_plan_preserves_latest_unexecuted_plan(self):
        await self.declare("environment")
        await self.install()
        first = await self.tool("install_dependencies", action="plan", requirements_files=["completion-requirements.txt"], modules=["packaging"])
        latest = await self.tool("install_dependencies", action="plan", requirements_files=["completion-requirements.txt"], modules=["packaging"])
        self.assertNotEqual(first["data"]["plan_id"], latest["data"]["plan_id"])
        installed = await self.tool("install_dependencies", action="apply", plan_id=first["data"]["plan_id"])
        self.assertTrue(installed["data"]["postcheck"]["passed"])
        self.assertEqual(self.harness.store.read(first["data"]["evidence_ref"])["status"], "completed")
        self.assertEqual(self.harness.store.read(latest["data"]["evidence_ref"])["status"], "planned")
        self.assertTrue(self.harness.state["maintenance_progress"]["install_pending"])
        self.assertEqual(self.harness.state["maintenance_progress"]["pending_plan_id"], latest["data"]["plan_id"])
        self.assertFalse(self.harness._maintenance_is_complete())
        self.assertFalse(self.harness.completion.evaluate()["ready"])

    async def test_real_function_check_requires_current_diff_and_version(self):
        self.require_linux()
        await self.declare("code")
        await self.harness.environment.prepare(self.harness.current_instance, install=False)
        self.harness.current_instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        await self.patch("feature.py", "value = 7\n")
        self.assertFalse(self.harness.completion.evaluate()["ready"])
        checked = await self.tool("execute_environment_command", argv=["python", "-c", "from feature import value; assert value == 7; print(value)"], completion_requirement_id="code")
        self.assertEqual(checked["data"]["exit_code"], 0)
        self.assertTrue(checked["data"]["command_executed"])
        self.assertTrue(self.harness.completion.evaluate()["ready"])
        failed_same_version = await self.harness.tools.execute({"id": "real-failing-check-same-version", "name": "execute_environment_command", "arguments": {"argv": ["python", "-c", "from feature import value; assert value == 8"], "completion_requirement_id": "code"}})
        self.assertEqual(failed_same_version["status"], "error")
        self.assertFalse(self.harness.completion.evaluate()["ready"])
        await self.tool("execute_environment_command", argv=["python", "-c", "from feature import value; assert value == 7"], completion_requirement_id="code")
        self.assertTrue(self.harness.completion.evaluate()["ready"])
        await self.patch("feature.py", "value = 8\n")
        self.assertFalse(self.harness.completion.evaluate()["ready"])
        failed = await self.harness.tools.execute({"id": "real-failing-functional-check", "name": "execute_environment_command", "arguments": {"argv": ["python", "-c", "from feature import value; assert value == 7"], "completion_requirement_id": "code"}})
        self.assertEqual(failed["status"], "error")
        self.assertFalse(self.harness.completion.evaluate()["ready"])

    async def test_reverted_patch_cannot_satisfy_code_requirement(self):
        self.require_linux()
        root = Path(self.harness.current_instance["code_root"])
        (root / "feature.py").write_text("value = 1\n", encoding="utf-8")
        await self.declare("code")
        await self.harness.environment.prepare(self.harness.current_instance, install=False)
        self.harness.current_instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        await self.patch("feature.py", "value = 7\n")
        await self.patch("feature.py", "value = 1\n")
        await self.tool("execute_environment_command", argv=["python", "-c", "from feature import value; assert value == 1"], completion_requirement_id="code")
        self.assertFalse(self.harness.completion.evaluate()["ready"])

    async def test_real_package_drift_invalidates_environment_and_function_evidence(self):
        await self.declare("environment", "code")
        await self.install()
        await self.patch("feature.py", "import packaging\nvalue = packaging.__version__\n")
        command = ["python", "-c", "from feature import value; assert value == '26.3'"]
        await self.tool("execute_environment_command", argv=command, completion_requirement_id="code")
        self.assertTrue(self.harness.completion.evaluate()["ready"])
        environment = self.harness.current_instance["environment"]
        process = await asyncio.create_subprocess_exec(environment["python_executable"], "-m", "pip", "uninstall", "-y", "packaging", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await process.communicate()
        (self.root / "package-drift-uninstall.log").write_bytes(output)
        self.assertEqual(process.returncode, 0, output)
        # 实际删除当前实例中的包；最终完成核对必须重新发现漂移。
        self.assertIsNone(await self.harness._complete_goal({"decision": "completed"}))
        self.assertTrue(self.harness.current_instance["environment_changed"])
        self.assertEqual([item["status"] for item in self.harness.state["completion_progress"]["requirements"]], ["pending", "pending"])
        self.assertEqual(self.harness.current_instance["environment"]["generation"], environment["generation"])
        plan = await self.tool("install_dependencies", action="plan", requirements_files=["completion-requirements.txt"], modules=["packaging"])
        await self.tool("install_dependencies", action="apply", plan_id=plan["data"]["plan_id"])
        self.harness.current_instance = await self.harness._run_sync(self.harness.instances.check_changes, self.harness.current_instance["instance_id"], False)
        self.assertIsNone(await self.harness._complete_goal())
        self.assertFalse(self.harness.current_instance["environment_changed"])
        self.assertEqual([item["status"] for item in self.harness.state["completion_progress"]["requirements"]], ["passed", "pending"])
        await self.tool("execute_environment_command", argv=command, completion_requirement_id="code")
        self.assertTrue(self.harness.completion.evaluate()["ready"])
        completed = await self.harness._complete_goal({"decision": "completed"})
        self.assertEqual(completed["status"], "completed")
        original = copy.deepcopy(self.harness.task)
        environment = self.harness.current_instance["environment"]
        process = await asyncio.create_subprocess_exec(environment["python_executable"], "-m", "pip", "uninstall", "-y", "packaging", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await process.communicate()
        (self.root / "completed-resume-package-drift.log").write_bytes(output)
        self.assertEqual(process.returncode, 0, output)
        credential_name = "IANVS_COMPLETED_RESUME_KEY_" + uuid.uuid4().hex
        self.harness.settings.profile.api_key_env = credential_name
        with self.assertRaisesRegex(ValueError, "缺少模型凭据环境变量 " + credential_name):
            await self.harness.resume_current()
        self.assertEqual(self.harness.task, original)
        self.assertEqual(self.harness.state["status"], "failed")
        self.assertEqual(self.harness.state["resume_count"], 1)
        self.assertFalse(self.harness.state["completion_progress"]["ready"])


if __name__ == "__main__":
    unittest.main()
