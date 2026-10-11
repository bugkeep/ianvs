# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import os
import unittest
import uuid
from pathlib import Path

from filelock import FileLock

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings, ModelProfile, load_settings
from ianvs_agent.context.builder import ContextManager
from ianvs_agent.context.compactor import complete_interactions, history_split, message_fingerprint
from ianvs_agent.context.evidence import ReadmeEvidence
from ianvs_agent.llm.provider import Provider
from ianvs_agent.llm.usage import BudgetExceededError, UsageTracker
from ianvs_agent.storage.local_store import LocalStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ContextUsageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/context-usage" / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        self.store = LocalStore(self.root)
        self.events = EventBus(self.store, session_id="SESSION-UNIT", instance_id="INSTANCE-UNIT")
        self.usage = UsageTracker(self.store, "sessions/SESSION-UNIT", self.events)
        self.profile = ModelProfile("测试请求配置", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3", context_window=16000, output_tokens=2048)
        self.settings = AgentSettings(PROJECT_ROOT, self.root, profiles={"ark": self.profile})
        self.provider = Provider(self.profile, self.events, self.usage)
        self.context = ContextManager(self.settings, self.store, self.events, self.provider)

    async def test_preparation_does_not_call_provider(self):
        task = {"task_id": "TASK-UNIT", "goal": "保持原模型和数据，恢复原任务", "acceptance": {"required_metric": "accuracy"}}
        state = {"verification": {"passed": False}, "hypotheses": [{"value": "路径可能错误"}]}
        prepared = await self.context.prepare(task, state, [], [])
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.provider.client)
        self.assertIn("原模型和数据", prepared["messages"][1]["content"])
        self.assertIn('"passed": false', prepared["messages"][2]["content"])
        self.assertGreater(prepared["estimated_tokens"], 0)
        self.assertIsNone(prepared["checkpoint_id"])

    def test_context_budget_configuration_accepts_model_window_or_explicit_limit(self):
        configuration = self.root / "context-config.yaml"
        for value, expected in (("null", None), ("4096", 4096)):
            with self.subTest(value=value):
                configuration.write_text("context_input_budget: " + value + "\n", encoding="utf-8")
                settings = load_settings(self.root, configuration, load_credentials=False)
                self.assertEqual(settings.context_input_budget, expected)
        for value in ("0", "-1", "true", "1.5", '"4096"'):
            with self.subTest(value=value):
                configuration.write_text("context_input_budget: " + value + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "为空值或正整数"):
                    load_settings(self.root, configuration, load_credentials=False)

    def test_default_input_budget_uses_each_selected_model_window(self):
        settings = load_settings(self.root, PROJECT_ROOT / "ianvs_agent/config.example.yaml", load_credentials=False)
        self.assertIsNone(settings.context_input_budget)
        self.assertIsNone(AgentSettings(PROJECT_ROOT, self.root).context_input_budget)
        self.context.settings = settings
        plan_budgets = {}
        for name, profile in settings.profiles.items():
            with self.subTest(profile=name):
                settings.selected_profile = name
                self.provider.profile = profile
                plan_budgets[name] = self.context._input_budget("plan")
                self.assertEqual(plan_budgets[name], profile.context_window - profile.output_tokens)
                self.assertEqual(self.context._input_budget("compaction"), profile.context_window - min(profile.output_tokens, profile.auxiliary_output_tokens))
                self.assertGreater(plan_budgets[name], 32000)
        self.assertNotEqual(plan_budgets["ark"], plan_budgets["glm"])
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.provider.client)

    async def test_real_history_above_32k_prepares_without_model_request(self):
        capture = os.environ.get("IANVS_CONTEXT_CAPTURE_HISTORY")
        if not capture:
            self.skipTest("需要真实会话历史路径 IANVS_CONTEXT_CAPTURE_HISTORY")
        source = Path(capture)
        original = source.read_bytes()
        history = json.loads(original)
        task = json.loads((source.parent / "task.json").read_text(encoding="utf-8"))
        state = json.loads((source.parent / "state.json").read_text(encoding="utf-8"))
        self.profile.context_window = 256000
        self.profile.output_tokens = 8192
        assembled = self.context._assemble(task, state, history, None, "plan")
        estimate = self.context.estimator.request(assembled, [])
        self.assertGreater(estimate, 32000)
        self.assertLess(estimate, self.context._input_budget() * self.settings.context_threshold)
        prepared = await self.context.prepare(task, state, history, [])
        self.assertEqual(prepared["estimated_tokens"], estimate)
        self.assertGreater(prepared["estimated_tokens"], 32000)
        self.assertIsNone(prepared["checkpoint_id"])
        self.assertEqual(prepared["messages"], assembled)
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.provider.client)
        self.assertFalse(any(event["event_type"] == "context.compacted" for event in self.store.read_events("events.jsonl")))
        self.assertEqual(source.read_bytes(), original)

    async def test_explicit_context_input_budget_remains_supported(self):
        self.settings.context_input_budget = 5000
        prepared = await self.context.prepare({"goal": "读取原始任务"}, {}, [], [])
        self.assertEqual(self.context._input_budget(), 5000)
        self.assertLessEqual(prepared["estimated_tokens"], 5000)
        self.settings.context_input_budget = self.profile.context_window * 2
        self.assertEqual(self.context._input_budget(), self.profile.context_window - self.profile.output_tokens)
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.provider.client)

    async def test_insufficient_context_input_budget_stops_before_model_request(self):
        self.settings.context_input_budget = 64
        with self.assertRaisesRegex(BudgetExceededError, "超过输入预算 64"):
            await self.context.prepare({"goal": "保留原始任务与验收要求"}, {"verification": {"passed": False}}, [], [])
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.provider.client)

    async def _captured_input_budget_material(self):
        capture = os.environ.get("IANVS_CONTEXT_CAPTURE_HISTORY")
        if not capture:
            self.skipTest("需要真实会话历史与 Usage 路径 IANVS_CONTEXT_CAPTURE_HISTORY")
        source = Path(capture)
        history = json.loads(source.read_text(encoding="utf-8"))
        task = json.loads((source.parent / "task.json").read_text(encoding="utf-8"))
        state = json.loads((source.parent / "state.json").read_text(encoding="utf-8"))
        self.store.path(self.usage.path).parent.mkdir(parents=True, exist_ok=True)
        self.store.path(self.usage.path).write_bytes((source.parent / "usage.jsonl").read_bytes())
        self.events.bind(task_id=task["task_id"])
        self.profile.context_window = 256000
        self.profile.output_tokens = 8192
        prepared = await self.context.prepare(task, state, history, [])
        return task, state, history, prepared, self.usage.summary()

    async def test_input_budget_preflight_reserves_real_summary_and_next_plan(self):
        task, state, history, prepared, totals = await self._captured_input_budget_material()
        used = totals["task_effective_input_tokens"]
        self.settings.max_input_tokens = int((used + prepared["estimated_tokens"]) / self.settings.context_threshold)
        plan = self.context._input_budget_compaction_plan(task, state, history, [], prepared, totals)
        self.assertEqual(plan["status"], "ready")
        self.assertGreaterEqual(plan["projected_input_tokens"], self.settings.max_input_tokens * self.settings.context_threshold)
        self.assertEqual(plan["used_input_tokens"], used)
        self.assertEqual(plan["budget_base_input_tokens"], 0)
        self.assertEqual(plan["period_input_tokens"], used)
        self.assertLess(plan["post_plan_input_reserve"], prepared["estimated_tokens"])
        self.assertEqual(plan["model_calls_reserve"], 2)
        self.assertEqual(plan["output_reserve"], self.profile.output_tokens + self.profile.auxiliary_output_tokens)
        self.assertLessEqual(plan["compaction_input_tokens"], self.context._input_budget("compaction"))
        self.assertEqual(self.context.estimator.request(plan["messages"]), plan["compaction_input_tokens"])
        complete_interactions(plan["older"])
        complete_interactions(plan["recent"])
        self.assertEqual(self.usage.summary()["model_calls"], totals["model_calls"])
        self.assertIsNone(self.provider.client)

    async def test_cumulative_input_threshold_does_not_block_real_preparation(self):
        task, state, history, prepared, totals = await self._captured_input_budget_material()
        original_usage = self.store.path(self.usage.path).read_bytes()
        self.settings.max_input_tokens = 1
        self.assertGreater(totals["task_effective_input_tokens"], self.settings.max_input_tokens)
        plan = self.context._input_budget_compaction_plan(task, state, history, [], prepared, totals)
        self.assertEqual(plan["status"], "ready", plan)
        self.assertEqual(plan["input_limit"], 1)
        self.assertEqual(plan["used_input_tokens"], totals["task_effective_input_tokens"])
        resumed_provider = Provider(self.profile, self.events, self.usage)
        resumed_context = ContextManager(self.settings, self.store, self.events, resumed_provider)
        restored = await resumed_context.prepare(task, state, history, [])
        self.assertEqual(restored["messages"], prepared["messages"])
        self.assertEqual(self.usage.check_budget(self.settings, restored["estimated_tokens"], self.profile.output_tokens, reserve_model_calls=1), self.usage.summary())
        events = self.store.read_events("events.jsonl")
        self.assertFalse(any(event["event_type"] in {"llm.started", "context.compacted"} for event in events))
        self.assertEqual(self.store.path(self.usage.path).read_bytes(), original_usage)
        self.assertEqual(self.usage.summary()["model_calls"], totals["model_calls"])
        self.assertIsNone(self.provider.client)
        self.assertIsNone(resumed_provider.client)

    async def test_budget_skip_rechecks_new_real_history_without_repeating_notice(self):
        task, state, history, prepared, totals = await self._captured_input_budget_material()
        short = []
        blocks = complete_interactions(history)
        selected_blocks = 0
        for block in blocks:
            short.extend(block)
            selected_blocks += 1
            if block[0].get("role") == "assistant":
                break
        current = await self.context.prepare(task, state, short, [])
        self.settings.max_input_tokens = int((totals["task_effective_input_tokens"] + current["estimated_tokens"]) / self.settings.context_threshold)
        original = await self.context._prepare_input_budget(task, state, short, [], current)
        self.assertEqual(original, current)
        record = self.store.read(self.context._input_budget_record_path(task))
        self.assertEqual(record["reason"], "no_older_history")
        updated = short + blocks[selected_blocks]
        next_prepared = await self.context.prepare(task, state, updated, [])
        await self.context._prepare_input_budget(task, state, updated, [], next_prepared)
        resumed_provider = Provider(self.profile, self.events, self.usage)
        resumed_context = ContextManager(self.settings, self.store, self.events, resumed_provider)
        plan = resumed_context._input_budget_compaction_plan(task, state, history, [], prepared, self.usage.summary())
        self.assertEqual(plan["status"], "ready")
        events = self.store.read_events("events.jsonl")
        self.assertEqual(sum(event["event_type"] == "context.compaction_skipped" for event in events), 1)
        self.assertEqual(self.usage.summary()["model_calls"], totals["model_calls"])
        self.assertIsNone(self.provider.client)
        self.assertIsNone(resumed_provider.client)

    async def test_compaction_estimates_survive_cumulative_call_and_output_thresholds(self):
        task, state, history, prepared, totals = await self._captured_input_budget_material()
        self.settings.max_input_tokens = int((totals["task_effective_input_tokens"] + prepared["estimated_tokens"]) / self.settings.context_threshold)
        self.settings.max_model_calls = 1
        self.settings.max_output_tokens = 1
        plan = self.context._input_budget_compaction_plan(task, state, history, [], prepared, totals)
        self.assertEqual(plan["status"], "ready", plan)
        self.assertEqual(plan["model_calls_reserve"], 2)
        self.assertEqual(plan["output_reserve"], self.profile.output_tokens + self.profile.auxiliary_output_tokens)
        self.assertGreater(totals["task_model_calls"], self.settings.max_model_calls)
        self.assertGreater(totals["task_effective_output_tokens"], self.settings.max_output_tokens)
        self.assertEqual(self.usage.check_budget(self.settings, plan["compaction_input_tokens"] + plan["post_plan_input_reserve"], plan["output_reserve"], reserve_model_calls=2), self.usage.summary())
        self.assertEqual(self.usage.summary()["model_calls"], totals["model_calls"])
        self.assertIsNone(self.provider.client)

    async def test_tool_schema_is_included_in_context_estimate(self):
        task = {"task_id": "TASK-UNIT", "goal": "读取配置"}
        first = await self.context.prepare(task, {}, [], [])
        tool = {"type": "function", "function": {"name": "read_file", "description": "读取配置字段及来源" * 100, "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
        second = await self.context.prepare(task, {}, [], [tool])
        self.assertGreater(second["estimated_tokens"], first["estimated_tokens"])
        observed = await self.context.prepare(task, {}, [], [tool], role="observe")
        self.assertEqual(observed["tools"], [])
        self.assertIn("Observe", observed["messages"][-1]["content"])
        self.assertIn("JSON object", observed["messages"][-1]["content"])

    async def test_current_environment_is_pinned_over_historical_failed_run(self):
        capture = os.environ.get("IANVS_CONTEXT_CAPTURE_INSTANCE")
        if not capture:
            self.skipTest("需要真实已准备实例记录的路径 IANVS_CONTEXT_CAPTURE_INSTANCE")
        source = Path(capture)
        original = source.read_bytes()
        instance = json.loads(original)
        environment = instance["environment"]
        self.assertEqual(environment["status"], "ready")
        run = json.loads((source.parent / "runs/run-0006/run.json").read_text(encoding="utf-8"))
        self.assertIsNone(run["environment_id"])
        self.assertLess(run["completed_at"], environment["completed_at"])
        self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        state = {"last_run": {key: run.get(key) for key in ("run_id", "environment_id", "status", "error", "completed_at")}, "verification": {"passed": False}}
        previous = copy.deepcopy(state)
        task = {"instance_id": instance["instance_id"], "kind": "conversation", "goal": "说明此前运行的失败原因与当前环境准备结果"}
        prepared = await self.context.prepare(task, state, [], [])
        pinned = next(message["content"] for message in prepared["messages"] if message["content"].startswith("程序读取的当前实例运行环境"))
        current = json.loads(pinned.split("\n", 1)[1])
        summary = current["environment"]
        self.assertEqual(summary["environment_id"], environment["environment_id"])
        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["package_count"], len(environment["packages"]))
        self.assertEqual(summary["python_executable"], environment["python_executable"])
        self.assertEqual(summary["completed_at"], environment["completed_at"])
        self.assertEqual([step["log_ref"] for step in summary["steps"]], [step["log_ref"] for step in environment["steps"]])
        self.assertNotIn("packages", summary)
        self.assertIn("历史 Run", pinned)
        self.assertIn('"environment_id": null', prepared["messages"][2]["content"])
        self.assertEqual(state, previous)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.provider.client)

    async def test_observe_uses_only_current_complete_tool_interaction(self):
        old_observation = json.dumps({"facts": [], "hypotheses": [], "next_goal": "较早观察内容不应重复发送", "decision": "continue"}, ensure_ascii=False)
        history = [
            {"role": "user", "content": "较早的大量调查材料" * 10000},
            {"role": "assistant", "content": old_observation, "_ianvs_stage": "observe"},
            {"role": "assistant", "content": "读取当前配置", "tool_calls": [{"id": "current-call", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"config.yaml"}'}}]},
            {"role": "tool", "tool_call_id": "current-call", "content": json.dumps({"operation_id": "TOOL-CURRENT", "name": "read_file", "data": {"path": "config.yaml", "content": "当前配置值"}})},
        ]
        task = {"task_id": "TASK-UNIT", "goal": "保持原始模型", "acceptance": {"required_checks": ["original_run"]}, "readme_path": "README.md"}
        prepared = await self.context.prepare(task, {"verification": {"passed": False}}, history, [], role="observe")
        text = json.dumps(prepared["messages"], ensure_ascii=False)
        self.assertNotIn("较早的大量调查材料", text)
        self.assertNotIn("较早观察内容不应重复发送", text)
        self.assertNotIn("README 原文", text)
        self.assertIn("TOOL-CURRENT", text)
        self.assertIn("original_run", text)
        self.assertIn('"passed": false', prepared["messages"][2]["content"])
        self.assertEqual([message["role"] for message in prepared["messages"][-3:]], ["assistant", "tool", "user"])
        self.assertEqual(self.usage.summary()["model_calls"], 0)

    async def test_plan_uses_formal_facts_without_repeating_old_observe_json(self):
        observation = {"facts": [{"value": "已登记配置证据", "evidence_ref": "tool:TOOL-OLD"}], "hypotheses": [], "next_goal": "较早观察的长篇文字", "decision": "continue"}
        goal = "只回答：可以继续。不要调用工具。"
        history = [{"role": "assistant", "content": json.dumps(observation, ensure_ascii=False)}, {"role": "user", "content": goal}]
        state = {"facts": observation["facts"], "verification": {"passed": False, "stale": True}, "next_goal": "继续检查历史模型环境"}
        prepared = await self.context.prepare({"kind": "conversation", "goal": goal, "acceptance": {"required_checks": ["original_run"]}}, state, history, [])
        text = json.dumps(prepared["messages"], ensure_ascii=False)
        self.assertNotIn("较早观察的长篇文字", text)
        self.assertIn("已登记配置证据", text)
        self.assertIn('"stale": true', prepared["messages"][2]["content"])
        self.assertIn(goal, prepared["messages"][-1]["content"])
        self.assertIn("用户明确要求不调用工具时直接答复", prepared["messages"][-1]["content"])
        self.assertNotIn(state["next_goal"], prepared["messages"][-1]["content"])
        self.assertNotIn("继续安排下一组工具调用", prepared["messages"][-1]["content"])
        self.assertIn("original_run", prepared["messages"][1]["content"])
        self.assertEqual(state["verification"], {"passed": False, "stale": True})
        self.assertEqual(len(history), 2)
        self.assertEqual(self.usage.summary()["model_calls"], 0)

    async def test_observe_budget_preserves_tool_pair_and_full_evidence(self):
        self.settings.context_input_budget = 3000
        content = "当前工具读取的真实文件文字及配置版本\n" * 5000
        source = self.root / "large-evidence.txt"
        source.write_text(content, encoding="utf-8")
        tool_result = {"operation_id": "TOOL-LARGE", "call_id": "large-call", "name": "read_file", "status": "success", "data": {"source": str(source), "content": source.read_text(encoding="utf-8")}}
        history = [
            {"role": "assistant", "tool_calls": [{"id": "large-call", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": str(source)})}}]},
            {"role": "tool", "tool_call_id": "large-call", "content": json.dumps(tool_result, ensure_ascii=False)},
        ]
        task = {"goal": "保留原始目标", "acceptance": {"required_metrics": ["accuracy"]}}
        prepared = await self.context.prepare(task, {"verification": {"passed": False}}, history, [], role="observe")
        self.assertLessEqual(prepared["estimated_tokens"], 3000)
        current = prepared["messages"][-3:-1]
        complete_interactions(current)
        excerpt = json.loads(current[-1]["content"])
        saved = self.store.read(excerpt["full_output_ref"])
        self.assertEqual(saved, tool_result)
        self.assertEqual(json.loads(history[-1]["content"]), tool_result)
        self.assertIn("accuracy", prepared["messages"][1]["content"])
        self.assertEqual(self.usage.summary()["model_calls"], 0)

    async def test_context_preparing_file_io_keeps_event_loop_responsive(self):
        (self.root / "README.md").write_text("# 当前真实示例说明\n", encoding="utf-8")
        readme_path = self.store.path(self.context.session_path + "/context/readme.json")
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.store.path(self.context.provider.usage.context_path)) + ".lock")
        lock.acquire()
        pending = asyncio.create_task(self.context.prepare({"code_root": str(self.root), "readme_path": "README.md", "goal": "保持原始目标"}, {"verification": {"passed": False}}, [], []))
        try:
            async with asyncio.timeout(10):
                while not readme_path.exists():
                    await asyncio.sleep(0.02)
            await asyncio.sleep(0.1)
            self.assertFalse(pending.done())
            pending.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(pending.done())
        finally:
            lock.release()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertEqual(self.store.read(readme_path.relative_to(self.store.root).as_posix())["content"], "# 当前真实示例说明\n")
        self.assertTrue(any(event["event_type"] == "context.preparing" for event in self.store.read_events("events.jsonl")))

    async def test_threshold_without_old_history_preserves_valid_request(self):
        self.settings.context_threshold = 0.01
        prepared = await self.context.prepare({"task_id": "TASK-UNIT"}, {}, [], [])
        self.assertGreater(prepared["context_fraction"], self.settings.context_threshold)
        self.assertEqual(self.usage.summary()["model_calls"], 0)

    async def test_context_limit_recovery_without_old_history_keeps_valid_request(self):
        prepared = await self.context.prepare({"task_id": "TASK-UNIT"}, {}, [], [], force_compact=True)
        self.assertLessEqual(prepared["estimated_tokens"], self.context._input_budget())
        with self.assertRaisesRegex(BudgetExceededError, "不足以压缩"):
            await self.context.compact({"task_id": "TASK-UNIT"}, {}, [], [], manual=False)
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        skipped = [event for event in self.store.read_events("events.jsonl") if event["event_type"] == "context.compaction_skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["payload"]["trigger"], "context_limit_recovery")

    async def test_manual_compaction_without_old_history_returns_visible_skip(self):
        task = {"task_id": "TASK-UNIT", "goal": "保留原任务和验收条件"}
        state = {"status": "pending", "verification": {"passed": False}}
        original = copy.deepcopy((task, state))
        result = await self.context.compact(task, state, [], [], manual=True)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "no_older_history")
        self.assertEqual((task, state), original)
        events = self.store.read_events("events.jsonl")
        skipped = [event for event in events if event["event_type"] == "context.compaction_skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["payload"], result)
        self.assertFalse(any(event["event_type"] == "context.compacted" for event in events))
        self.assertIsNone(self.store.read(self.context.session_path + "/context/latest.json"))
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertEqual(self.usage.summary()["compactions"], 0)
        self.assertIsNone(self.provider.client)

    async def test_checkpoint_restores_history_and_reinjects_formal_state(self):
        historical = [{"role": "user", "content": "历史任务"}, {"role": "assistant", "content": "历史调查内容"}]
        checkpoint = {
            "checkpoint_id": "CP-UNIT",
            "summarized_message_count": 2,
            "history_fingerprint": message_fingerprint(historical),
            "summary_text": "历史摘要认为已经完成，当前状态需要重新检查",
        }
        self.store.write("sessions/SESSION-UNIT/context/CP-UNIT/checkpoint.json", checkpoint)
        self.store.write("sessions/SESSION-UNIT/context/latest.json", {"path": "sessions/SESSION-UNIT/context/CP-UNIT/checkpoint.json"})
        resumed_store = LocalStore(self.root)
        resumed_usage = UsageTracker(resumed_store, "sessions/SESSION-UNIT", self.events)
        resumed_provider = Provider(self.profile, self.events, resumed_usage)
        resumed_context = ContextManager(self.settings, resumed_store, self.events, resumed_provider)
        state = {"verification": {"passed": False, "reason": "源码已变化"}}
        history = historical + [{"role": "user", "content": "继续调查"}]
        prepared = await resumed_context.prepare({"task_id": "TASK-UNIT", "acceptance": {"required_checks": ["current_run"]}}, state, history, [])
        self.assertEqual(prepared["checkpoint_id"], "CP-UNIT")
        self.assertIn("源码已变化", prepared["messages"][2]["content"])
        self.assertFalse(any(message.get("content") == "历史调查内容" for message in prepared["messages"]))
        self.assertEqual(prepared["messages"][-2]["content"], "继续调查")
        self.assertIn("新一轮 Plan", prepared["messages"][-1]["content"])
        self.assertEqual(resumed_usage.summary()["model_calls"], 0)

    async def test_changed_history_invalidates_checkpoint(self):
        historical = [{"role": "user", "content": "原内容"}]
        self.store.write("sessions/SESSION-UNIT/context/checkpoint.json", {"checkpoint_id": "CP-UNIT", "summarized_message_count": 1, "history_fingerprint": message_fingerprint(historical), "summary_text": "旧摘要"})
        self.store.write("sessions/SESSION-UNIT/context/latest.json", {"path": "sessions/SESSION-UNIT/context/checkpoint.json"})
        prepared = await self.context.prepare({"task_id": "TASK-UNIT"}, {}, [{"role": "user", "content": "内容已更改"}], [])
        self.assertIsNone(prepared["checkpoint_id"])
        self.assertEqual(prepared["messages"][-2]["content"], "内容已更改")

    def test_accounting_cache_cost_and_restart(self):
        self.usage.record({"status": "completed", "purpose": "plan", "input_tokens": 1000, "output_tokens": 1000, "cache_read_tokens": 200, "prices": {"input_per_million": 2, "output_per_million": 4, "cache_per_million": 1}})
        self.usage.record_tool("success", 0.1)
        summary = UsageTracker(LocalStore(self.root), "sessions/SESSION-UNIT", self.events).summary()
        self.assertEqual(summary["input_tokens"], 1000)
        self.assertEqual(summary["cache_read_tokens"], 200)
        self.assertEqual(summary["output_tokens"], 1000)
        self.assertAlmostEqual(summary["cost"], 0.0058)
        self.assertTrue(summary["estimated_cost"])
        self.assertEqual(summary["tool_calls"], 1)
        self.assertEqual(summary["model_calls"], 1)

    def test_unknown_usage_is_not_zero_and_estimate_is_separate(self):
        self.usage.record({"status": "failed", "purpose": "plan", "attempt": 1, "estimated_input_tokens": 100, "estimated_output_tokens": 20})
        self.usage.record({"status": "completed", "purpose": "compaction", "attempt": 2, "input_tokens": 50, "output_tokens": 10})
        self.usage.record_compaction("CP-UNIT")
        summary = self.usage.summary()
        self.assertIsNone(summary["input_tokens"])
        self.assertIsNone(summary["output_tokens"])
        self.assertIsNone(summary["cache_read_tokens"])
        self.assertIsNone(summary["cost"])
        self.assertEqual(summary["input_tokens_known"], 50)
        self.assertEqual(summary["effective_input_tokens"], 150)
        self.assertEqual(summary["unknown_usage_calls"], 1)
        self.assertEqual(summary["model_failures"], 1)
        self.assertEqual(summary["model_retries"], 1)
        self.assertEqual(summary["compactions"], 1)
        self.assertEqual(summary["compaction_tokens"], 60)

    def test_cumulative_threshold_checks_preserve_usage(self):
        self.settings.max_input_tokens = 100
        self.usage.record({"status": "failed", "estimated_input_tokens": 100})
        before = self.usage.summary()
        self.assertEqual(self.usage.check_budget(self.settings, estimated_input=1), before)
        self.settings.max_input_tokens = 1000
        self.settings.max_tool_calls = 1
        self.usage.record_tool("denied")
        before = self.usage.summary()
        self.assertEqual(self.usage.check_budget(self.settings, reserve_tool_calls=1), before)

    def test_context_limit_retry_is_counted_without_transport_retry(self):
        self.usage.record({"status": "failed", "purpose": "plan", "attempt": 1, "estimated_input_tokens": 100})
        self.usage.record({"status": "completed", "purpose": "plan", "attempt": 1, "retry_reason": "context_limit", "input_tokens": 50, "output_tokens": 10})
        summary = self.usage.summary()
        self.assertEqual(summary["model_calls"], 2)
        self.assertEqual(summary["model_retries"], 1)
        self.assertEqual(summary["model_failures"], 1)
        self.assertEqual(summary["last_request"]["retry_reason"], "context_limit")

    def test_task_usage_and_budget_keep_session_totals(self):
        self.events.bind(task_id="TASK-FIRST")
        self.usage.record({"status": "completed", "input_tokens": 90, "output_tokens": 10})
        self.usage.record_tool("success")
        self.settings.max_model_calls = 1
        self.assertEqual(self.usage.check_budget(self.settings, reserve_model_calls=1), self.usage.summary())
        self.events.bind(task_id="TASK-SECOND")
        self.usage.check_budget(self.settings, estimated_input=20, output_reserve=10)
        self.usage.record({"status": "failed", "input_tokens": None, "output_tokens": 5, "estimated_input_tokens": 20, "task_id": "FORGED-ID"})
        summary = self.usage.summary()
        self.assertEqual(summary["current_task_id"], "TASK-SECOND")
        self.assertEqual(summary["model_calls"], 2)
        self.assertEqual(summary["tool_calls"], 1)
        self.assertEqual(summary["task_model_calls"], 1)
        self.assertEqual(summary["task_tool_calls"], 0)
        self.assertIsNone(summary["task_input_tokens"])
        self.assertEqual(summary["task_output_tokens"], 5)
        self.assertEqual(summary["task_effective_input_tokens"], 20)
        self.assertEqual(self.store.read_events(self.usage.path)[-1]["task_id"], "TASK-SECOND")
        self.events.bind(task_id=None)
        saved = self.usage.record({"status": "completed", "input_tokens": 3, "output_tokens": 2})
        self.assertIsNone(saved["task_id"])
        self.assertIsNone(self.usage.summary()["current_task_id"])

    def test_cumulative_threshold_reservations_do_not_change_actual_totals(self):
        self.events.bind(task_id="TASK-BOUNDARY")
        self.settings.max_model_calls = 1
        self.settings.max_tool_calls = 100
        self.settings.max_input_tokens = 100
        self.settings.max_output_tokens = 50
        self.usage.check_budget(self.settings, estimated_input=100, output_reserve=50, reserve_model_calls=1)
        self.usage.record({"status": "completed", "input_tokens": 100, "output_tokens": 50})
        for _ in range(99):
            self.usage.record_tool("success")
        self.usage.check_budget(self.settings, reserve_tool_calls=1)
        self.usage.record_tool("success")
        self.usage.check_budget(self.settings)
        before = self.usage.summary()
        for reserved in (
            {"reserve_model_calls": 1},
            {"reserve_tool_calls": 1},
            {"estimated_input": 1},
            {"output_reserve": 1},
        ):
            with self.subTest(reserved=reserved):
                self.assertEqual(self.usage.check_budget(self.settings, **reserved), before)
        self.usage.record_tool("denied")
        self.assertEqual(self.usage.check_budget(self.settings)["task_tool_calls"], 101)

    def test_request_preflight_remains_statistics_after_call_threshold(self):
        self.settings.max_model_calls = 1
        self.usage.record({"status": "completed", "input_tokens": 10, "output_tokens": 2})
        before = self.store.path(self.usage.path).read_bytes()
        totals = self.usage.check_budget(self.settings, estimated_input=100, output_reserve=2048, reserve_model_calls=1)
        self.assertEqual(totals["task_model_calls"], 1)
        self.assertEqual(totals["task_effective_input_tokens"], 10)
        self.assertEqual(self.store.path(self.usage.path).read_bytes(), before)
        self.assertIsNone(self.provider.client)
        self.assertEqual(self.usage.summary()["model_calls"], 1)

    def test_compaction_output_limit_and_summary_validation(self):
        self.profile.output_tokens = 8192
        self.assertEqual(self.provider.output_limit("plan"), 8192)
        self.assertEqual(self.provider.output_limit("compaction"), 2048)
        self.profile.output_tokens = 1024
        self.assertEqual(self.provider.output_limit("compaction"), 1024)
        summary = {"content": "已确认路径错误，引用 run-001/stderr.log，下一步检查 YAML", "tool_calls": [], "finish_reason": "stop"}
        self.assertEqual(ContextManager._summary(summary), summary["content"])
        for invalid, message in (
            ({**summary, "tool_calls": [{"id": "unexpected-call"}]}, "工具调用"),
            ({**summary, "finish_reason": "length"}, "未完整"),
            ({**summary, "content": " "}, "没有返回摘要"),
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, message):
                ContextManager._summary(invalid)

    async def test_observe_rejects_tools_before_model_access(self):
        with self.assertRaisesRegex(ValueError, "不允许开放工具"):
            await self.provider.complete([], [{"type": "function"}], purpose="observe")
        self.assertIsNone(self.provider.client)
        self.assertEqual(self.usage.summary()["model_calls"], 0)

    async def test_provider_rejects_invalid_attempt_limit_before_model_access(self):
        for value in (0, -1, True, 1.5, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "正整数"):
                await self.provider.complete([{"role": "user", "content": "检查请求次数参数"}], max_attempts=value)
        self.assertIsNone(self.provider.client)
        self.assertEqual(self.usage.summary()["model_calls"], 0)

    def test_context_error_classification_preserves_other_limits(self):
        self.assertTrue(Provider._context_limit_detail(400, "OutofContextError", "Total tokens of image and text exceed max message tokens."))
        self.assertTrue(Provider._context_limit_detail(400, "context_length_exceeded", ""))
        self.assertFalse(Provider._context_limit_detail(429, "QuotaExceeded", "You have exceeded the weekly usage quota."))
        self.assertFalse(Provider._context_limit_detail(429, "ModelAccountTpmRateLimitExceeded", "TPM limit of the model is exceeded."))
        self.assertFalse(Provider._context_limit_detail(413, "RequestBodyTooLarge", "The request body is larger than the limit allowed."))

    def test_invalid_usage_rejected(self):
        with self.assertRaises(ValueError):
            self.usage.record({"input_tokens": -1})
        with self.assertRaises(ValueError):
            self.usage.record({"input_tokens": 10, "cache_read_tokens": 20})
        self.assertEqual(self.usage.summary()["model_calls"], 0)

    def test_history_preserves_tool_pairs(self):
        history = []
        for index in range(5):
            history.extend([
                {"role": "user", "content": f"读取第 {index} 个配置"},
                {"role": "assistant", "tool_calls": [{"id": f"call-{index}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "benchmarkingjob.yaml"})}}]},
                {"role": "tool", "tool_call_id": f"call-{index}", "content": "配置内容"},
            ])
        older, recent = history_split(history, 3)
        self.assertEqual(len(older), 6)
        self.assertEqual(len(recent), 9)
        self.assertEqual(len(complete_interactions(older)), 4)
        self.assertEqual(recent[0]["role"], "user")
        complete_interactions(recent)
        with self.assertRaises(ValueError):
            complete_interactions(history[:-1])
        with self.assertRaises(ValueError):
            complete_interactions([history[-1]])
        repeated = history[:3] + history[:3]
        self.assertEqual(len(complete_interactions(repeated)), 4)

    def test_real_markdown_evidence_and_missing_path(self):
        reader = ReadmeEvidence(3000)
        evidence = reader.read(PROJECT_ROOT / "README.md")
        self.assertEqual(evidence["status"], "available")
        self.assertTrue(evidence["sections"])
        self.assertTrue(evidence["revision"])
        self.assertTrue(evidence["ranges"])
        missing = reader.read(self.root / "unavailable-readme.md")
        self.assertEqual(missing["status"], "missing")
        self.assertIsNone(missing["content"])

    def test_automatic_readme_respects_instance_source_range(self):
        code_root = self.root / "code"
        example = code_root / "examples/readme-check"
        example.mkdir(parents=True)
        outside = self.root / "outside-readme.md"
        outside.write_text("# 实例目录外的材料\n", encoding="utf-8")
        reader = ReadmeEvidence()
        self.assertIsNone(reader.locate({"code_root": str(code_root), "readme_path": str(outside)}, {}, PROJECT_ROOT))
        self.assertIsNone(reader.locate({"code_root": str(code_root), "readme_path": "../outside-readme.md"}, {}, PROJECT_ROOT))
        (example / "README.md").write_text("# 当前实例的使用说明\n", encoding="utf-8")
        selected = reader.locate({"code_root": str(code_root), "benchmark_path": "examples/readme-check/benchmarkingjob.yaml"}, {}, PROJECT_ROOT)
        self.assertEqual(selected, (example / "README.md").resolve())


if __name__ == "__main__":
    unittest.main()
