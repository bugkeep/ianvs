# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import unittest
import uuid
from pathlib import Path

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.context.builder import ContextManager
from ianvs_agent.context.compactor import complete_interactions, message_fingerprint, request_history
from ianvs_agent.llm.provider import Provider
from ianvs_agent.llm.usage import UsageTracker
from ianvs_agent.storage.local_store import LocalStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RepeatedBudgetCompactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        capture = os.environ.get("IANVS_REPEATED_CONTEXT_CAPTURE_SESSION")
        if not capture:
            self.skipTest("需要真实 CP-0007 会话目录 IANVS_REPEATED_CONTEXT_CAPTURE_SESSION")
        self.source = Path(capture)
        self.originals = {}
        self.task = self._read_source("task.json")
        self.state = self._read_source("state.json")
        self.history = self._read_source("history.json")
        self.checkpoint = self._read_source("context/CP-0007/checkpoint.json")
        self._read_source("context/CP-0007/source.json")
        self.record = self._read_source("context/input-budget/" + self.task["task_id"] + ".json")
        self.usage_bytes = self._source_bytes("usage.jsonl")
        self.usage_lines = self.usage_bytes.splitlines(keepends=True)
        self.usage_records = [json.loads(line) for line in self.usage_lines]
        compaction_request = self.checkpoint["model_requests"][-1]["usage"]["request_group_id"]
        compaction_boundary = next(index for index, record in enumerate(self.usage_records) if record.get("request_group_id") == compaction_request)
        self.budget_base = sum(record.get("input_tokens") if record.get("input_tokens") is not None else record.get("estimated_input_tokens") or 0 for record in self.usage_records[:compaction_boundary + 1] if record.get("kind") == "llm" and record.get("task_id") == self.task["task_id"])
        self.assertEqual(self.checkpoint["checkpoint_id"], "CP-0007")
        self.assertEqual(self.record["status"], "completed")
        self.assertEqual(self.record["checkpoint_id"], "CP-0007")
        self.assertEqual(self.task["task_id"], "task-0006")
        self.root = PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/repeated-budget-compaction" / uuid.uuid4().hex
        self.store = LocalStore(self.root)
        self.session_path = f"instances/{self.task['instance_id']}/sessions/{self.source.name}"
        self.events = EventBus(self.store, session_id=self.source.name, instance_id=self.task["instance_id"], task_id=self.task["task_id"])
        self.usage = UsageTracker(self.store, self.session_path, self.events)
        self.profile = ModelProfile("真实预算压缩记录检查", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3")
        self.settings = AgentSettings(PROJECT_ROOT, self.root, profiles={"ark": self.profile})
        self.provider = Provider(self.profile, self.events, self.usage)
        self.context = ContextManager(self.settings, self.store, self.events, self.provider)
        checkpoint_path = self.session_path + "/context/CP-0007/checkpoint.json"
        self.store.write(checkpoint_path, self.checkpoint)
        self.store.write(self.session_path + "/context/latest.json", {"path": checkpoint_path, "checkpoint_id": "CP-0007"})
        self.store.write(self.context._input_budget_record_path(self.task), self.record)
        instance_path = self.source.parents[1] / "instance.json"
        instance_bytes = instance_path.read_bytes()
        self.originals[instance_path] = instance_bytes
        self.store.write(f"instances/{self.task['instance_id']}/instance.json", json.loads(instance_bytes))
        self._write_usage(self.usage_bytes)

    def _source_bytes(self, relative):
        path = self.source / relative
        content = path.read_bytes()
        self.originals[path] = content
        return content

    def _read_source(self, relative):
        return json.loads(self._source_bytes(relative))

    def _write_usage(self, content):
        path = self.store.path(self.usage.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def _before_request(self, request_id):
        request = self._read_source(f"requests/{request_id}.json")
        calls = {call["id"] for call in request["response"]["tool_calls"]}
        self.assertTrue(calls)
        boundary = next(index for index, message in enumerate(self.history) if {call["id"] for call in message.get("tool_calls", [])} == calls)
        history = copy.deepcopy(self.history[:boundary])
        complete_interactions(history)
        usage_boundary = next(index for index, record in enumerate(self.usage_records) if record.get("request_group_id") == request_id)
        self._write_usage(b"".join(self.usage_lines[:usage_boundary]))
        state = json.loads(request["messages"][2]["content"].split("\n", 1)[1])
        return history, state, request["tools"]

    def _assert_no_request(self, usage_before):
        self.assertEqual(self.store.path(self.usage.path).read_bytes(), usage_before)
        self.assertIsNone(self.provider.client)
        self.assertFalse(any(event["event_type"].startswith("llm.") for event in self.store.read_events("events.jsonl")))

    async def asyncTearDown(self):
        if hasattr(self, "provider"):
            await self.provider.close()

    def tearDown(self):
        for path, original in getattr(self, "originals", {}).items():
            self.assertEqual(path.read_bytes(), original, str(path))

    async def test_completed_checkpoint_with_new_real_history_can_compact_again(self):
        history, state, tools = self._before_request("LLM-0117")
        self.settings.max_input_tokens = 100000
        usage_before = self.store.path(self.usage.path).read_bytes()
        prepared = await self.context.prepare(self.task, state, history, tools)
        totals = self.usage.summary()
        self.assertEqual(totals["task_effective_input_tokens"], 419959)
        plan = self.context._input_budget_compaction_plan(self.task, state, history, tools, prepared, totals)
        self.assertEqual(plan["status"], "ready", plan)
        self.assertEqual(plan["previous"]["checkpoint_id"], "CP-0007")
        self.assertEqual(plan["input_limit"], 100000)
        self.assertEqual(plan["used_input_tokens"], 419959)
        self.assertEqual(plan["budget_base_input_tokens"], self.budget_base)
        self.assertEqual(plan["period_input_tokens"], 419959 - self.budget_base)
        self.assertGreater(len(history), self.checkpoint["summarized_message_count"])
        self.assertNotEqual(plan["history_fingerprint"], self.record["history_fingerprint"])
        view, restored = self.context._history_view(history)
        self.assertEqual(restored, self.checkpoint)
        self.assertEqual(plan["older"] + plan["recent"], view)
        complete_interactions(plan["older"])
        complete_interactions(plan["recent"])
        self.assertEqual(plan["model_calls_reserve"], 3)
        self.assertGreater(plan["post_observe_input_reserve"], 0)
        self._assert_no_request(usage_before)

    async def test_real_tool_round_estimates_observe_after_cumulative_thresholds(self):
        history, state, tools = self._before_request("LLM-0117")
        self.settings.max_input_tokens = 100000
        self.settings.max_model_calls = 1
        self.settings.max_output_tokens = 1
        usage_before = self.store.path(self.usage.path).read_bytes()
        prepared = await self.context.prepare(self.task, state, history, tools)
        totals = self.usage.summary()
        plan = self.context._input_budget_compaction_plan(self.task, state, history, tools, prepared, totals)
        self.assertEqual(plan["status"], "ready", plan)
        self.assertEqual(plan["model_calls_reserve"], 3)
        self.assertGreater(plan["post_observe_input_reserve"], 0)
        self.assertGreater(totals["task_model_calls"], self.settings.max_model_calls)
        self.assertGreater(totals["task_effective_output_tokens"], self.settings.max_output_tokens)
        self.assertEqual(self.usage.check_budget(self.settings, estimated_input=plan["compaction_input_tokens"] + plan["post_plan_input_reserve"], output_reserve=plan["output_reserve"], reserve_model_calls=3), totals)
        self._assert_no_request(usage_before)

    async def test_restored_checkpoint_does_not_repeat_the_same_history(self):
        history, state, tools = self._before_request("LLM-0113")
        self.assertEqual(message_fingerprint(request_history(history)), self.record["history_fingerprint"])
        usage_before = self.store.path(self.usage.path).read_bytes()
        prepared = await self.context.prepare(self.task, state, history, tools)
        plan = self.context._input_budget_compaction_plan(self.task, state, history, tools, prepared, self.usage.summary())
        self.assertEqual(plan["status"], "not_needed", plan)
        resumed_provider = Provider(self.profile, self.events, self.usage)
        resumed_context = ContextManager(self.settings, self.store, self.events, resumed_provider)
        restored = await resumed_context.prepare(self.task, state, history, tools)
        repeated = resumed_context._input_budget_compaction_plan(self.task, state, history, tools, restored, self.usage.summary())
        self.assertEqual(repeated, plan)
        self.assertEqual(restored["checkpoint_id"], "CP-0007")
        self.assertEqual(json.loads(restored["messages"][1]["content"].split("\n", 1)[1])["acceptance"], self.task["acceptance"])
        self.assertIsNone(resumed_provider.client)
        self._assert_no_request(usage_before)

    async def test_existing_summary_starts_another_period_without_resetting_usage(self):
        history, state, tools = self._before_request("LLM-0115")
        usage_before = self.store.path(self.usage.path).read_bytes()
        prepared = await self.context.prepare(self.task, state, history, tools)
        totals = self.usage.summary()
        self.assertEqual(totals["task_effective_input_tokens"], 393308)
        plan = self.context._input_budget_compaction_plan(self.task, state, history, tools, prepared, totals)
        self.assertEqual(plan["status"], "not_needed", plan)
        self.assertEqual(self.budget_base, 367937)
        self.assertGreater(totals["task_effective_input_tokens"], self.budget_base)
        self.assertEqual(self.usage.summary()["task_effective_input_tokens"], 393308)
        self._assert_no_request(usage_before)

    async def test_real_task_over_cumulative_threshold_keeps_preparing_and_compacting(self):
        usage_before = self.store.path(self.usage.path).read_bytes()
        self.settings.max_input_tokens = 100000
        prepared = await self.context.prepare(self.task, self.state, self.history, [])
        totals = self.usage.summary()
        self.assertEqual(totals["task_effective_input_tokens"], 476191)
        plan = self.context._input_budget_compaction_plan(self.task, self.state, self.history, [], prepared, totals)
        self.assertEqual(plan["status"], "ready", plan)
        self.assertEqual(plan["used_input_tokens"], 476191)
        self.assertEqual(plan["budget_base_input_tokens"], self.budget_base)
        self.assertEqual(plan["period_input_tokens"], 476191 - self.budget_base)
        self.assertGreater(totals["task_effective_input_tokens"], self.settings.max_input_tokens)
        self.assertEqual(self.usage.check_budget(self.settings, estimated_input=plan["compaction_input_tokens"] + plan["post_plan_input_reserve"], reserve_model_calls=2), totals)
        self._assert_no_request(usage_before)

    async def test_observe_single_request_window_keeps_complete_real_tool_evidence(self):
        request = self._read_source("requests/LLM-0114.json")
        interaction = [message for message in request["messages"] if message.get("role") in {"assistant", "tool"}]
        complete_interactions(interaction)
        state = json.loads(request["messages"][2]["content"].split("\n", 1)[1])
        prepared = await self.context.prepare(self.task, state, interaction, [], role="observe")
        self.assertEqual(prepared["tools"], [])
        minimum = self.context._bound_tool_evidence(prepared["messages"], [], 0, 1200)
        minimum_tokens = self.context.estimator.request(minimum, [])
        self.assertLess(minimum_tokens, prepared["estimated_tokens"])
        input_window = (minimum_tokens + prepared["estimated_tokens"]) // 2
        self.settings.context_input_budget = input_window
        self.settings.max_input_tokens = 1
        usage_before = self.store.path(self.usage.path).read_bytes()
        original_interaction = copy.deepcopy(interaction)
        fitted = await self.context.prepare(self.task, state, interaction, [], role="observe")
        self.assertLessEqual(fitted["estimated_tokens"], input_window)
        self.assertEqual(self.context.estimator.request(fitted["messages"], []), fitted["estimated_tokens"])
        outgoing = [message for message in fitted["messages"] if message.get("role") in {"assistant", "tool"}]
        complete_interactions(outgoing)
        self.assertEqual(outgoing[0], interaction[0])
        original_tools = {message["tool_call_id"]: json.loads(message["content"]) for message in interaction if message["role"] == "tool"}
        for message in outgoing:
            if message["role"] != "tool":
                continue
            result = json.loads(message["content"])
            expected = original_tools[message["tool_call_id"]]
            if result.get("truncated"):
                self.assertEqual(self.store.read(result["full_output_ref"]), expected)
            else:
                self.assertEqual(result, expected)
        self.assertEqual(interaction, original_interaction)
        self.assertEqual(json.loads(fitted["messages"][1]["content"].split("\n", 1)[1])["acceptance"], self.task["acceptance"])
        self._assert_no_request(usage_before)

    async def test_real_text_plan_enters_observe_with_original_acceptance(self):
        capture = os.environ.get("IANVS_TEXT_PLAN_CAPTURE_REQUEST")
        if not capture:
            self.skipTest("需要真实纯文字 Plan 请求 IANVS_TEXT_PLAN_CAPTURE_REQUEST")
        path = Path(capture)
        original = path.read_bytes()
        self.originals[path] = original
        request = json.loads(original)
        response = request["response"]
        self.assertEqual(request["purpose"], "plan")
        self.assertEqual(response["tool_calls"], [])
        self.assertTrue(response["content"].strip())
        history = self.history + [response["raw_message"]]
        original_history = copy.deepcopy(history)
        usage_before = self.store.path(self.usage.path).read_bytes()
        prepared = await self.context.prepare(self.task, self.state, history, [], role="observe", allow_text=True)
        self.assertEqual(prepared["tools"], [])
        outgoing = [message for message in prepared["messages"] if message.get("role") == "assistant"]
        self.assertEqual(outgoing, [response["raw_message"]])
        task_view = json.loads(prepared["messages"][1]["content"].split("\n", 1)[1])
        self.assertEqual(task_view["goal"], self.task["goal"])
        self.assertEqual(task_view["acceptance"], self.task["acceptance"])
        self.assertEqual(history, original_history)
        self._assert_no_request(usage_before)
