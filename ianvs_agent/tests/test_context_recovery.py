# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import os
import unittest
import uuid
from pathlib import Path

from openai import AuthenticationError

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.context.builder import ContextManager
from ianvs_agent.context.compactor import complete_interactions, request_history
from ianvs_agent.llm.provider import Provider
from ianvs_agent.llm.usage import BudgetExceededError, UsageTracker
from ianvs_agent.storage.local_store import LocalStore
from ianvs_agent.tests.provider_stream_server import CapturedStreamServer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ContextRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        source = os.environ.get("IANVS_REPEATED_CONTEXT_CAPTURE_SESSION")
        if not source:
            self.skipTest("需要真实 CP-0007 会话 IANVS_REPEATED_CONTEXT_CAPTURE_SESSION")
        self.source = Path(source)
        self.originals = {}
        self.task = self.read_source("task.json")
        self.state = self.read_source("state.json")
        self.history = self.read_source("history.json")
        self.checkpoint = self.read_source("context/CP-0007/checkpoint.json")
        self.budget_record = self.read_source("context/input-budget/task-0006.json")
        self.usage_bytes = self.source_bytes(self.source / "usage.jsonl")
        self.usage_lines = self.usage_bytes.splitlines(keepends=True)
        self.usage_records = [json.loads(line) for line in self.usage_lines]
        self.assertEqual(self.checkpoint["checkpoint_id"], "CP-0007")
        self.assertEqual(self.task["task_id"], "task-0006")
        artifact = Path(os.environ.get("IANVS_CONTEXT_RECOVERY_TEST_ROOT", PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/context-recovery"))
        self.root = artifact / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        self.store = LocalStore(self.root)
        self.relative = "instances/instance-0006/sessions/session-0005"
        self.events = EventBus(self.store, instance_id=self.task["instance_id"], session_id=self.source.name, task_id=self.task["task_id"])
        self.usage = UsageTracker(self.store, self.relative, self.events)
        self.credential_name = "IANVS_CONTEXT_CAPTURE_TEST_KEY_" + uuid.uuid4().hex
        os.environ[self.credential_name] = "local-capture-only"
        self.profile = ModelProfile("真实上下文恢复记录检查", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3", api_key_env=self.credential_name, timeout_seconds=0.25, request_timeout_seconds=3)
        self.settings = AgentSettings(PROJECT_ROOT, self.root, profiles={"ark": self.profile}, max_input_tokens=100000)
        self.provider = Provider(self.profile, self.events, self.usage)
        self.context = ContextManager(self.settings, self.store, self.events, self.provider)
        self.server = None
        checkpoint_path = self.relative + "/context/CP-0007/checkpoint.json"
        self.store.write(checkpoint_path, self.checkpoint)
        self.store.write(self.relative + "/context/latest.json", {"checkpoint_id": "CP-0007", "path": checkpoint_path})
        self.store.write(self.context._input_budget_record_path(self.task), self.budget_record)
        instance_path = self.source.parents[1] / "instance.json"
        self.store.write("instances/instance-0006/instance.json", json.loads(self.source_bytes(instance_path)))
        request = self.read_source("requests/LLM-0117.json")
        call_ids = {item["id"] for item in request["response"]["tool_calls"]}
        boundary = next(index for index, message in enumerate(self.history) if {item["id"] for item in message.get("tool_calls", [])} == call_ids)
        self.current_history = copy.deepcopy(self.history[:boundary])
        self.current_state = json.loads(request["messages"][2]["content"].split("\n", 1)[1])
        self.tools = request["tools"]
        usage_boundary = next(index for index, record in enumerate(self.usage_records) if record.get("request_group_id") == "LLM-0117")
        usage_path = self.store.path(self.usage.path)
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        usage_path.write_bytes(b"".join(self.usage_lines[:usage_boundary]))

    def source_bytes(self, path):
        content = path.read_bytes()
        self.originals[path] = content
        return content

    def read_source(self, relative):
        return json.loads(self.source_bytes(self.source / relative))

    async def asyncTearDown(self):
        if hasattr(self, "provider"):
            await self.provider.close()
        if getattr(self, "server", None) is not None:
            await asyncio.to_thread(self.server.close)
        if hasattr(self, "credential_name"):
            os.environ.pop(self.credential_name, None)
        for path, content in getattr(self, "originals", {}).items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def start(self, modes):
        capture_path = os.environ.get("IANVS_PROVIDER_CAPTURE_REQUEST")
        if not capture_path:
            self.skipTest("需要真实已完成模型请求 IANVS_PROVIDER_CAPTURE_REQUEST；仅重放原始分片")
        self.capture = json.loads(self.source_bytes(Path(capture_path)))
        self.server = CapturedStreamServer(self.capture, modes)
        self.profile.model = self.capture["request"]["model"]
        self.profile.base_url = self.server.base_url

    def attempts(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.root.glob(self.relative + "/requests/LLM-*/attempt-*.json"))]

    async def check_deferred_request(self, mode):
        self.start([mode, "complete"])
        original = copy.deepcopy((self.task, self.current_state, self.current_history))
        before = self.usage.summary()
        prepared = await self.context.prepare(self.task, self.current_state, self.current_history, self.tools)
        plan = self.context._input_budget_compaction_plan(self.task, self.current_state, self.current_history, self.tools, prepared, before)
        self.assertEqual(plan["status"], "ready")
        response = await self.context.complete(self.task, self.current_state, self.current_history, self.tools)
        self.assertEqual(response["content"], self.capture["response"]["content"])
        self.assertEqual(response["tool_calls"], self.capture["response"]["tool_calls"])
        self.assertEqual(len(self.server.requests), 2)
        self.assertNotIn("tools", self.server.requests[0])
        self.assertEqual(self.server.requests[1]["tools"], self.tools)
        attempts = self.attempts()
        self.assertEqual([(item["purpose"], item["status"]) for item in attempts], [("compaction", "failed"), ("plan", "completed")])
        self.assertTrue(attempts[0]["diagnostics"]["retryable"])
        record = self.store.read(self.context._input_budget_record_path(self.task))
        self.assertEqual(record["status"], "deferred")
        self.assertEqual(record["reason"], "retryable_request_failed")
        self.assertTrue(record["request_failure"]["retryable"])
        self.assertEqual(self.store.read(record["request_failure"]["attempt_ref"]), attempts[0])
        totals = self.usage.summary()
        self.assertEqual(totals["model_calls"], before["model_calls"] + 2)
        self.assertEqual(totals["model_failures"], before["model_failures"] + 1)
        self.assertGreater(totals["task_effective_input_tokens"], before["task_effective_input_tokens"])
        events = self.store.read_events("events.jsonl")
        self.assertEqual(sum(event["event_type"] == "context.compaction_deferred" for event in events), 1)
        self.assertFalse(any(event["event_type"].startswith("tool.") or event["event_type"] == "context.compacted" for event in events))
        repeated = self.context._input_budget_compaction_plan(self.task, self.current_state, self.current_history, self.tools, prepared, totals)
        self.assertEqual(repeated, {"status": "not_needed", "reason": "already_checked"})
        next_prepared = await self.context.prepare(self.task, self.state, self.history, self.tools)
        next_plan = self.context._input_budget_compaction_plan(self.task, self.state, self.history, self.tools, next_prepared, totals)
        self.assertEqual(next_plan["status"], "ready")
        self.assertEqual((self.task, self.current_state, self.current_history), original)
        return attempts

    async def test_periodic_summary_timeout_keeps_evidence_and_sends_prepared_plan(self):
        attempts = await self.check_deferred_request("wait_headers")
        self.assertEqual(attempts[0]["diagnostics"]["failure_stage"], "idle_timeout")
        self.assertEqual(attempts[0]["diagnostics"]["chunk_count"], 0)

    async def test_periodic_summary_disconnect_keeps_partial_record_and_sends_plan(self):
        attempts = await self.check_deferred_request("disconnect")
        self.assertEqual(attempts[0]["diagnostics"]["error_type"], "RemoteProtocolError")
        self.assertEqual(attempts[0]["raw_chunks"], self.capture["raw_chunks"][:self.server.prefix_count])

    async def test_periodic_summary_authentication_error_propagates(self):
        self.start(["unauthorized", "complete"])
        with self.assertRaises(AuthenticationError):
            await self.context.complete(self.task, self.current_state, self.current_history, self.tools)
        self.assertEqual(len(self.server.requests), 1)
        self.assertFalse(self.attempts()[0]["diagnostics"]["retryable"])
        self.assertEqual(self.store.read(self.context._input_budget_record_path(self.task))["status"], "attempted")
        self.assertFalse(any(event["event_type"] == "context.compaction_deferred" for event in self.store.read_events("events.jsonl")))

    async def test_periodic_summary_unfinished_response_propagates(self):
        self.start(["unfinished", "complete"])
        with self.assertRaisesRegex(ValueError, "未完整结束"):
            await self.context.complete(self.task, self.current_state, self.current_history, self.tools)
        self.assertEqual(len(self.server.requests), 1)
        attempt = self.attempts()[0]
        self.assertFalse(attempt["diagnostics"]["retryable"])
        self.assertEqual(attempt["raw_chunks"], self.capture["raw_chunks"][:self.server.prefix_count])
        self.assertEqual(self.store.read(self.context._input_budget_record_path(self.task))["status"], "attempted")
        self.assertFalse(any(event["event_type"] == "context.compaction_deferred" for event in self.store.read_events("events.jsonl")))

    async def test_periodic_summary_cancellation_propagates(self):
        self.start(["wait_headers", "complete"])
        self.profile.timeout_seconds = 5
        self.profile.request_timeout_seconds = 10
        task = asyncio.create_task(self.context.complete(self.task, self.current_state, self.current_history, self.tools))
        try:
            self.assertTrue(await asyncio.to_thread(self.server.received.wait, 5))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.attempts()[0]["status"], "cancelled")
        self.assertFalse(any(event["event_type"] == "context.compaction_deferred" for event in self.store.read_events("events.jsonl")))

    async def test_deferred_record_storage_error_propagates(self):
        self.start(["wait_headers", "complete"])
        self.profile.timeout_seconds = 1
        task = asyncio.create_task(self.context.complete(self.task, self.current_state, self.current_history, self.tools))
        try:
            self.assertTrue(await asyncio.to_thread(self.server.received.wait, 5))
            record_path = self.store.path(self.context._input_budget_record_path(self.task))
            record_path.unlink()
            record_path.mkdir()
            with self.assertRaises(OSError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.attempts()[0]["status"], "failed")
        self.assertFalse(any(event["event_type"] == "context.compaction_deferred" for event in self.store.read_events("events.jsonl")))

    async def test_forced_recovery_shrinks_one_real_tool_block_without_old_history(self):
        block = next(item for item in reversed(complete_interactions(request_history(self.history))) if item[0].get("tool_calls"))
        original = copy.deepcopy((self.task, self.state, block))
        normal = await self.context.prepare(self.task, self.state, block, self.tools)
        recovered = await self.context.prepare(self.task, self.state, block, self.tools, force_compact=True)
        self.assertLess(recovered["estimated_tokens"], normal["estimated_tokens"])
        self.assertLessEqual(recovered["estimated_tokens"], self.context._input_budget())
        history = [message for message in recovered["messages"] if message.get("role") in {"assistant", "tool"}]
        complete_interactions(history)
        self.assertEqual(history[0], block[0])
        result = json.loads(history[-1]["content"])
        self.assertTrue(result["truncated"])
        self.assertEqual(self.store.read(result["full_output_ref"]), json.loads(block[-1]["content"]))
        self.assertEqual(json.loads(recovered["messages"][1]["content"].split("\n", 1)[1])["acceptance"], self.task["acceptance"])
        self.assertEqual((self.task, self.state, block), original)
        self.assertIsNone(self.provider.client)

    async def test_forced_recovery_fixed_context_overflow_remains_explicit(self):
        self.settings.context_input_budget = 64
        with self.assertRaisesRegex(BudgetExceededError, "超过输入预算 64"):
            await self.context.prepare(self.task, self.state, [], self.tools, force_compact=True)
        self.assertIsNone(self.provider.client)
        self.assertFalse(any(event["event_type"] == "llm.started" for event in self.store.read_events("events.jsonl")))
