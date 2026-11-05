# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import unittest
import uuid
from pathlib import Path

from openai import AuthenticationError

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import ModelProfile
from ianvs_agent.llm.provider import Provider
from ianvs_agent.llm.usage import UsageTracker
from ianvs_agent.storage.local_store import LocalStore
from ianvs_agent.tests.provider_stream_server import CapturedStreamServer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProviderRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        source = os.environ.get("IANVS_PROVIDER_CAPTURE_REQUEST")
        if not source:
            self.skipTest("需要真实已完成模型请求 IANVS_PROVIDER_CAPTURE_REQUEST；仅重放原始分片，不生成推理结果")
        self.source = Path(source)
        self.original = self.source.read_bytes()
        self.capture = json.loads(self.original)
        if self.capture.get("status") != "completed" or not self.capture.get("raw_chunks") or not self.capture.get("response", {}).get("tool_calls"):
            raise ValueError("Provider 回归需要真实完成的流式工具调用证据")
        self.root = PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/provider-recovery" / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        self.store = LocalStore(self.root)
        self.events = EventBus(self.store, session_id="RECOVERY", instance_id="CAPTURE")
        self.usage = UsageTracker(self.store, "sessions/RECOVERY", self.events)
        self.credential_name = "IANVS_CAPTURE_TEST_KEY_" + uuid.uuid4().hex
        os.environ[self.credential_name] = "local-capture-only"
        self.server = None
        self.provider = None

    async def asyncTearDown(self):
        if self.provider is not None:
            await self.provider.close()
        if self.server is not None:
            await asyncio.to_thread(self.server.close)
        os.environ.pop(self.credential_name, None)
        self.assertEqual(self.source.read_bytes(), self.original)

    def start(self, modes, *, idle=0.25, deadline=3, retries=1):
        self.server = CapturedStreamServer(self.capture, modes)
        profile = ModelProfile("真实流记录连接恢复检查", self.capture["request"]["model"], self.server.base_url,
                               api_key_env=self.credential_name, timeout_seconds=idle,
                               request_timeout_seconds=deadline, max_retries=retries)
        self.provider = Provider(profile, self.events, self.usage)

    async def request(self, **kwargs):
        request = self.capture["request"]
        return await self.provider.complete(request["messages"], request.get("tools"), **kwargs)

    def attempts(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.root.glob("sessions/RECOVERY/requests/LLM-*/attempt-*.json"))]

    def check_recovery(self, response, *, partial):
        attempts = self.attempts()
        self.assertEqual([item["status"] for item in attempts], ["failed", "completed"])
        self.assertTrue(attempts[0]["diagnostics"]["retryable"])
        self.assertEqual(bool(attempts[0]["diagnostics"]["chunk_count"]), partial)
        self.assertEqual(response["content"], self.capture["response"]["content"])
        self.assertEqual(response["tool_calls"], self.capture["response"]["tool_calls"])
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(self.server.requests[0], self.server.requests[1])
        events = self.store.read_events("events.jsonl")
        self.assertEqual(sum(event["event_type"] == "llm.completed" for event in events), 1)
        self.assertEqual(sum(event["event_type"] == "llm.retry" for event in events), 1)
        self.assertFalse(any(event["event_type"].startswith("tool.") for event in events))
        self.assertEqual(len(self.store.read_events(self.usage.path)), 2)

    async def test_read_timeout_before_headers_retries_and_preserves_attempts(self):
        self.start(["wait_headers", "complete"])
        response = await self.request()
        self.check_recovery(response, partial=False)
        self.assertEqual(self.attempts()[0]["diagnostics"]["failure_stage"], "idle_timeout")

    async def test_partial_stream_disconnect_retries_without_delivering_partial_tools(self):
        self.start(["disconnect", "complete"])
        response = await self.request()
        self.check_recovery(response, partial=True)
        failed = self.attempts()[0]
        self.assertEqual(failed["raw_chunks"], self.capture["raw_chunks"][:self.server.prefix_count])
        self.assertEqual(failed["diagnostics"]["error_type"], "RemoteProtocolError")

    async def test_request_deadline_retries_with_separate_stream_state(self):
        self.start(["wait_body", "complete"], idle=5, deadline=0.4)
        response = await self.request()
        self.check_recovery(response, partial=True)
        self.assertEqual(self.attempts()[0]["diagnostics"]["failure_stage"], "absolute_deadline")

    async def test_retry_limit_keeps_failure_evidence(self):
        self.start(["wait_headers"])
        with self.assertRaises(TimeoutError) as raised:
            await self.request()
        self.assertTrue(raised.exception.request_failure["retryable"])
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual([item["status"] for item in self.attempts()], ["failed", "failed"])
        failures = [event for event in self.store.read_events("events.jsonl") if event["event_type"] == "llm.failed"]
        self.assertEqual([event["payload"]["will_retry"] for event in failures], [True, False])

    async def test_per_call_attempt_limit_is_preserved(self):
        self.start(["wait_headers", "complete"])
        with self.assertRaises(TimeoutError):
            await self.request(max_attempts=1)
        self.assertEqual(len(self.server.requests), 1)

    async def test_authentication_failure_is_not_retried(self):
        self.start(["unauthorized", "complete"])
        with self.assertRaises(AuthenticationError) as raised:
            await self.request()
        self.assertFalse(raised.exception.request_failure["retryable"])
        self.assertEqual(len(self.server.requests), 1)

    async def test_invalid_completed_stream_is_not_retried(self):
        self.start(["unfinished", "complete"])
        with self.assertRaisesRegex(ValueError, "未完整结束") as raised:
            await self.request()
        self.assertFalse(raised.exception.request_failure["retryable"])
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual([item["status"] for item in self.attempts()], ["failed"])

    async def test_cancellation_during_stream_is_not_retried(self):
        self.start(["wait_body", "complete"], idle=5, deadline=10)
        task = asyncio.create_task(self.request())
        try:
            self.assertTrue(await asyncio.to_thread(self.server.sent_prefix.wait, 5))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual([item["status"] for item in self.attempts()], ["cancelled"])
        self.assertFalse(self.attempts()[0]["diagnostics"]["retryable"])
