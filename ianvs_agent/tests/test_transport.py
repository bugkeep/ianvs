# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import difflib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.storage.local_store import LocalStore
from ianvs_agent.transport.remote import RemoteHarness
from ianvs_agent.transport.protocol import MAX_FRAME_BYTES, RemoteCallError, TransportError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def backend_path(path):
    path = Path(path).resolve()
    if os.name == "nt" and os.environ.get("IANVS_TRANSPORT_BACKEND_PREFIX"):
        if not path.drive or path.drive.startswith("\\\\"):
            raise ValueError("WSL 桥接测试需要本地盘符目录")
        return "/mnt/" + path.drive[0].lower() + "/" + path.relative_to(path.anchor).as_posix()
    return str(path)


class ObservedRemoteHarness(RemoteHarness):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received_snapshots = []

    def _apply_snapshot(self, frame):
        self.received_snapshots.append({"revision": frame["revision"], "replace": frame["replace"], "fields": set(frame["data"]), "history_delta": "history_delta" in frame})
        super()._apply_snapshot(frame)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        artifacts = PROJECT_ROOT / "workspace/ianvs-agent/test-artifacts/transport"
        artifacts.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="真实桥接-", dir=artifacts)
        self.root = Path(self.directory.name)
        self.project = self.root / "source with spaces"
        example = self.project / "examples/transport-check"
        example.mkdir(parents=True)
        self.benchmark = "examples/transport-check/benchmarkingjob.yaml"
        (self.project / self.benchmark).write_text(yaml.safe_dump({"benchmarkingjob": {"name": "stdio真实工具检查", "workspace": "workspace/output", "testenv": "examples/transport-check/testenv.yaml", "test_object": []}}), encoding="utf-8")
        (example / "testenv.yaml").write_text(yaml.safe_dump({"testenv": {"dataset": {}, "metrics": []}}), encoding="utf-8")
        (example / "README.md").write_text("真实文件、工具、审批与持久化检查，不执行模型或伪造评测结果。\n", encoding="utf-8")
        self.state_root = self.root / "backend state"
        self.config = self.root / "backend-config.yaml"
        profiles = {
            "ark": {"name": "桥接检查", "model": "ark-code-latest", "base_url": "https://ark.cn-beijing.volces.com/api/plan/v3", "api_key_env": "IANVS_TRANSPORT_UNUSED_API_KEY"},
            "glm": {"name": "GLM桥接检查", "model": "glm-5.3", "base_url": "https://ark.cn-beijing.volces.com/api/plan/v3", "api_key_env": "IANVS_TRANSPORT_UNUSED_API_KEY"},
        }
        self.config.write_text(yaml.safe_dump({"state_root": backend_path(self.state_root), "selected_profile": "ark", "approval_policy": "manual", "profiles": profiles}), encoding="utf-8")
        self.settings = AgentSettings(project_root=self.project, state_root=self.root / "frontend-unused", profiles={name: ModelProfile(**profile) for name, profile in profiles.items()})
        prefix = json.loads(os.environ["IANVS_TRANSPORT_BACKEND_PREFIX"]) if os.environ.get("IANVS_TRANSPORT_BACKEND_PREFIX") else [sys.executable, "-u", "-m", "ianvs_agent.transport.server"]
        self.command = [*prefix, "--project-root", backend_path(self.project), "--config", backend_path(self.config)]
        self.remote = ObservedRemoteHarness(self.settings, self.command, backend_label="真实stdio检查")
        self.events = []
        self.unsubscribe = self.remote.events.subscribe(self.events.append)

    async def asyncTearDown(self):
        await self.remote.close()
        self.unsubscribe()
        self.directory.cleanup()

    async def wait_event(self, event_type, call_id=None):
        async with asyncio.timeout(20):
            while True:
                for event in reversed(self.events):
                    if event["event_type"] == event_type and (call_id is None or event.get("payload", {}).get("call_id") == call_id):
                        return event
                await asyncio.sleep(0.01)

    async def start_instance(self):
        await self.remote.start()
        return await self.remote.create_instance(self.benchmark)

    @staticmethod
    def call(identity, name, arguments):
        return {"id": identity, "name": name, "arguments": arguments}

    async def patch_call(self, identity):
        relative = "examples/transport-check/README.md"
        read = await self.remote.execute_tool(self.call(identity + "-read", "read_file", {"path": relative}))
        self.assertEqual(read["status"], "success")
        path = self.state_root / "instances" / self.remote.current_instance["instance_id"] / "code" / relative
        original = path.read_text(encoding="utf-8")
        updated = original + "\n真实审批写入检查。\n"
        patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True), fromfile="a/" + relative, tofile="b/" + relative))
        return self.call(identity, "apply_patch", {"patch": patch}), path, original, updated

    async def test_instance_deletion_with_concurrent_snapshots_keeps_backend_usable(self):
        first = await self.start_instance()
        second = await self.remote.create_instance(self.benchmark)
        second_record = self.state_root / "instances" / second["instance_id"] / "instance.json"
        second_before = second_record.read_bytes()
        await self.remote.restore_instance(first["instance_id"])
        results = await asyncio.gather(self.remote.delete_instance(first["instance_id"]), *(self.remote.refresh() for _ in range(12)))
        self.assertEqual(results[0]["status"], "deleted")
        self.assertTrue(self.remote.connected)
        self.assertIsNone(self.remote.current_instance)
        self.assertIsNone(self.remote.session)
        self.assertEqual(self.remote.history(), [])
        self.assertEqual(self.remote.diff(), "")
        self.assertEqual(self.remote.verification_report(), {})
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)
        self.assertFalse((self.state_root / "instances" / first["instance_id"]).exists())
        self.assertEqual(second_record.read_bytes(), second_before)
        self.assertEqual([item["instance_id"] for item in self.remote.instances.list_instances()], [second["instance_id"]])
        deletion = await self.wait_event("instance.deleted")
        self.assertIsNone(deletion["session_id"])
        await self.remote.close()
        self.assertFalse((self.state_root / "instances" / first["instance_id"]).exists())
        await self.remote.start()
        self.assertIsNone(self.remote.current_instance)
        await self.remote.restore_instance(second["instance_id"])
        self.assertEqual(self.remote.current_instance["instance_id"], second["instance_id"])
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)

    async def test_invalid_deletion_preserves_current_session_and_backend(self):
        instance = await self.start_instance()
        before = copy.deepcopy((self.remote.current_instance, self.remote.session, self.remote.task, self.remote.state, self.remote.history()))
        with self.assertRaises(RemoteCallError):
            await self.remote.delete_instance("../outside")
        after = (self.remote.current_instance, self.remote.session, self.remote.task, self.remote.state, self.remote.history())
        self.assertEqual(after, before)
        self.assertTrue(self.remote.connected)
        self.assertTrue((self.state_root / "instances" / instance["instance_id"] / "instance.json").is_file())
        self.assertEqual((await self.remote.refresh())["status"], "refreshed")

    async def test_real_backend_browsing_instance_and_model_switch_use_no_llm(self):
        self.assertFalse(self.remote.connected)
        self.assertFalse(self.state_root.exists())
        await self.remote.start()
        self.assertTrue(self.remote.connected)
        self.assertGreater(self.remote.backend_info["pid"], 0)
        self.assertEqual(self.remote.backend_info["project_root"], backend_path(self.project))
        self.assertEqual([item["benchmark_path"] for item in self.remote.catalog.list_examples()], [self.benchmark])
        self.assertIsNone(self.remote.current_instance)
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)
        instance = await self.remote.create_instance(self.benchmark)
        self.assertEqual(instance["instance_id"], self.remote.current_instance["instance_id"])
        actual = self.state_root / "instances" / instance["instance_id"] / "code" / self.benchmark
        self.assertEqual(actual.read_bytes(), (self.project / self.benchmark).read_bytes())
        self.assertEqual(self.remote.instances.list_instances()[0]["instance_id"], instance["instance_id"])
        await self.remote.switch_model("glm")
        self.assertEqual(self.settings.selected_profile, "glm")
        self.assertEqual(self.remote.session["model_profile"], "glm")
        result = await self.remote.execute_tool(self.call("inspect-real", "inspect_configuration", {}))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["path"], self.benchmark)
        self.assertIsNone(self.remote.settings.context_input_budget)
        saved = copy.deepcopy((self.remote.task, self.remote.state, self.remote.history()))
        skipped = await self.remote.compact()
        self.assertEqual(skipped["status"], "skipped")
        self.assertEqual(skipped["reason"], "no_older_history")
        self.assertEqual((self.remote.task, self.remote.state, self.remote.history()), saved)
        await self.wait_event("context.compaction_skipped")
        self.assertEqual(sum(event["event_type"] == "context.compaction_skipped" for event in self.events), 1)
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)
        self.assertFalse(any(event["event_type"] == "llm.started" for event in self.events))
        self.assertFalse((self.root / "frontend-unused").exists())

    async def test_real_approval_approve_and_reject_while_request_pending(self):
        await self.start_instance()
        call, path, original, updated = await self.patch_call("approved-patch")
        operation = asyncio.create_task(self.remote.execute_tool(call))
        request = await self.wait_event("approval.requested", "approved-patch")
        self.assertFalse(operation.done())
        self.assertTrue(self.remote.busy)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        await self.remote.approve(request["payload"]["approval_id"], True)
        result = await asyncio.wait_for(operation, 20)
        self.assertEqual(result["status"], "success")
        self.assertEqual(path.read_text(encoding="utf-8"), updated)
        call, path, original, _ = await self.patch_call("rejected-patch")
        rejected = asyncio.create_task(self.remote.execute_tool(call))
        request = await self.wait_event("approval.requested", "rejected-patch")
        self.assertFalse(rejected.done())
        await self.remote.approve(request["payload"]["approval_id"], False)
        result = await asyncio.wait_for(rejected, 20)
        self.assertEqual(result["status"], "denied")
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        await self.wait_event("approval.decided", "rejected-patch")
        decisions = [event for event in self.events if event["event_type"] == "approval.decided"]
        self.assertEqual([event["payload"]["approved"] for event in decisions], [True, False])
        recorded = self.remote.store.read_events(self.remote.session_relative_path + "/events.jsonl")
        self.assertTrue(any(event["event_type"] == "approval.decided" for event in recorded))
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)

    async def test_cancel_pending_real_approval_keeps_backend_usable(self):
        await self.start_instance()
        call, path, original, _ = await self.patch_call("cancel-patch")
        operation = asyncio.create_task(self.remote.execute_tool(call))
        request = await self.wait_event("approval.requested", "cancel-patch")
        self.assertFalse(operation.done())
        await asyncio.wait_for(self.remote.cancel(), 20)
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 20)
        self.assertFalse(self.remote.busy)
        self.assertTrue(self.remote.connected)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        with self.assertRaises(RemoteCallError):
            await self.remote.approve(request["payload"]["approval_id"], True)
        result = await self.remote.execute_tool(self.call("state-after-cancel", "get_instance_state", {}))
        self.assertEqual(result["status"], "success")
        self.assertFalse(any(event["event_type"] == "tool.completed" and event["payload"].get("call_id") == "cancel-patch" and event["payload"].get("status") == "success" for event in self.events))
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)

    async def test_close_then_restart_restores_actual_session_and_tool_audit(self):
        instance = await self.start_instance()
        result = await self.remote.execute_tool(self.call("configuration-before-restart", "inspect_configuration", {}))
        self.assertEqual(result["status"], "success")
        session_id = self.remote.session["session_id"]
        await self.remote.switch_model("glm")
        await self.remote.close()
        self.assertFalse(self.remote.connected)
        self.unsubscribe()
        self.remote = RemoteHarness(self.settings, self.command, backend_label="真实stdio重启检查")
        self.unsubscribe = self.remote.events.subscribe(self.events.append)
        await self.remote.start()
        self.assertEqual(self.remote.current_instance["instance_id"], instance["instance_id"])
        self.assertEqual(self.remote.session["session_id"], session_id)
        self.assertEqual(self.remote.session["model_profile"], "glm")
        recorded = self.remote.store.read_events(self.remote.session_relative_path + "/events.jsonl")
        self.assertTrue(any(event["event_type"] == "tool.completed" and event["payload"].get("call_id") == "configuration-before-restart" for event in recorded))
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)
        self.assertEqual(self.remote.diff(), "")
        self.assertEqual(self.remote.verification_report(), {})

    async def test_restarted_rpc_tools_continue_after_statistics_threshold(self):
        await self.start_instance()
        for identity, name in (("resume-configuration", "inspect_configuration"), ("resume-instance", "get_instance_state")):
            result = await self.remote.execute_tool(self.call(identity, name, {}))
            self.assertEqual(result["status"], "success")
        with self.assertRaises(RemoteCallError):
            await self.remote.run_current()
        self.assertEqual(self.remote.state["status"], "failed")
        original_task = copy.deepcopy(self.remote.task)
        original_state = copy.deepcopy(self.remote.state)
        self.assertEqual(self.remote.usage_summary()["task_tool_calls"], 2)
        await self.remote.close()
        configured = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        configured.update(max_tool_calls=1, max_input_tokens=500000, max_task_seconds=0.000001)
        self.config.write_text(yaml.safe_dump(configured), encoding="utf-8")
        await self.remote.start()
        self.assertEqual(self.remote.task, original_task)
        self.assertEqual(self.remote.state, original_state)
        self.assertEqual(self.remote.settings.max_input_tokens, 500000)
        result = await self.remote.execute_tool(self.call("read-after-statistics-restart", "read_file", {"path": self.benchmark}))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["path"], self.benchmark)
        self.assertIn("benchmarkingjob:", result["data"]["content"])
        continued_task = copy.deepcopy(self.remote.task)
        archived = json.loads((self.state_root / self.remote.session_relative_path / "tasks" / f"{original_task['task_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(archived["task"], original_task)
        result = await self.remote.execute_tool(self.call("read-after-same-task-threshold", "read_file", {"path": self.benchmark}))
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.remote.task, continued_task)
        self.assertEqual(self.remote.usage_summary()["task_tool_calls"], 2)
        self.assertEqual(self.remote.usage_summary()["tool_calls"], 4)
        recorded = self.remote.store.read_events(self.remote.session_relative_path + "/events.jsonl")
        self.assertTrue(any(event["event_type"] == "tool.completed" and event["payload"].get("call_id") == "read-after-statistics-restart" for event in recorded))
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)
        self.assertFalse(any(event["event_type"] == "llm.started" for event in self.events))

    async def test_second_backend_cannot_own_same_real_state(self):
        await self.start_instance()
        competing = RemoteHarness(self.settings, self.command, backend_label="第二个真实后台")
        try:
            with self.assertRaises(TransportError):
                await competing.start()
            self.assertFalse(competing.connected)
            self.assertTrue(self.remote.connected)
            result = await self.remote.execute_tool(self.call("first-owner-still-alive", "get_instance_state", {}))
            self.assertEqual(result["status"], "success")
        finally:
            await competing.close()

    async def test_unregistered_method_and_invalid_params_are_rejected(self):
        await self.start_instance()
        for method, params in (("write_file", {"path": "outside", "content": "unauthorized"}), ("refresh", {"unexpected": True}), ("approve", {"approval_id": "missing", "approved": "yes"})):
            with self.subTest(method=method), self.assertRaises(RemoteCallError):
                await self.remote._rpc(method, params)
        self.assertTrue(self.remote.connected)
        result = await self.remote.execute_tool(self.call("registered-after-invalid", "get_instance_state", {}))
        self.assertEqual(result["status"], "success")
        self.assertFalse((self.project / "outside").exists())
        self.assertFalse((self.state_root / "outside").exists())
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)

    async def test_real_stdin_eof_cancels_approval_and_saves_session(self):
        await self.start_instance()
        call, path, original, _ = await self.patch_call("eof-patch")
        operation = asyncio.create_task(self.remote.execute_tool(call))
        await self.wait_event("approval.requested", "eof-patch")
        session_path = self.remote.session_relative_path
        process = self.remote.process
        process.stdin.close()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 30)
        self.assertEqual(await asyncio.wait_for(process.wait(), 30), 0)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        snapshot = json.loads((self.state_root / session_path / "session_snapshot.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["state"]["status"], "cancelled")
        recorded = [json.loads(line) for line in (self.state_root / session_path / "tools.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(item["call_id"] == "eof-patch" and item["status"] == "cancelled" for item in recorded))
        self.assertFalse(any(item["call_id"] == "eof-patch" and item["status"] == "success" for item in recorded))
        await self.remote.close()
        await self.remote.start()
        self.assertEqual(self.remote.session_relative_path, session_path)
        self.assertEqual(self.remote.state["status"], "cancelled")
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)

    async def test_regular_rpc_sends_incremental_snapshots_and_preserves_event_order(self):
        await self.start_instance()
        self.remote.received_snapshots.clear()
        for identity in ("incremental-first", "incremental-second"):
            result = await self.remote.execute_tool(self.call(identity, "read_file", {"path": self.benchmark}))
            self.assertEqual(result["status"], "success")
        await self.remote.switch_model("glm")
        await self.remote.refresh()
        self.assertTrue(self.remote.received_snapshots)
        self.assertFalse(any(snapshot["replace"] or "session_events" in snapshot["fields"] or "examples" in snapshot["fields"] or "history" in snapshot["fields"] for snapshot in self.remote.received_snapshots))
        self.assertFalse(self.remote.busy)
        self.assertEqual(self.settings.selected_profile, "glm")
        cached = self.remote.store.read_events(self.remote.session_relative_path + "/events.jsonl")
        stored = LocalStore(self.state_root).read_events(self.remote.session_relative_path + "/events.jsonl")
        self.assertEqual([event["event_id"] for event in cached], [event["event_id"] for event in stored])
        self.assertEqual(len(cached), len({event["event_id"] for event in cached}))
        self.assertEqual([event["sequence"] for event in cached], sorted(event["sequence"] for event in cached))
        self.assertEqual(self.remote.usage_summary()["tool_calls"], 2)

    async def test_real_backend_restores_full_large_session_and_continues_with_default_tools(self):
        instance = await self.start_instance()
        await self.remote.execute_tool(self.call("before-capacity-restart", "read_file", {"path": self.benchmark}))
        session_path = self.remote.session_relative_path
        session_id = self.remote.session["session_id"]
        await self.remote.close()
        evidence = self.root / "真实文件读取容量检查.txt"
        evidence.write_text("完整记录容量检查。🙂\n" * (MAX_FRAME_BYTES // 32 + 1), encoding="utf-8")
        content = evidence.read_text(encoding="utf-8")
        store = LocalStore(self.state_root)
        event = EventBus(store, session_id=session_id, instance_id=instance["instance_id"]).emit("transport.capacity_read", {"source": str(evidence), "content": content, "purpose": "真实文件读取的通信容量检查"})
        self.assertGreater(store.path(session_path + "/events.jsonl").stat().st_size, MAX_FRAME_BYTES)
        previous_cache = copy.deepcopy(self.remote.cache)
        interrupted_command = list(self.command)
        module_index = interrupted_command.index("ianvs_agent.transport.server")
        interrupted_command[module_index] = "ianvs_agent.tests.test_transport_interrupt"
        self.remote.command = interrupted_command
        with self.assertRaisesRegex(TransportError, "完整消息接收完成之前"):
            await self.remote.start()
        self.assertFalse(self.remote.connected)
        self.assertEqual(self.remote.cache, previous_cache)
        self.remote.command = list(self.command)
        await self.remote.start()
        self.assertTrue(self.remote.connected)
        cached = self.remote.store.read_events(session_path + "/events.jsonl")
        restored = next(item for item in cached if item["event_id"] == event["event_id"])
        self.assertEqual(restored["payload"]["content"], content)
        self.assertEqual(self.remote.session["session_id"], session_id)
        self.assertEqual(self.remote.current_instance["instance_id"], instance["instance_id"])
        self.assertEqual(self.remote.settings.tool_output_chars, 16000)
        self.remote.received_snapshots.clear()
        with self.assertRaises(RemoteCallError):
            await self.remote._rpc("refresh", {"unexpected": content})
        result = await self.remote.execute_tool(self.call("after-capacity-restart", "read_file", {"path": self.benchmark}))
        self.assertEqual(result["status"], "success")
        self.assertFalse(self.remote.busy)
        self.assertFalse(any(snapshot["replace"] or "session_events" in snapshot["fields"] for snapshot in self.remote.received_snapshots))
        cached = self.remote.store.read_events(session_path + "/events.jsonl")
        stored = store.read_events(session_path + "/events.jsonl")
        self.assertEqual([item["event_id"] for item in cached], [item["event_id"] for item in stored])
        self.assertEqual(len(cached), len({item["event_id"] for item in cached}))
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)

    async def test_publisher_failure_cancels_real_approval_and_exits_without_waiting_on_queue(self):
        interrupted = list(self.command)
        interrupted[interrupted.index("ianvs_agent.transport.server")] = "ianvs_agent.tests.test_transport_interrupt"
        self.remote.command = [*interrupted, "--failure-mode", "publisher"]
        await self.start_instance()
        call, path, original, _ = await self.patch_call("publisher-failure-patch")
        session_path = self.remote.session_relative_path
        with self.assertRaises(TransportError):
            await asyncio.wait_for(self.remote.execute_tool(call), 20)
        self.assertNotEqual(await asyncio.wait_for(self.remote.process.wait(), 5), 0)
        self.assertFalse(self.remote.connected)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        store = LocalStore(self.state_root)
        snapshot = store.read(session_path + "/session_snapshot.json")
        self.assertEqual(snapshot["state"]["status"], "cancelled")
        recorded = store.read_events(session_path + "/tools.jsonl")
        self.assertTrue(any(item["call_id"] == call["id"] and item["status"] == "cancelled" for item in recorded))
        self.remote.command = list(self.command)
        await self.remote.start()
        self.assertEqual(self.remote.state["status"], "cancelled")
        result = await self.remote.execute_tool(self.call("after-publisher-failure", "get_instance_state", {}))
        self.assertEqual(result["status"], "success")
        self.assertFalse(self.remote.busy)
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)

    async def test_cancel_during_fragmented_real_request_preserves_connection(self):
        await self.start_instance()
        call, path, original, _ = await self.patch_call("large-send-cancel")
        relative = "examples/transport-check/README.md"
        updated = original + "容量检查🙂" * (MAX_FRAME_BYTES // 16 + 1) + "\n"
        call["arguments"]["patch"] = "".join(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True), fromfile="a/" + relative, tofile="b/" + relative))
        operation = asyncio.create_task(self.remote.execute_tool(call))
        async with asyncio.timeout(20):
            while not self.remote._write_lock.locked() or self.remote.process.stdin.transport.get_write_buffer_size() == 0:
                self.assertFalse(operation.done())
                await asyncio.sleep(0)
        operation.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 30)
        self.assertTrue(self.remote.connected)
        self.assertFalse(self.remote.busy)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.assertEqual(self.remote.state["status"], "cancelled")
        stored = LocalStore(self.state_root).read_events(self.remote.session_relative_path + "/tools.jsonl")
        self.assertFalse(any(item["call_id"] == call["id"] and item["status"] == "success" for item in stored))
        result = await self.remote.execute_tool(self.call("after-fragmented-send-cancel", "get_instance_state", {}))
        self.assertEqual(result["status"], "success")
        self.assertFalse(self.remote.busy)
        self.assertEqual(self.remote.usage_summary()["model_calls"], 0)

    async def test_stdin_eof_with_publisher_or_writer_failure_preserves_cancelled_session(self):
        for failure_mode in ("closing-publisher", "closing-writer"):
            with self.subTest(failure_mode=failure_mode):
                interrupted = list(self.command)
                interrupted[interrupted.index("ianvs_agent.transport.server")] = "ianvs_agent.tests.test_transport_interrupt"
                self.remote.command = [*interrupted, "--failure-mode", failure_mode]
                await self.remote.start()
                if self.remote.current_instance is None:
                    await self.remote.create_instance(self.benchmark)
                call, path, original, _ = await self.patch_call(failure_mode)
                session_path = self.remote.session_relative_path
                operation = asyncio.create_task(self.remote.execute_tool(call))
                await self.wait_event("approval.requested", call["id"])
                self.remote.process.stdin.close()
                with self.assertRaises(TransportError):
                    await asyncio.wait_for(operation, 20)
                self.assertNotEqual(await asyncio.wait_for(self.remote.process.wait(), 5), 0)
                self.assertEqual(path.read_text(encoding="utf-8"), original)
                saved = LocalStore(self.state_root).read(session_path + "/session_snapshot.json")
                self.assertEqual(saved["state"]["status"], "cancelled")
                self.remote.command = list(self.command)
                await self.remote.start()
                self.assertEqual(self.remote.state["status"], "cancelled")
                result = await self.remote.execute_tool(self.call("after-" + failure_mode, "get_instance_state", {}))
                self.assertEqual(result["status"], "success")
                self.assertFalse(self.remote.busy)
                await self.remote.close()


if __name__ == "__main__":
    unittest.main()
