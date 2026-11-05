# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import unittest
import uuid
from pathlib import Path

from ianvs_agent.agent.events import EventBus
from ianvs_agent.storage.local_store import LocalStore


class StorageEventTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2] / "workspace/ianvs-agent/test-evidence/storage"
        self.store = LocalStore(root / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8]))

    def test_failed_snapshot_preserves_previous_record(self):
        self.store.write("task.json", {"goal": "保持原任务"})
        with self.assertRaises(ValueError):
            self.store.write("task.json", {"invalid_number": float("nan")})
        self.assertEqual(self.store.read("task.json"), {"goal": "保持原任务"})

    def test_concurrent_allocation_preserves_unique_ids_after_restart(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            values = list(executor.map(lambda _: self.store.allocate_id("Run"), range(20)))
        self.assertEqual(len(set(values)), 20)
        restored = LocalStore(self.store.root)
        self.assertEqual(restored.allocate_id("Run"), "Run-0021")

    def test_event_is_saved_before_delivery_and_retains_original_payload(self):
        bus = EventBus(self.store, session_id="session-0001", instance_id="instance-0001", task_id="task-0001")
        delivered = []

        def receive(event):
            self.assertEqual(self.store.read_events("events.jsonl")[-1], event)
            session = "instances/instance-0001/sessions/session-0001/events.jsonl"
            self.assertEqual(self.store.read_events(session)[-1], event)
            delivered.append(event)

        unsubscribe = bus.subscribe(receive)
        payload = {"files": ["examples/check/config.yaml"]}
        bus.emit("patch.applied", payload)
        payload["files"].append("examples/check/external-change.yaml")
        self.assertEqual(delivered[0]["payload"]["files"], ["examples/check/config.yaml"])
        unsubscribe()
        bus.emit("session.saved", {})
        self.assertEqual(len(delivered), 1)

    def test_concurrent_event_sequences_remain_ordered(self):
        def emit(index):
            return EventBus(self.store).emit("storage.checked", {"index": index})

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            values = list(executor.map(emit, range(20)))
        self.assertEqual(len({event["event_id"] for event in values}), 20)
        saved = self.store.read_events("events.jsonl")
        self.assertEqual([event["sequence"] for event in saved], list(range(1, 21)))

    def test_record_path_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            self.store.write("../outside.json", {"changed": True})


if __name__ == "__main__":
    unittest.main()
