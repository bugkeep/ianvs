# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import sys
import unittest
from pathlib import Path

from ianvs_agent.config import AgentSettings
from ianvs_agent.transport.protocol import MAX_FRAME_BYTES, MessageAssembler, TransportError, decode_frame, encode_frame, encode_message
from ianvs_agent.transport.remote import RemoteHarness


class MessageProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = "中文完整记录🙂" * (MAX_FRAME_BYTES // 20 + 1)
        cls.message = {"type": "snapshot", "revision": 2, "replace": True, "data": {"history": [{"role": "user", "content": cls.content}], "session_events": [], "state": {"status": "interrupted"}}}
        cls.payloads = list(encode_message(cls.message))

    def test_large_unicode_message_is_complete_and_each_frame_is_bounded(self):
        self.assertGreater(len(self.payloads), 1)
        assembler = MessageAssembler()
        for payload in self.payloads[:-1]:
            self.assertLessEqual(len(payload), MAX_FRAME_BYTES)
            self.assertIsNone(assembler.feed(payload))
        complete = assembler.feed(self.payloads[-1])
        self.assertEqual(complete["data"], self.message["data"])
        self.assertIsNone(assembler.message_id)
        following = {"type": "ready", "backend_info": {"pid": 1}}
        self.assertEqual(assembler.feed(encode_frame(following))["backend_info"], following["backend_info"])

    def test_partial_snapshot_keeps_previous_state_and_can_restart(self):
        remote = RemoteHarness(AgentSettings(Path.cwd(), Path.cwd() / "unused-protocol-state"), [sys.executable])
        remote._apply_snapshot({"revision": 1, "replace": True, "data": {"history": [{"role": "user", "content": "已有会话"}], "session_events": [], "state": {"status": "failed"}}})
        previous = copy.deepcopy(remote.cache)
        interrupted = MessageAssembler()
        for payload in self.payloads[:-1]:
            self.assertIsNone(interrupted.feed(payload))
        self.assertEqual(remote.cache, previous)
        self.assertEqual(remote._revision, 1)
        restarted = MessageAssembler()
        for payload in self.payloads:
            complete = restarted.feed(payload)
            if complete is not None:
                remote._apply_snapshot(complete)
        self.assertEqual(remote.history(), self.message["data"]["history"])
        self.assertEqual(remote.state["status"], "interrupted")

    def test_missing_duplicate_and_interleaved_fragments_are_rejected(self):
        for next_payload in (self.payloads[0], self.payloads[2], encode_frame({"type": "ready"})):
            with self.subTest(kind=decode_frame(next_payload)["type"]):
                assembler = MessageAssembler()
                self.assertIsNone(assembler.feed(self.payloads[0]))
                with self.assertRaises(TransportError):
                    assembler.feed(next_payload)
        with self.assertRaises(TransportError):
            MessageAssembler().feed(self.payloads[1])
        different = decode_frame(self.payloads[1])
        different["message_id"] = "different-message"
        assembler = MessageAssembler()
        assembler.feed(self.payloads[0])
        with self.assertRaises(TransportError):
            assembler.feed(encode_frame(different))

    def test_large_request_response_and_event_share_the_same_protocol(self):
        for kind in ("request", "response", "event"):
            with self.subTest(kind=kind):
                assembler = MessageAssembler()
                completed = None
                for payload in encode_message({"type": kind, "content": self.content}):
                    completed = assembler.feed(payload)
                self.assertEqual(completed["type"], kind)
                self.assertEqual(completed["content"], self.content)

    def test_history_increment_validates_before_publishing_and_supports_changed_suffix(self):
        remote = RemoteHarness(AgentSettings(Path.cwd(), Path.cwd() / "unused-protocol-state"), [sys.executable])
        original = [{"role": "user", "content": "目标"}, {"role": "assistant", "content": "待继续"}]
        remote._apply_snapshot({"revision": 1, "replace": True, "data": {"history": original, "session_events": [], "busy": True}})
        with self.assertRaises(TransportError):
            remote._apply_snapshot({"revision": 2, "replace": False, "data": {"busy": False}, "history_delta": {"start": 3, "items": []}})
        self.assertTrue(remote.busy)
        self.assertEqual(remote.history(), original)
        remote._apply_snapshot({"revision": 2, "replace": False, "data": {"busy": False}, "history_delta": {"start": 1, "items": [{"role": "assistant", "content": "已继续"}]}})
        self.assertFalse(remote.busy)
        self.assertEqual(remote.history(), [original[0], {"role": "assistant", "content": "已继续"}])


if __name__ == "__main__":
    unittest.main()
