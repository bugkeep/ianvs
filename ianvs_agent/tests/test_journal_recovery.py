# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
import uuid
from pathlib import Path

from ianvs_agent.storage.local_store import LocalStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class JournalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/journal" / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        self.store = LocalStore(self.root)
        self.first = {"sequence": 1, "event_type": "real_file_check", "payload": {"value": "完整中文记录"}}
        self.prefix = (json.dumps(self.first, ensure_ascii=False) + "\n").encode("utf-8")

    def journal(self, original):
        path = self.store.path("events.jsonl")
        path.write_bytes(original)
        return path

    def recovery(self, original, reason):
        files = list(self.root.glob("journal-recovery/*/recovery.json"))
        self.assertEqual(len(files), 1)
        metadata = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "recovered")
        self.assertEqual(metadata["reason"], reason)
        self.assertEqual(metadata["retained_records"], 1)
        self.assertEqual(metadata["source_ref"], "events.jsonl")
        self.assertEqual(self.store.path(metadata["original_ref"]).read_bytes(), original)
        self.assertEqual(metadata["retained_bytes"], len(self.prefix))
        self.assertGreater(metadata["discarded_tail_bytes"], 0)

    def test_read_recovers_incomplete_json_tail_and_keeps_original(self):
        original = self.prefix + b'{"sequence":2,"payload":'
        path = self.journal(original)
        self.assertEqual(self.store.read_events("events.jsonl"), [self.first])
        self.assertEqual(path.read_bytes(), self.prefix)
        self.recovery(original, "incomplete_json_tail")
        self.assertEqual(LocalStore(self.root).read_events("events.jsonl"), [self.first])
        self.assertEqual(len(list(self.root.glob("journal-recovery/*/recovery.json"))), 1)

    def test_append_recovers_tail_before_writing_next_record(self):
        original = self.prefix + b'{"sequence":2,"payload":"half'
        self.journal(original)
        second = {"sequence": 2, "payload": "恢复后的真实写入"}
        self.store.append("events.jsonl", second)
        self.assertEqual(self.store.read_events("events.jsonl"), [self.first, second])
        self.recovery(original, "incomplete_json_tail")

    def test_sequenced_append_recovers_utf8_tail(self):
        original = self.prefix + b'{"sequence":2,"payload":"' + "故".encode("utf-8")[:2]
        self.journal(original)
        second = self.store.append_sequenced_event("events.jsonl", {"event_type": "after_recovery"})
        self.assertEqual(second["sequence"], 2)
        self.assertEqual(self.store.read_events("events.jsonl"), [self.first, second])
        self.recovery(original, "incomplete_utf8_tail")

    def test_valid_json_without_newline_is_preserved_for_both_appends(self):
        for mode in ("append", "sequence"):
            with self.subTest(mode=mode):
                store = LocalStore(self.root / mode)
                path = store.path("events.jsonl")
                original = json.dumps(self.first, ensure_ascii=False).encode("utf-8")
                path.write_bytes(original)
                self.assertEqual(store.read_events("events.jsonl"), [self.first])
                self.assertEqual(path.read_bytes(), original)
                second = {"sequence": 2, "event_type": "next"}
                if mode == "append":
                    store.append("events.jsonl", second)
                else:
                    store.append_sequenced_event("events.jsonl", second)
                self.assertEqual(store.read_events("events.jsonl"), [self.first, second])
                self.assertEqual(path.read_bytes().count(b"\n"), 2)
                self.assertFalse((store.root / "journal-recovery").exists())

    def test_complete_line_corruption_and_invalid_tail_fail_closed(self):
        originals = (
            self.prefix + b'{"sequence":2,"payload":}\n',
            b'{"payload":}\n' + self.prefix,
            self.prefix + b'{"sequence":2,"payload":}',
            self.prefix + b'{"sequence":2,"payload": truu',
            self.prefix + b'{"sequence":2,"payload":"\\q',
            self.prefix + b'{"sequence":2,"payload":"\\uZZZZ and \\u123',
        )
        for index, original in enumerate(originals):
            with self.subTest(index=index):
                store = LocalStore(self.root / str(index))
                path = store.path("events.jsonl")
                path.write_bytes(original)
                for action in (lambda: store.read_events("events.jsonl"), lambda: store.append("events.jsonl", self.first), lambda: store.append_sequenced_event("events.jsonl", {"event_type": "next"})):
                    with self.assertRaises(json.JSONDecodeError):
                        action()
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse((store.root / "journal-recovery").exists())

    def test_nontruncated_utf8_error_is_not_repaired(self):
        for suffix in (b'{"value":"\xff', b'{"value":"\xe4\xb8\n', b'{"value":??? "\xe4\xb8', b'{"value":true}\xe4\xb8'):
            with self.subTest(suffix=suffix):
                path = self.journal(self.prefix + suffix)
                with self.assertRaises(UnicodeDecodeError):
                    self.store.read_events("events.jsonl")
                self.assertEqual(path.read_bytes(), self.prefix + suffix)
                self.assertFalse((self.root / "journal-recovery").exists())

    def test_standard_partial_literals_and_escape_are_incomplete(self):
        tails = (b'{"value":tru', b'{"value":nul', b'{"value":-', b'{"value":1e+', b'{"value":1.', b'{"value":"\\u123')
        for index, tail in enumerate(tails):
            with self.subTest(tail=tail):
                store = LocalStore(self.root / str(index))
                path = store.path("events.jsonl")
                path.write_bytes(self.prefix + tail)
                self.assertEqual(store.read_events("events.jsonl"), [self.first])
                self.assertEqual(path.read_bytes(), self.prefix)


if __name__ == "__main__":
    unittest.main()
