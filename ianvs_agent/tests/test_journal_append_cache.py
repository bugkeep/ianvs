# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import json
import os
import sys
import unittest
import uuid
from pathlib import Path

from ianvs_agent.storage.local_store import LocalStore


class JournalReadAudit:
    def __init__(self, path):
        self.path = str(path)
        self.reads = 0
        self.active = False
        sys.addaudithook(self.observe)

    def observe(self, event, arguments):
        if self.active and event == "open" and str(arguments[0]) == self.path and arguments[1] == "r":
            self.reads += 1

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *arguments):
        self.active = False


def append_from_process(root, count):
    store = LocalStore(root)
    return [store.append_sequenced_event("events.jsonl", {"event_type": "process.checked"})["sequence"] for _ in range(count)]


class JournalAppendCacheTests(unittest.TestCase):
    def setUp(self):
        project_root = Path(__file__).resolve().parents[2]
        artifact_root = Path(os.environ.get("IANVS_STORAGE_TEST_ROOT", project_root / "workspace/ianvs-agent/usability-20261011/storage"))
        self.root = artifact_root / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        self.store = LocalStore(self.root)
        self.path = self.store.path("events.jsonl")

    def test_unchanged_journal_uses_platform_validation_policy(self):
        self.store.append("events.jsonl", {"value": "真实文件追加检查"})
        with JournalReadAudit(self.path) as audit:
            for index in range(8):
                self.store.append("events.jsonl", {"index": index})
        self.assertEqual(audit.reads, 8 if os.name == "nt" else 0)
        self.assertEqual(len(self.store.read_events("events.jsonl")), 9)

    def test_sequence_cache_and_restart_preserve_actual_order(self):
        self.assertEqual(self.store.append_sequenced_event("events.jsonl", {"event_type": "first"})["sequence"], 1)
        with JournalReadAudit(self.path) as audit:
            self.assertEqual(self.store.append_sequenced_event("events.jsonl", {"event_type": "second"})["sequence"], 2)
        self.assertEqual(audit.reads, 1 if os.name == "nt" else 0)
        restored = LocalStore(self.root)
        with JournalReadAudit(self.path) as audit:
            self.assertEqual(restored.append_sequenced_event("events.jsonl", {"event_type": "third"})["sequence"], 3)
        self.assertEqual(audit.reads, 1)
        self.assertEqual([record["sequence"] for record in restored.read_events("events.jsonl")], [1, 2, 3])

    def test_other_store_append_requires_validation_before_next_sequence(self):
        self.store.append_sequenced_event("events.jsonl", {"event_type": "first"})
        second_store = LocalStore(self.root)
        second_store.append_sequenced_event("events.jsonl", {"event_type": "second"})
        with JournalReadAudit(self.path) as audit:
            event = self.store.append_sequenced_event("events.jsonl", {"event_type": "third"})
        self.assertEqual(audit.reads, 1)
        self.assertEqual(event["sequence"], 3)

    def test_same_size_external_corruption_remains_an_error(self):
        self.store.append_sequenced_event("events.jsonl", {"event_type": "first"})
        original = self.path.read_bytes()
        corrupted = original.replace(b'"sequence": 1', b'"sequence": }')
        self.assertEqual(len(corrupted), len(original))
        self.assertNotEqual(corrupted, original)
        self.path.write_bytes(corrupted)
        for action in (lambda: self.store.append("events.jsonl", {"value": "next"}), lambda: self.store.append_sequenced_event("events.jsonl", {"event_type": "next"}), lambda: self.store.read_events("events.jsonl")):
            with self.assertRaises(json.JSONDecodeError):
                action()
            self.assertEqual(self.path.read_bytes(), corrupted)
        self.assertFalse((self.root / "journal-recovery").exists())

    def test_replaced_file_with_preserved_size_and_mtime_is_revalidated(self):
        self.store.append_sequenced_event("events.jsonl", {"event_type": "first"})
        original = self.path.read_bytes()
        metadata = self.path.stat()
        replacement = self.root / "replacement.jsonl"
        replacement.write_bytes(original.replace(b'"sequence": 1', b'"sequence": 7'))
        os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        os.replace(replacement, self.path)
        self.assertEqual(self.path.stat().st_size, metadata.st_size)
        self.assertEqual(self.path.stat().st_mtime_ns, metadata.st_mtime_ns)
        with JournalReadAudit(self.path) as audit:
            event = self.store.append_sequenced_event("events.jsonl", {"event_type": "after_replacement"})
        self.assertEqual(audit.reads, 1)
        self.assertEqual(event["sequence"], 8)
        self.assertEqual([record["sequence"] for record in self.store.read_events("events.jsonl")], [7, 8])

    def test_same_size_inplace_corruption_with_restored_mtime_is_rejected(self):
        self.store.append_sequenced_event("events.jsonl", {"event_type": "first"})
        original = self.path.read_bytes()
        metadata = self.path.stat()
        corrupted = original.replace(b'"sequence": 1', b'"sequence": }')
        with self.path.open("r+b") as stream:
            stream.write(corrupted)
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(self.path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        updated = self.path.stat()
        self.assertEqual(updated.st_ino, metadata.st_ino)
        self.assertEqual(updated.st_size, metadata.st_size)
        self.assertEqual(updated.st_mtime_ns, metadata.st_mtime_ns)
        with JournalReadAudit(self.path) as audit:
            with self.assertRaises(json.JSONDecodeError):
                self.store.append_sequenced_event("events.jsonl", {"event_type": "after_corruption"})
        self.assertEqual(audit.reads, 1)
        self.assertEqual(self.path.read_bytes(), corrupted)
        self.assertFalse((self.root / "journal-recovery").exists())

    def test_truncated_journal_starts_again_from_its_actual_content(self):
        self.store.append_sequenced_event("events.jsonl", {"event_type": "first"})
        self.path.write_bytes(b"")
        with JournalReadAudit(self.path) as audit:
            event = self.store.append_sequenced_event("events.jsonl", {"event_type": "after_truncation"})
        self.assertEqual(audit.reads, 1)
        self.assertEqual(event["sequence"], 1)

    def test_external_incomplete_tails_keep_recovery_evidence_after_cached_append(self):
        tails = (b'{"sequence":2,"payload":"partial', b'{"sequence":2,"payload":"' + "故".encode("utf-8")[:2])
        for index, tail in enumerate(tails):
            with self.subTest(index=index):
                store = LocalStore(self.root / str(index))
                path = store.path("events.jsonl")
                first = store.append_sequenced_event("events.jsonl", {"event_type": "before_interruption"})
                prefix = path.read_bytes()
                with path.open("ab") as stream:
                    stream.write(tail)
                event = store.append_sequenced_event("events.jsonl", {"event_type": "after_recovery"})
                self.assertEqual(event["sequence"], 2)
                self.assertEqual(store.read_events("events.jsonl"), [first, event])
                receipts = list(store.root.glob("journal-recovery/*/recovery.json"))
                self.assertEqual(len(receipts), 1)
                receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
                self.assertEqual(receipt["status"], "recovered")
                self.assertEqual(store.path(receipt["original_ref"]).read_bytes(), prefix + tail)
                self.assertEqual(receipt["retained_records"], 1)

    def test_valid_external_record_without_newline_is_kept(self):
        self.store.append_sequenced_event("events.jsonl", {"event_type": "first"})
        second = {"sequence": 2, "event_type": "external"}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(second))
        event = self.store.append_sequenced_event("events.jsonl", {"event_type": "third"})
        self.assertEqual(event["sequence"], 3)
        self.assertEqual(self.store.read_events("events.jsonl")[1], second)
        self.assertEqual(self.path.read_bytes().count(b"\n"), 3)

    def test_processes_keep_sequence_order_with_independent_caches(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(append_from_process, self.root, 8) for _ in range(3)]
            sequences = [sequence for future in futures for sequence in future.result()]
        self.assertEqual(sorted(sequences), list(range(1, 25)))
        self.assertEqual([record["sequence"] for record in self.store.read_events("events.jsonl")], list(range(1, 25)))

    def test_evicted_journal_is_validated_again(self):
        self.store.append("events.jsonl", {"value": "first"})
        for index in range(140):
            self.store.append(f"journals/{index}.jsonl", {"index": index})
        with JournalReadAudit(self.path) as audit:
            self.store.append("events.jsonl", {"value": "after_eviction"})
        self.assertEqual(audit.reads, 1)
        self.assertEqual(len(self.store.read_events("events.jsonl")), 2)

    def test_sequence_requires_actual_sequence_after_ordinary_append(self):
        record = {"value": "普通工具记录"}
        self.store.append("events.jsonl", record)
        original = self.path.read_bytes()
        with self.assertRaises(KeyError):
            self.store.append_sequenced_event("events.jsonl", {"event_type": "next"})
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
