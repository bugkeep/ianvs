# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


class LocalStore:
    _JOURNAL_CACHE_LIMIT = 128

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._journal_cursors = OrderedDict()
        self._journal_cursor_lock = threading.RLock()

    def path(self, relative_path):
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("记录路径超出允许的存储目录")
        return path

    def read(self, relative_path, default=None):
        path = self.path(relative_path)
        if not path.exists():
            return default
        with FileLock(str(path) + ".lock"):
            return json.loads(path.read_text(encoding="utf-8"))

    def write(self, relative_path, data):
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            self._write_atomic(path, data)
        return data

    @staticmethod
    def _write_atomic(path, data):
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def append(self, relative_path, data):
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n"
        with FileLock(str(path) + ".lock"):
            separator, _ = self._append_cursor(path)
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                if separator:
                    stream.write("\n")
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            self._remember_append(path, data)
        return data

    @staticmethod
    def _journal_signature(path):
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return None
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def _append_cursor(self, path, *, sequenced=False):
        if os.name != "nt":
            signature = self._journal_signature(path)
            with self._journal_cursor_lock:
                cursor = self._journal_cursors.get(path)
                if (
                    cursor is not None
                    and cursor["signature"] == signature
                    and (not sequenced or cursor["sequence_available"])
                ):
                    self._journal_cursors.move_to_end(path)
                    return False, cursor["last_sequence"]
        # 首次访问或文件变化时仍检查全部原文，保持损坏检测和尾部恢复。
        records, separator = self._journal_records(path)
        sequence = records[-1]["sequence"] if sequenced and records else 0
        return separator, sequence

    def _remember_append(self, path, record):
        # Windows 的 ctime 表示创建时间，无法识别保留 mtime 的同长度修改。
        if os.name == "nt":
            return
        sequence = record.get("sequence") if isinstance(record, dict) else None
        available = isinstance(sequence, (int, float))
        cursor = {
            "signature": self._journal_signature(path),
            "last_sequence": sequence if available else None,
            "sequence_available": available,
        }
        # 仅保留有限数量的文件元信息，不保存日志正文。
        with self._journal_cursor_lock:
            self._journal_cursors[path] = cursor
            self._journal_cursors.move_to_end(path)
            while len(self._journal_cursors) > self._JOURNAL_CACHE_LIMIT:
                self._journal_cursors.popitem(last=False)

    def read_events(self, relative_path):
        path = self.path(relative_path)
        if not path.exists():
            return []
        with FileLock(str(path) + ".lock"):
            records, _ = self._journal_records(path)
            return records

    @staticmethod
    def _incomplete_json_tail(error, text):
        if error.msg.startswith("Unterminated string") or error.pos >= len(text.rstrip()):
            return True
        remaining = text[error.pos:].rstrip()
        if error.msg == "Expecting value" and remaining in {"-", "t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul"}:
            return True
        if error.msg == "Expecting ',' delimiter" and remaining in {".", "e", "e+", "e-", "E", "E+", "E-"}:
            return True
        return error.msg.startswith("Invalid \\uXXXX escape") and re.fullmatch(r"\\?u[0-9a-fA-F]{0,3}", remaining) is not None

    def _journal_records(self, path):
        if not path.exists():
            return [], False
        original = path.read_bytes()
        segments = original.split(b"\n")
        complete_count = len(segments) - 1
        records = []
        for segment in segments[:complete_count]:
            text = segment.decode("utf-8")
            if text.strip():
                records.append(json.loads(text))
        tail = segments[-1]
        if not tail:
            return records, False
        reason = None
        try:
            text = tail.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason != "unexpected end of data" or error.end != len(tail):
                raise
            decoded_prefix = tail[:error.start].decode("utf-8")
            try:
                json.loads(decoded_prefix)
            except json.JSONDecodeError as json_error:
                if not self._incomplete_json_tail(json_error, decoded_prefix):
                    raise error
            else:
                raise error
            reason = "incomplete_utf8_tail"
        else:
            if text.strip():
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError as error:
                    if not self._incomplete_json_tail(error, text):
                        raise
                    reason = "incomplete_json_tail"
        if reason is None:
            return records, True
        prefix = original[:len(original) - len(tail)]
        recovery_id = "RECOVERY-" + uuid.uuid4().hex
        reference = "journal-recovery/" + recovery_id
        directory = self.path(reference)
        directory.mkdir(parents=True, exist_ok=False)
        original_path = directory / "original.bin"
        self._write_bytes_atomic(original_path, original)
        metadata = {
            "schema_version": 1,
            "recovery_id": recovery_id,
            "source_ref": path.relative_to(self.root).as_posix(),
            "original_ref": reference + "/original.bin",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "original_bytes": len(original),
            "retained_bytes": len(prefix),
            "discarded_tail_bytes": len(tail),
            "retained_records": len(records),
            "status": "prepared",
        }
        metadata_path = directory / "recovery.json"
        self._write_atomic(metadata_path, metadata)
        self._write_bytes_atomic(path, prefix)
        metadata["status"] = "recovered"
        self._write_atomic(metadata_path, metadata)
        return records, False

    @staticmethod
    def _write_bytes_atomic(path, data):
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def allocate_id(self, prefix):
        path = self.path("counters.json")
        with FileLock(str(path) + ".lock"):
            values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            values[prefix] = values.get(prefix, 0) + 1
            self._write_atomic(path, values)
        return f"{prefix}-{values[prefix]:04d}"

    def append_sequenced_event(self, relative_path, event):
        path = self.path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            separator, sequence = self._append_cursor(path, sequenced=True)
            event["sequence"] = sequence + 1
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                if separator:
                    stream.write("\n")
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._remember_append(path, event)
        return event
