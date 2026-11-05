# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import difflib
import hashlib
import unittest
from pathlib import Path

from ianvs_agent.tests import test_tools


class PatchApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.case = test_tools.ToolTests("test_patch_uses_read_version_and_invalidates_verification")
        await self.case.asyncSetUp()
        self.path = "examples/tool-check/sample.txt"
        self.target = self.case.code_root / self.path

    async def asyncTearDown(self):
        await self.case.asyncTearDown()

    async def execute(self, patch, versions):
        for path, version in versions.items():
            read = await self.case.harness.tools.execute(self.case.call("read_file", {"path": path}))
            self.assertEqual(read["status"], "success")
            self.assertEqual(read["data"]["file_version"], version)
        return await self.case.harness.tools.execute(self.case.call("apply_patch", {"patch": patch}))

    def changes_root(self):
        instance = self.case.harness.current_instance["instance_id"]
        return self.case.settings.state_root / f"instances/{instance}/changes"

    async def test_read_file_reports_full_file_eof_for_all_pages_without_writing(self):
        cases = [(b"", False), (b"first\nlast", False), (b"first\nlast\n", True), (b"first\r\nlast\r\n", True), ("first\nlast\n".encode("utf-16"), True)]
        for content, expected in cases:
            with self.subTest(content=content):
                self.target.write_bytes(content)
                first = await self.case.harness.tools.execute(self.case.call("read_file", {"path": self.path, "start_line": 1, "end_line": 1}))
                self.assertEqual(first["status"], "success")
                self.assertEqual(first["data"]["ends_with_newline"], expected)
                self.assertEqual(first["data"]["file_version"], hashlib.sha256(content).hexdigest())
                if content:
                    second = await self.case.harness.tools.execute(self.case.call("read_file", {"path": self.path, "start_line": 2, "end_line": 2}))
                    self.assertEqual(second["data"]["ends_with_newline"], expected)
                    self.assertEqual(second["data"]["file_version"], first["data"]["file_version"])
                self.assertEqual(self.target.read_bytes(), content)

    async def test_unterminated_source_context_is_rejected_without_source_write(self):
        before = b"accelerate >=1.0.0"
        self.target.write_bytes(before)
        patch = f"--- a/{self.path}\n+++ b/{self.path}\n@@ -1 +1,2 @@\n accelerate >=1.0.0\n+safetensors>=0.8.0\n"
        result = await self.execute(patch, {self.path: hashlib.sha256(before).hexdigest()})
        self.assertEqual(result["status"], "error")
        self.assertIn("FAILED", result["error"])
        self.assertIn("read_file.ends_with_newline", result["error"])
        self.assertIn("\\ No newline at end of file", result["error"])
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(self.case.harness.state.get("changes"), [])
        directory = next(self.changes_root().iterdir())
        self.assertEqual((directory / "input.diff").read_bytes(), patch.encode())
        operation = self.case.harness.store.read((directory / "dry-run-operation.json").relative_to(self.case.harness.store.root).as_posix())
        self.assertEqual(operation["status"], "failed")
        self.assertEqual(operation["exit_code"], 1)
        self.assertTrue(Path(operation["log_ref"]).is_file())
        self.assertFalse((directory / "apply-operation.json").exists())
        self.assertFalse((directory / "change.json").exists())

    async def test_canonical_source_eof_append_preserves_exact_bytes_and_logs(self):
        before = b"accelerate >=1.0.0"
        self.target.write_bytes(before)
        patch = f"--- a/{self.path}\n+++ b/{self.path}\n@@ -1 +1,2 @@\n-accelerate >=1.0.0\n\\ No newline at end of file\n+accelerate >=1.0.0\n+safetensors>=0.8.0\n"
        result = await self.execute(patch, {self.path: hashlib.sha256(before).hexdigest()})
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(self.target.read_bytes(), b"accelerate >=1.0.0\nsafetensors>=0.8.0\n")
        directory = self.changes_root() / result["data"]["modification_id"]
        record = self.case.harness.store.read((directory / "change.json").relative_to(self.case.harness.store.root).as_posix())
        self.assertEqual(self.case.harness.store.path(record["input_ref"]).read_bytes(), patch.encode())
        self.assertEqual(self.case.harness.store.path(record["execution_input_ref"]).read_bytes(), patch.encode())
        for name in ("dry_run_ref", "apply_ref"):
            operation = self.case.harness.store.read(record[name])
            self.assertEqual(operation["status"], "completed")
            self.assertEqual(operation["exit_code"], 0)
            self.assertTrue(operation["command_executed"])
            self.assertTrue(operation["completed_at"])
            self.assertTrue(Path(operation["log_ref"]).is_file())
            self.assertTrue(operation["isolation"]["private_home"])
            self.assertFalse(operation["isolation"]["network"])
        self.assertEqual(self.case.harness.usage.summary()["model_calls"], 0)

    async def test_canonical_target_eof_remains_without_newline(self):
        before = self.target.read_bytes()
        patch = f"--- a/{self.path}\n+++ b/{self.path}\n@@ -1 +1 @@\n-value=before\n+value=after\n\\ No newline at end of file\n"
        result = await self.execute(patch, {self.path: hashlib.sha256(before).hexdigest()})
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(self.target.read_bytes(), b"value=after")

    async def test_lf_patch_preserves_uniform_crlf_source_and_eof(self):
        for terminated in (True, False):
            with self.subTest(terminated=terminated):
                before = b"first\r\nvalue=before" + (b"\r\n" if terminated else b"")
                self.target.write_bytes(before)
                marker = "" if terminated else "\\ No newline at end of file\n"
                patch = f"--- a/{self.path}\n+++ b/{self.path}\n@@ -1,2 +1,2 @@\n first\n-value=before\n{marker}+value=after\n{marker}"
                result = await self.execute(patch, {self.path: hashlib.sha256(before).hexdigest()})
                self.assertEqual(result["status"], "success", result.get("error"))
                expected = b"first\r\nvalue=after" + (b"\r\n" if terminated else b"")
                self.assertEqual(self.target.read_bytes(), expected)
                record = self.case.harness.store.read_events(self.case.harness.session_relative_path + "/tools.jsonl")[-1]
                self.assertEqual(record["approval_context"]["expected_versions"][self.path], hashlib.sha256(before).hexdigest())
                directory = self.changes_root() / result["data"]["modification_id"]
                self.assertEqual((directory / "input.diff").read_bytes(), patch.encode("utf-8"))
                self.assertEqual(self.case.harness.usage.summary()["model_calls"], 0)

    async def test_crlf_diff_preserves_raw_input_and_records_lf_execution_copy(self):
        before = b"first\r\nvalue=before\r\n"
        expected = b"first\r\nvalue=after\r\n"
        self.target.write_bytes(before)
        patch = "".join(difflib.unified_diff(before.decode().splitlines(keepends=True), expected.decode().splitlines(keepends=True), fromfile="a/" + self.path, tofile="b/" + self.path))
        result = await self.execute(patch, {self.path: hashlib.sha256(before).hexdigest()})
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(self.target.read_bytes(), expected)
        tool = self.case.harness.store.read_events(self.case.harness.session_relative_path + "/tools.jsonl")[-1]
        self.assertEqual(tool["approval_context"]["expected_versions"][self.path], hashlib.sha256(before).hexdigest())
        directory = self.changes_root() / result["data"]["modification_id"]
        record = self.case.harness.store.read((directory / "change.json").relative_to(self.case.harness.store.root).as_posix())
        original = self.case.harness.store.path(record["input_ref"])
        self.assertEqual(original.read_bytes(), patch.encode("utf-8"))
        with original.open(encoding="utf-8", newline=None) as source:
            normalized = source.read().encode("utf-8")
        execution = self.case.harness.store.path(record["execution_input_ref"])
        self.assertEqual(execution.read_bytes(), normalized)
        for reference in (record["dry_run_ref"], record["apply_ref"]):
            operation = self.case.harness.store.read(reference)
            self.assertEqual(operation["input_ref"], record["input_ref"])
            self.assertEqual(operation["execution_input_ref"], record["execution_input_ref"])
            self.assertEqual(operation["command"][operation["command"].index("--input") + 1], str(execution))
        self.assertEqual(self.case.harness.usage.summary()["model_calls"], 0)

    async def test_multifile_failure_does_not_commit_first_file(self):
        second_path = "examples/tool-check/second.txt"
        second = self.case.code_root / second_path
        second.write_bytes(b"accelerate >=1.0.0")
        before = self.target.read_bytes()
        patch = self.case.patch_for(self.path) + f"--- a/{second_path}\n+++ b/{second_path}\n@@ -1 +1,2 @@\n accelerate >=1.0.0\n+safetensors>=0.8.0\n"
        versions = {self.path: hashlib.sha256(before).hexdigest(), second_path: hashlib.sha256(second.read_bytes()).hexdigest()}
        result = await self.execute(patch, versions)
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(second.read_bytes(), b"accelerate >=1.0.0")
        self.assertEqual(self.case.harness.state.get("changes"), [])
        self.assertFalse(list(self.changes_root().glob("*/change.json")))

    async def test_external_line_ending_change_during_gnu_patch_is_not_overwritten(self):
        before = self.target.read_bytes()
        external = before.replace(b"\n", b"\r\n")
        operation = asyncio.create_task(self.execute(self.case.patch_for(self.path), {self.path: hashlib.sha256(before).hexdigest()}))
        try:
            for _ in range(1000):
                if list(self.changes_root().glob("*/input.diff")):
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(list(self.changes_root().glob("*/input.diff")))
            self.assertFalse(operation.done(), "需要在真实 GNU patch 执行期间写入外部变更")
            self.target.write_bytes(external)
            result = await operation
            self.assertEqual(result["status"], "error")
            self.assertIn("应用补丁期间文件已变化", result["error"])
            self.assertEqual(self.target.read_bytes(), external)
            self.assertEqual(self.case.harness.state.get("changes"), [])
            self.assertFalse(list(self.changes_root().glob("*/change.json")))
            applied = next(self.changes_root().glob("*/apply-operation.json"))
            record = self.case.harness.store.read(applied.relative_to(self.case.harness.store.root).as_posix())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["exit_code"], 0)
        finally:
            if not operation.done():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
