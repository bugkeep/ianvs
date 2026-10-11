# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import os
import sys
import unittest
import uuid
from pathlib import Path

import yaml
from filelock import FileLock

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.context.compactor import complete_interactions
from ianvs_agent.tools.filesystem import CodeTools


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        evidence_root = Path(os.environ.get("IANVS_FILE_TOOL_TEST_ROOT", PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/tools"))
        self.root = evidence_root / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        self.project = self.root / "project"
        fixture = self.project / "examples/tool-check"
        fixture.mkdir(parents=True, exist_ok=False)
        (self.project / "core").mkdir()
        (self.project / "core/sample.txt").write_text("value=before\n", encoding="utf-8")
        (fixture / "sample.txt").write_text("value=before\n", encoding="utf-8")
        (self.project / "README.md").write_text("# 工具检查项目\n\n仅检查真实文件、补丁与受限命令。\n", encoding="utf-8")
        (fixture / "README.md").write_text("# 工具检查 Example\n\n运行和验证不在本组文件工具检查范围。\n", encoding="utf-8")
        (fixture / "benchmarkingjob.yaml").write_text(yaml.safe_dump({"benchmarkingjob": {"name": "tool-check", "workspace": "workspace/output", "testenv": "examples/tool-check/testenv.yaml", "test_object": []}}), encoding="utf-8")
        (fixture / "testenv.yaml").write_text(yaml.safe_dump({"testenv": {"metrics": []}}), encoding="utf-8")
        self.outside = self.root / "outside.txt"
        self.outside.write_text("outside-preserved\n", encoding="utf-8")
        profile = ModelProfile("文件工具检查", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3")
        self.settings = AgentSettings(self.project, self.root / "state", python_executable=sys.executable, profiles={"ark": profile}, approval_policy="auto_scoped")
        self.harness = AgentHarness(self.settings)
        await self.harness.create_instance("examples/tool-check/benchmarkingjob.yaml")
        self.harness._ensure_task("检查文件范围、补丁版本依据与审批记录")
        self.code_root = Path(self.harness.current_instance["code_root"])

    async def asyncTearDown(self):
        await self.harness.close()

    def call(self, name, arguments, call_id=None):
        return {"id": call_id or "call-" + uuid.uuid4().hex[:8], "name": name, "arguments": arguments}

    async def read_sample(self):
        return await self.harness.tools.execute(self.call("read_file", {"path": "examples/tool-check/sample.txt"}))

    @staticmethod
    def patch_for(path, before="value=before", after="value=after"):
        return f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{before}\n+{after}\n"

    async def test_patch_uses_read_version_and_invalidates_verification(self):
        read = await self.read_sample()
        version = read["data"]["file_version"]
        result = await self.harness.tools.execute(self.call("apply_patch", {"patch": self.patch_for("examples/tool-check/sample.txt")}))
        self.assertEqual(result["status"], "success")
        self.assertEqual((self.code_root / "examples/tool-check/sample.txt").read_text(encoding="utf-8"), "value=after\n")
        self.assertEqual((self.project / "examples/tool-check/sample.txt").read_text(encoding="utf-8"), "value=before\n")
        self.assertEqual(self.harness.state["verification"]["status"], "needs_verification")
        self.assertFalse(self.harness.state["verification"]["passed"])
        modification_id = result["data"]["modification_id"]
        saved = self.harness.store.read(f"instances/{self.harness.current_instance['instance_id']}/changes/{modification_id}/change.json")
        self.assertIsNone(saved["verified_run_id"])
        self.assertIn("+value=after", saved["changes"][0]["diff"])
        record = self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")[-1]
        self.assertEqual(record["approval_context"]["expected_versions"], {"examples/tool-check/sample.txt": version})
        self.assertEqual(record["approval_context"]["read_evidence_refs"], {"examples/tool-check/sample.txt": "tool:" + read["operation_id"]})

    async def test_patch_rejects_real_version_conflict(self):
        read = await self.read_sample()
        target = self.code_root / "examples/tool-check/sample.txt"
        target.write_text("value=external-change\n", encoding="utf-8")
        result = await self.harness.tools.execute(self.call("apply_patch", {"patch": self.patch_for("examples/tool-check/sample.txt")}))
        self.assertIn(result["status"], {"error", "denied"})
        self.assertIn("文件已变化", result["error"])
        self.assertEqual(target.read_text(encoding="utf-8"), "value=external-change\n")
        self.assertEqual(self.harness.state.get("changes"), [])

    async def test_core_patch_is_rejected_with_real_file_version(self):
        path = "core/sample.txt"
        read = await self.harness.tools.execute(self.call("read_file", {"path": path}))
        result = await self.harness.tools.execute(self.call("apply_patch", {"patch": self.patch_for(path)}))
        self.assertIn(result["status"], {"error", "denied"})
        self.assertIn("无权修改运行控制", result["error"])
        self.assertEqual((self.code_root / path).read_text(encoding="utf-8"), "value=before\n")
        self.assertEqual((self.project / path).read_text(encoding="utf-8"), "value=before\n")
        self.assertEqual(self.harness.state.get("changes"), [])

    async def test_patch_and_file_read_reject_outside_paths(self):
        patch = self.patch_for("../../outside.txt", "outside-preserved", "outside-changed")
        result = await self.harness.tools.execute(self.call("apply_patch", {"patch": patch}))
        self.assertIn(result["status"], {"error", "denied"})
        self.assertEqual(self.outside.read_text(encoding="utf-8"), "outside-preserved\n")
        read = await self.harness.tools.execute(self.call("read_file", {"path": str(self.outside)}))
        self.assertIn(read["status"], {"error", "denied"})
        self.assertEqual(self.harness.state.get("changes"), [])

    @unittest.skipIf(os.name == "nt", "符号链接范围检查在 Linux 环境执行")
    async def test_symbolic_link_does_not_expand_read_range(self):
        (self.code_root / "core/outside-link.txt").symlink_to(self.outside)
        result = await self.harness.tools.execute(self.call("read_file", {"path": "core/outside-link.txt"}))
        self.assertIn(result["status"], {"error", "denied"})
        self.assertIn("超出", result["error"])

    def full_data(self, result):
        self.assertEqual(result["status"], "success", result.get("error"))
        return self.harness.store.read(result["full_output_ref"]) if result["truncated"] else result["data"]

    async def test_search_decodes_bom_text_and_reports_unreadable_files(self):
        directory = self.code_root / "examples/tool-check/encodings"
        directory.mkdir()
        text = "前置行\nneedle 真实证据\n"
        for name, encoding in (("utf8.txt", "utf-8"), ("utf8-bom.txt", "utf-8-sig"), ("utf16.txt", "utf-16"), ("utf32.txt", "utf-32")):
            (directory / name).write_text(text, encoding=encoding)
        (directory / "binary.txt").write_bytes(b"needle\x00binary\n")
        (directory / "legacy.txt").write_bytes("needle 中文编码\n".encode("gb18030"))
        (directory / "malformed.txt").write_bytes(b"\xff\xfea")
        result = await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", "paths": ["examples/tool-check/encodings"], "limit": 300}))
        data = self.full_data(result)
        self.assertEqual(data["total_matches"], 4)
        self.assertEqual({Path(item["path"]).name for item in data["matches"]}, {"utf8.txt", "utf8-bom.txt", "utf16.txt", "utf32.txt"})
        for item in data["matches"]:
            self.assertEqual(item["line"], 2)
            self.assertEqual(item["text"], "needle 真实证据")
            self.assertEqual(item["source"], str(self.code_root / item["path"]))
            self.assertNotIn("\\", item["path"])
        skipped = {Path(item["path"]).name: item["reason"] for item in data["skipped_paths"]}
        self.assertEqual(skipped, {"binary.txt": "binary", "legacy.txt": "invalid_encoding", "malformed.txt": "invalid_encoding"})
        self.assertEqual(data["scope"]["paths"], ["examples/tool-check/encodings"])
        self.assertFalse(data["scope"]["follow_symlinks"])
        self.assertEqual(data["cwd"], str(self.code_root))
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_read_file_decodes_bom_and_keeps_original_byte_version(self):
        path = "examples/tool-check/utf16.txt"
        target = self.code_root / path
        target.write_text("第一行\nneedle 第二行\n第三行\n", encoding="utf-16")
        read = self.full_data(await self.harness.tools.execute(self.call("read_file", {"path": path, "start_line": 2, "end_line": 2})))
        self.assertEqual(read["content"], "2: needle 第二行")
        self.assertEqual(read["encoding"], "utf-16")
        self.assertEqual(read["file_version"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(read["continuation"], {"path": path, "start_line": 3})
        target.write_bytes(b"needle\x00binary\n")
        refused = await self.harness.tools.execute(self.call("read_file", {"path": path}))
        self.assertEqual(refused["status"], "error")
        self.assertIn(path, refused["error"])
        self.assertIn("binary", refused["error"])

    async def test_directory_read_records_tool_error_and_keeps_task_usable(self):
        path = "examples/tool-check"
        original_task = json.loads(json.dumps(self.harness.task))
        original_state = json.loads(json.dumps(self.harness.state))
        call = self.call("read_file", {"path": path})
        refused = await self.harness._action(self.harness.tools.execute(call))
        self.assertEqual(refused["status"], "error")
        self.assertIn("普通文件", refused["error"])
        self.assertIn(path, refused["error"])
        self.assertIsNone(refused["data"])
        self.assertFalse(self.harness.busy)
        self.assertEqual(self.harness.task, original_task)
        self.assertEqual(self.harness.state, original_state)

        read = await self.harness._action(self.read_sample())
        self.assertEqual(read["status"], "success")
        self.assertEqual(read["data"]["content"], "1: value=before")
        records = self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")
        self.assertEqual([record["status"] for record in records], ["error", "success"])
        self.assertEqual(records[0]["operation_id"], refused["operation_id"])
        self.assertEqual(records[0]["call_id"], call["id"])
        self.assertEqual(records[0]["arguments"], {"path": path})
        self.assertEqual(records[0]["error"], refused["error"])
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        failed = [event for event in events if event["event_type"] == "tool.failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["payload"]["operation_id"], refused["operation_id"])
        self.assertFalse(any(event["event_type"] == "agent.failed" for event in events))
        self.assertEqual(self.harness.task, original_task)
        self.assertEqual(self.harness.state, original_state)
        self.assertEqual(self.harness.usage.summary()["tool_calls"], 2)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_search_limit_300_pages_real_matches_and_keeps_full_output(self):
        path = "examples/tool-check/paged.txt"
        (self.code_root / path).write_text("".join(f"needle {index}\n" for index in range(305)), encoding="utf-8")
        first = await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", "paths": [path, "examples/tool-check"], "limit": 300}))
        self.assertTrue(first["truncated"])
        self.assertIn('"scope":', first["data"])
        self.assertIn('"next_cursor": 300', first["data"])
        data = self.full_data(first)
        self.assertEqual(len(data["matches"]), 300)
        self.assertEqual(data["total_matches"], 305)
        self.assertEqual(data["next_cursor"], 300)
        last = self.full_data(await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", "paths": [path, "examples/tool-check"], "limit": 300, "cursor": data["next_cursor"]})))
        self.assertEqual([item["line"] for item in last["matches"]], [301, 302, 303, 304, 305])
        self.assertIsNone(last["next_cursor"])
        definition = next(item["function"] for item in self.harness.tools.definitions() if item["function"]["name"] == "search_text")
        self.assertEqual(definition["parameters"]["properties"]["limit"]["maximum"], 1000)
        self.assertEqual(definition["parameters"]["properties"]["limit"]["minimum"], 1)
        for arguments in ({"limit": 1001}, {"cursor": -1}, {"paths": []}):
            refused = await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", **arguments}))
            self.assertEqual(refused["status"], "error")

    async def test_search_reports_size_filter_and_fails_missing_path_or_invalid_regex(self):
        path = "examples/tool-check/large.txt"
        (self.code_root / path).write_bytes(b"needle" + b"x" * 2_000_000)
        result = self.full_data(await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", "paths": [path]})))
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["skipped_paths"], [{"path": path, "reason": "file_too_large", "size_bytes": 2_000_006}])
        missing = await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", "paths": ["examples/missing.py"]}))
        self.assertEqual(missing["status"], "error")
        self.assertIn("missing.py", missing["error"])
        invalid = await self.harness.tools.execute(self.call("search_text", {"pattern": "[", "paths": ["examples/tool-check"]}))
        self.assertEqual(invalid["status"], "error")

    @unittest.skipIf(os.name == "nt", "文件系统权限错误检查需要 Linux")
    async def test_search_preserves_actual_filesystem_permission_error(self):
        if os.geteuid() == 0:
            self.skipTest("root 身份会绕过文件读取权限")
        path = "examples/tool-check/unreadable.txt"
        target = self.code_root / path
        target.write_text("needle unreadable evidence\n", encoding="utf-8")
        target.chmod(0)
        try:
            if target.stat().st_mode & 0o777:
                self.skipTest("当前挂载不支持 POSIX 文件权限；可设置 IANVS_FILE_TOOL_TEST_ROOT 指向 Linux 文件系统")
            result = await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", "paths": ["examples/tool-check"]}))
            self.assertEqual(result["status"], "error")
            self.assertIn("Permission denied", result["error"])
            self.assertIn("unreadable.txt", result["error"])
        finally:
            target.chmod(0o600)

    async def test_search_prunes_managed_and_credential_paths(self):
        for directory in ("workspace", "sessions", ".env-backup"):
            target = self.code_root / "examples/tool-check" / directory / "private.txt"
            target.parent.mkdir()
            target.write_text("needle restricted content\n", encoding="utf-8")
        secret = self.code_root / "examples/tool-check/.env-local.txt"
        secret.write_text("needle restricted content\n", encoding="utf-8")
        (self.code_root / "examples/tool-check/visible.txt").write_text("needle public evidence\n", encoding="utf-8")
        result = self.full_data(await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", "paths": ["examples/tool-check"]})))
        self.assertEqual([item["text"] for item in result["matches"]], ["needle public evidence"])
        self.assertEqual(len(result["skipped_paths"]), 4)
        self.assertTrue(all(item["reason"] == "restricted_path" for item in result["skipped_paths"]))
        for path in ("examples/tool-check/.env-local.txt", "examples/tool-check/.env-backup/private.txt", str(self.outside)):
            refused = await self.harness.tools.execute(self.call("search_text", {"pattern": "needle", "paths": [path]}))
            self.assertEqual(refused["status"], "error")

    @unittest.skipIf(os.name == "nt", "符号链接范围检查在 Linux 环境执行")
    async def test_search_skips_links_without_reading_targets(self):
        directory = self.code_root / "examples/tool-check"
        (directory / "outside.txt").symlink_to(self.outside)
        (directory / "outside-dir").symlink_to(self.root, target_is_directory=True)
        (directory / ".env-alias.txt").symlink_to(directory / "sample.txt")
        result = self.full_data(await self.harness.tools.execute(self.call("search_text", {"pattern": "outside-preserved|value=before", "paths": ["examples/tool-check"]})))
        self.assertEqual([item["path"] for item in result["matches"]], ["examples/tool-check/sample.txt"])
        self.assertEqual(len(result["skipped_paths"]), 3)
        self.assertTrue(all(item["reason"] == "symbolic_link" for item in result["skipped_paths"]))
        refused = await self.harness.tools.execute(self.call("read_file", {"path": "examples/tool-check/.env-alias.txt"}))
        self.assertEqual(refused["status"], "error")
        self.assertIn("凭据", refused["error"])

    async def test_unregistered_validation_report_and_cross_instance_report_are_denied(self):
        reference = f"instances/{self.harness.current_instance['instance_id']}/validations/validation-unregistered/report.json"
        self.harness.store.write(reference, {"message": "未登记的真实文件"})
        refused = await self.harness.tools.execute(self.call("read_evidence", {"reference": reference}))
        self.assertEqual(refused["status"], "error")
        self.assertIn("已登记", refused["error"])
        other = self.harness.instances.create("examples/tool-check/benchmarkingjob.yaml")
        reference = f"instances/{other['instance_id']}/validations/validation-unregistered/report.json"
        self.harness.store.write(reference, {"message": "其他实例的真实文件"})
        refused = await self.harness.tools.execute(self.call("read_evidence", {"reference": reference}))
        self.assertEqual(refused["status"], "error")
        self.assertIn("当前实例", refused["error"])

    async def test_repeated_call_id_keeps_distinct_full_output(self):
        self.settings.tool_output_chars = 160
        target = self.code_root / "core/long-evidence.txt"
        target.write_text("第一轮证据\n" * 100, encoding="utf-8")
        first = await self.harness.tools.execute(self.call("read_file", {"path": "core/long-evidence.txt"}, "call-reused"))
        target.write_text("第二轮证据\n" * 100, encoding="utf-8")
        second = await self.harness.tools.execute(self.call("read_file", {"path": "core/long-evidence.txt"}, "call-reused"))
        self.assertTrue(first["truncated"])
        self.assertTrue(second["truncated"])
        self.assertNotEqual(first["full_output_ref"], second["full_output_ref"])
        saved_first = self.harness.store.read(first["full_output_ref"])
        saved_second = self.harness.store.read(second["full_output_ref"])
        self.assertIn("第一轮证据", saved_first["content"])
        self.assertNotIn("第二轮证据", saved_first["content"])
        self.assertIn("第二轮证据", saved_second["content"])
        records = self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")
        self.assertEqual(len([record for record in records if record["call_id"] == "call-reused"]), 2)
        self.assertEqual(records[-2]["full_output_ref"], first["full_output_ref"])
        self.assertEqual(records[-1]["full_output_ref"], second["full_output_ref"])
        completed = [event["payload"] for event in self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl") if event["event_type"] == "tool.completed"]
        self.assertTrue(completed[-1]["truncated"])
        self.assertEqual(completed[-1]["data"], second["data"])
        self.assertEqual(completed[-1]["full_output_ref"], second["full_output_ref"])
        self.assertEqual(records[-1]["data"]["content"], saved_second["content"])
        self.harness._update_observation({"facts": [{"value": "第一轮读取的完整输出仍可追溯", "evidence_ref": first["full_output_ref"]}], "hypotheses": []})
        self.assertEqual(len(self.harness.state["facts"]), 1)
        self.assertEqual(self.harness.state["facts"][0]["evidence_status"], "registered_reference")

    async def test_failed_environment_tool_retains_readable_owned_log(self):
        (self.code_root / "requirements.txt").write_text("", encoding="utf-8")
        result = await self.harness.tools.execute(self.call("prepare_instance", {}))
        self.assertEqual(result["status"], "error")
        environment = self.harness.current_instance["environment"]
        self.assertEqual(environment["status"], "failed")
        self.assertEqual(environment["failed_step"]["name"], "core_install")
        reference = environment["failed_step"]["log_ref"]
        evidence = await self.harness.tools.execute(self.call("read_evidence", {"reference": reference}))
        self.assertEqual(evidence["status"], "success")
        self.assertIn("does not exist", evidence["data"]["content"])
        state = await self.harness.tools.execute(self.call("get_instance_state", {}))
        self.assertIn("failed_step", state["data"]["environment"])
        self.assertTrue(state["data"]["environment"]["warnings"])
        records = self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")
        self.assertEqual(records[0]["status"], "error")
        self.assertEqual(records[0]["name"], "prepare_instance")
        other = self.harness.instances.create("examples/tool-check/benchmarkingjob.yaml")
        other_environment = await self.harness.environment.prepare(other, install=False)
        denied = await self.harness.tools.execute(self.call("read_evidence", {"reference": other_environment["steps"][0]["log_ref"]}))
        self.assertEqual(denied["status"], "error")
        self.assertIn("只能读取当前实例", denied["error"])
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_session_recovery_does_not_reuse_previous_call_result(self):
        call = self.call("read_file", {"path": "core/sample.txt"}, "call-reused")
        first = await self.harness.tools.execute(call)
        tool_path = self.harness.session_relative_path + "/tools.jsonl"
        self.harness._messages = [{"role": "assistant", "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}]
        self.harness.state["open_tool_round"] = {"assistant_index": 0, "request_id": "stored-history-check", "tools_start_index": len(self.harness.store.read_events(tool_path))}
        self.harness._save_session()
        await self.harness.close()
        self.harness = AgentHarness(self.settings)
        recovered = json.loads(self.harness._messages[-1]["content"])
        self.assertEqual(recovered["status"], "interrupted")
        self.assertNotEqual(recovered.get("operation_id"), first["operation_id"])
        self.assertNotIn("open_tool_round", self.harness.state)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_session_recovery_uses_current_operation_and_arguments(self):
        call = self.call("read_file", {"path": "core/sample.txt"}, "call-reused")
        await self.harness.tools.execute(call)
        tool_path = self.harness.session_relative_path + "/tools.jsonl"
        self.harness._messages = [{"role": "assistant", "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}]
        self.harness.state["open_tool_round"] = {"assistant_index": 0, "request_id": "stored-history-check", "tools_start_index": len(self.harness.store.read_events(tool_path))}
        (self.code_root / "core/sample.txt").write_text("value=current-operation\n", encoding="utf-8")
        current = await self.harness.tools.execute(call)
        self.harness._save_session()
        await self.harness.close()
        self.harness = AgentHarness(self.settings)
        recovered = json.loads(self.harness._messages[-1]["content"])
        self.assertEqual(recovered["status"], "success")
        self.assertEqual(recovered["operation_id"], current["operation_id"])
        self.assertIn("value=current-operation", recovered["data"]["content"])
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_long_result_recovery_reuses_saved_output_and_preview(self):
        self.settings.tool_output_chars = 160
        target = self.code_root / "core/long-recovery.txt"
        target.write_text("真实文件证据及必要配置值\n" * 150, encoding="utf-8")
        call = self.call("read_file", {"path": "core/long-recovery.txt"}, "call-long-recovery")
        tool_path = self.harness.session_relative_path + "/tools.jsonl"
        self.harness._messages = [{"role": "assistant", "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}]
        self.harness.state["open_tool_round"] = {"assistant_index": 0, "request_id": "interrupted-real-file-operation", "tools_start_index": len(self.harness.store.read_events(tool_path))}
        self.harness._save_session()
        actual = await self.harness.tools.execute(call)
        output_before = self.harness.store.path(actual["full_output_ref"]).read_bytes()
        counters_before = self.harness.store.read("counters.json")
        await self.harness.close()
        self.harness = AgentHarness(self.settings)
        recovered = json.loads(self.harness.history()[-1]["content"])
        self.assertTrue(recovered["truncated"])
        self.assertEqual(recovered["data"], actual["data"])
        self.assertEqual(recovered["operation_id"], actual["operation_id"])
        self.assertEqual(recovered["full_output_ref"], actual["full_output_ref"])
        self.assertEqual(self.harness.store.path(recovered["full_output_ref"]).read_bytes(), output_before)
        self.assertEqual(self.harness.store.read("counters.json")["OUTPUT"], counters_before["OUTPUT"])
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_session_snapshot_wins_over_partly_updated_projections(self):
        actual = await self.read_sample()
        self.harness.state["next_goal"] = "已有真实工具记录，下一步仍由当前任务决定"
        self.harness._save_session()
        path = self.harness.session_relative_path
        snapshot = self.harness.store.read(path + "/session_snapshot.json")
        self.harness.store.write(path + "/task.json", {"task_id": "partial-projection-task"})
        self.harness.store.write(path + "/state.json", {"status": "completed", "verification": {"passed": True}})
        self.harness.store.write(path + "/history.json", [])
        await self.harness.close()
        self.harness = AgentHarness(self.settings)
        self.assertEqual(self.harness.task, snapshot["task"])
        self.assertEqual(self.harness.state, snapshot["state"])
        self.assertEqual(self.harness.history(), snapshot["history"])
        self.assertEqual(self.harness.session["instance_id"], snapshot["session"]["instance_id"])
        self.assertEqual(self.harness.store.read_events(path + "/tools.jsonl")[-1]["operation_id"], actual["operation_id"])
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_busy_actions_preserve_task_and_history(self):
        before = json.dumps({"task": self.harness.task, "state": self.harness.state, "history": self.harness.history()}, ensure_ascii=False, sort_keys=True)
        self.harness.busy = True
        try:
            with self.assertRaises(RuntimeError):
                await self.harness.send_message("繁忙时不得修改当前任务")
            with self.assertRaises(RuntimeError):
                await self.harness.run_current()
        finally:
            self.harness.busy = False
        after = json.dumps({"task": self.harness.task, "state": self.harness.state, "history": self.harness.history()}, ensure_ascii=False, sort_keys=True)
        self.assertEqual(before, after)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_instance_tools_return_relevant_summary_and_saved_full_record(self):
        for index in range(40):
            (self.code_root / f"examples/tool-check/evidence-{index}.txt").write_text(f"current-value-{index}\n", encoding="utf-8")
        impact = await self.harness.tools.execute(self.call("get_change_impact", {}))
        self.assertEqual(impact["status"], "success")
        self.assertFalse(impact["truncated"])
        summary = impact["data"]
        self.assertNotIn("source_manifest", summary)
        self.assertNotIn("current_manifest", summary)
        self.assertNotIn("packages", summary["environment"])
        self.assertTrue(summary["needs_verification"])
        self.assertEqual(summary["changed_files_count"], 40)
        self.assertTrue(summary["changed_files_truncated"])
        full = self.harness.store.read(summary["full_output_ref"])
        self.assertIn("examples/tool-check/evidence-39.txt", full["current_manifest"])
        read = await self.harness.tools.execute(self.call("read_evidence", {"reference": summary["full_output_ref"], "start_line": 1, "end_line": 20}))
        self.assertEqual(read["status"], "success")
        self.assertIn(summary["instance_id"], read["data"]["content"])
        current = await self.harness.tools.execute(self.call("get_instance_state", {}))
        self.assertEqual(current["data"]["verification"], summary["verification"])
        self.assertNotEqual(current["data"]["full_output_ref"], summary["full_output_ref"])
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_async_session_save_finishes_before_cancellation_and_recovery(self):
        self.harness.state["next_goal"] = "等待真实文件写入完成后保留此状态"
        lock = FileLock(str(self.harness.store.path(self.harness.session_relative_path + "/session_snapshot.json")) + ".lock")
        lock.acquire()
        pending = asyncio.create_task(self.harness._save_session_async())
        try:
            await asyncio.sleep(0.05)
            self.assertFalse(pending.done())
            pending.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(pending.done())
        finally:
            lock.release()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        saved = self.harness.store.read(self.harness.session_relative_path + "/session_snapshot.json")
        self.assertEqual(saved["task"], self.harness.task)
        self.assertEqual(saved["state"]["next_goal"], self.harness.state["next_goal"])
        self.assertEqual(saved["history"], self.harness.history())

    async def test_cancellation_closes_a_persisted_unexecuted_tool_round(self):
        call = self.call("read_file", {"path": "core/sample.txt"})
        self.harness._messages.append({"role": "assistant", "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]})
        self.harness.state["open_tool_round"] = {"assistant_index": len(self.harness._messages) - 1, "tools_start_index": 0}
        await self.harness._save_session_async()
        pending = asyncio.create_task(self.harness._action(asyncio.sleep(60)))
        await asyncio.sleep(0)
        await self.harness.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        complete_interactions(self.harness.history())
        result = json.loads(self.harness.history()[-1]["content"])
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(self.harness.state["status"], "cancelled")
        self.assertNotIn("open_tool_round", self.harness.state)
        saved = self.harness.store.read(self.harness.session_relative_path + "/session_snapshot.json")
        complete_interactions(saved["history"])
        self.assertEqual(saved["state"]["status"], "cancelled")
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_observation_merges_registered_file_facts_without_changing_acceptance(self):
        first = await self.read_sample()
        original_task = json.loads(json.dumps(self.harness.task))
        original_verification = dict(self.harness.state["verification"])
        first_fact = {"value": "第一次文件读取包含原始值", "evidence_ref": "tool:" + first["operation_id"]}
        self.harness._update_observation({"facts": [first_fact], "hypotheses": []})
        (self.code_root / "examples/tool-check/sample.txt").write_text("value=current\n", encoding="utf-8")
        second = await self.read_sample()
        second_fact = {"value": "第二次文件读取包含当前值", "evidence_ref": "tool:" + second["operation_id"]}
        self.harness._update_observation({"facts": [second_fact, first_fact], "hypotheses": []})
        self.assertEqual(len(self.harness.state["facts"]), 2)
        self.assertEqual({fact["evidence_ref"] for fact in self.harness.state["facts"]}, {first_fact["evidence_ref"], second_fact["evidence_ref"]})
        self.assertEqual(self.harness.task, original_task)
        self.assertEqual(self.harness.state["verification"], original_verification)
        self.harness._update_observation({"facts": [{"value": "没有登记的文件不能成为事实", "evidence_ref": "instances/another-instance/runs/run-0001/run.json"}], "hypotheses": []})
        self.assertEqual(len(self.harness.state["facts"]), 2)
        self.assertEqual(self.harness.state["hypotheses"][-1]["status"], "unverified")

    async def test_new_message_after_terminal_conversation_keeps_old_task_record(self):
        for status in ("completed", "cancelled", "failed", "blocked", "interrupted"):
            self.harness.state["status"] = status
            old_task = json.loads(json.dumps(self.harness.task))
            old_state = json.loads(json.dumps(self.harness.state))
            self.harness._ensure_task("只回复新的短消息")
            self.assertNotEqual(self.harness.task["task_id"], old_task["task_id"])
            self.assertEqual(self.harness.task["goal"], "只回复新的短消息")
            saved = self.harness.store.read(self.harness.session_relative_path + f"/tasks/{old_task['task_id']}.json")
            self.assertEqual(saved, {"task": old_task, "state": old_state})
        read = await self.read_sample()
        self.assertEqual(read["status"], "success")
        self.assertFalse(self.harness._requires_observation([read]))
        self.assertTrue(self.harness._requires_observation([self.call("apply_patch", {})]))
        self.assertTrue(self.harness._requires_observation([self.call("unregistered_tool", {})]))
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_interrupted_repair_retains_original_goal_and_acceptance(self):
        self.harness.task["kind"] = "repair"
        self.harness.task["acceptance"]["required_metrics"] = ["accuracy"]
        original = json.loads(json.dumps(self.harness.task))
        for status in ("cancelled", "failed", "blocked", "interrupted"):
            self.harness.state["status"] = status
            self.harness._ensure_task("继续查看本轮证据")
            self.assertEqual(self.harness.task, original)
        read = await self.read_sample()
        self.assertEqual(read["status"], "success")
        self.assertTrue(self.harness.tools.is_read_only("read_file"))
        self.assertTrue(self.harness._requires_observation([read]))
        self.assertTrue(self.harness.tools.is_read_only("get_instance_state"))
        self.assertFalse(self.harness.tools.is_read_only("apply_patch"))
        self.assertFalse(self.harness.tools.is_read_only("run_example"))
        self.assertFalse(self.harness.tools.is_read_only("unregistered_tool"))
        self.assertEqual(self.harness.task, original)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_failed_real_run_can_be_followed_by_independent_diagnosis(self):
        await self.harness._finish("completed", "文件检查完成")
        with self.assertRaisesRegex(RuntimeError, "环境尚未准备"):
            await self.harness.run_current()
        previous_task = json.loads(json.dumps(self.harness.task))
        self.assertEqual(previous_task["kind"], "run")
        self.assertEqual(self.harness.state["status"], "failed")
        self.assertIn("环境尚未准备", self.harness.state["result"]["content"])
        instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        run_id = instance["last_run_id"]
        self.harness._ensure_task("分析这次运行失败的原因")
        self.assertEqual(self.harness.task["kind"], "conversation")
        self.assertNotEqual(self.harness.task["task_id"], previous_task["task_id"])
        self.assertEqual(self.harness.state["last_run_id"], run_id)
        self.assertEqual(self.harness.state["last_run"]["run_id"], run_id)
        self.assertEqual(self.harness.state["last_run"]["command"], [])
        self.assertIn("环境尚未准备", self.harness.state["last_run"]["error"])
        archived = self.harness.store.read(self.harness.session_relative_path + f"/tasks/{previous_task['task_id']}.json")
        self.assertEqual(archived["task"], previous_task)
        self.assertEqual(archived["state"]["status"], "failed")
        self.assertFalse(self.harness._requires_observation([self.call("get_run_state", {})]))
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_manual_compaction_preserves_actual_failed_run_result(self):
        with self.assertRaisesRegex(RuntimeError, "环境尚未准备"):
            await self.harness.run_current()
        original_task = json.loads(json.dumps(self.harness.task))
        original_state = json.loads(json.dumps(self.harness.state))
        instance_id = self.harness.current_instance["instance_id"]
        original_instance = self.harness.instances.get(instance_id)
        run_id = original_instance["last_run_id"]
        run_reference = f"instances/{instance_id}/runs/{run_id}/run.json"
        original_run = self.harness.store.read(run_reference)
        before = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        result = await self.harness.compact()
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.harness.task, original_task)
        self.assertEqual(self.harness.state, original_state)
        self.assertEqual(self.harness.instances.get(instance_id), original_instance)
        self.assertEqual(self.harness.store.read(run_reference), original_run)
        saved = self.harness.store.read(self.harness.session_relative_path + "/session_snapshot.json")
        self.assertEqual(saved["task"], original_task)
        self.assertEqual(saved["state"], original_state)
        after = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        added = after[len(before):]
        self.assertEqual(sum(event["event_type"] == "context.compaction_skipped" for event in added), 1)
        self.assertFalse(any(event["event_type"].startswith("agent.") for event in added))
        self.assertFalse(self.harness._compact_requested)
        self.assertFalse(self.harness.busy)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)

    async def test_queued_compaction_skip_is_consumed_before_next_plan(self):
        self.settings.approval_policy = "manual"
        read = await self.read_sample()
        call = self.call("apply_patch", {"patch": self.patch_for("examples/tool-check/sample.txt")})
        requested = asyncio.Event()
        approvals = []

        def observe(event):
            if event["event_type"] == "approval.requested":
                approvals.append(event["payload"]["approval_id"])
                requested.set()

        unsubscribe = self.harness.events.subscribe(observe)
        active = asyncio.create_task(self.harness._action(self.harness.tools.execute(call)))
        try:
            await asyncio.wait_for(requested.wait(), timeout=10)
            self.assertTrue(self.harness.busy)
            queued = await self.harness.compact()
        finally:
            unsubscribe()
            if approvals:
                await self.harness.approve(approvals[-1], False)
            else:
                active.cancel()
        denied = await active
        self.assertEqual(denied["status"], "denied")
        self.assertEqual(queued, {"status": "queued"})
        self.assertTrue(self.harness._compact_requested)
        credential_name = "IANVS_QUEUE_TEST_KEY_" + uuid.uuid4().hex
        self.settings.profile.api_key_env = credential_name
        with self.assertRaisesRegex(ValueError, "缺少模型凭据环境变量 " + credential_name):
            await self.harness._action(self.harness._loop())
        self.assertFalse(self.harness._compact_requested)
        self.assertFalse(self.harness.busy)
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        skipped = [event for event in events if event["event_type"] == "context.compaction_skipped"]
        prepared = [event for event in events if event["event_type"] == "context.prepared"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(len(prepared), 1)
        self.assertLess(skipped[0]["sequence"], prepared[0]["sequence"])
        self.assertFalse(any(event["event_type"] in {"llm.started", "context.compacted"} for event in events))
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.harness.provider.client)

    async def test_terminal_diagnosis_displays_evidence_and_survives_restore(self):
        read = await self.read_sample()
        self.harness._update_observation({
            "facts": [{"text": "实例文件实际内容为 value=before", "evidence_ref": "tool:" + read["operation_id"]}, {"value": "尚未核实的远程模型条件", "evidence_ref": "unregistered:model"}],
            "hypotheses": ["模型是否可取得仍需核实"],
        })
        self.harness.state["next_goal"] = "取得合法模型后重新执行原任务"
        before_history = self.harness.history()
        before_verification = dict(self.harness.state["verification"])
        result = await self.harness._finish("blocked", "当前模型尚未取得")
        self.assertIn("实例文件实际内容为 value=before", result["content"])
        self.assertIn("tool:" + read["operation_id"], result["content"])
        self.assertIn("尚待核实", result["content"])
        self.assertIn("尚未核实的远程模型条件", result["content"])
        self.assertIn("取得合法模型后重新执行原任务", result["content"])
        self.assertIn("尚未通过", result["content"])
        self.assertEqual(self.harness.history(), before_history)
        self.assertEqual(self.harness.state["verification"], before_verification)
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        visible = [event["payload"] for event in events if event["event_type"] == "agent.message"]
        self.assertEqual(visible[-1], {"role": "result", "content": result["content"]})
        terminal = [event["payload"] for event in events if event["event_type"] == "agent.blocked"][-1]
        self.assertEqual(terminal["result"], result)
        await self.harness.close()
        self.harness = AgentHarness(self.settings)
        self.assertEqual(self.harness.state["result"], result)
        self.assertEqual(self.harness.state["verification"], before_verification)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_completed_conversation_does_not_duplicate_final_answer(self):
        result = await self.harness._finish("completed", "文件检查已经完成")
        self.assertEqual(result["content"], "文件检查已经完成")
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        self.assertFalse(any(event["event_type"] == "agent.message" for event in events))
        self.assertEqual(self.harness.state["result"], result)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_registry_keeps_executing_after_cumulative_tool_threshold(self):
        self.settings.max_tool_calls = 1
        first = await self.read_sample()
        self.assertEqual(first["status"], "success")
        self.harness.usage.check_budget(self.settings)
        second = await self.read_sample()
        self.assertEqual(second["status"], "success")
        self.assertEqual(len(self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")), 2)
        self.assertEqual(self.harness.usage.summary()["tool_calls"], 2)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def approval_task(self, call):
        self.settings.approval_policy = "manual"
        requested = asyncio.Event()
        approvals = []

        def observe(event):
            if event["event_type"] == "approval.requested":
                approvals.append(event["payload"]["approval_id"])
                requested.set()

        unsubscribe = self.harness.events.subscribe(observe)
        task = asyncio.create_task(self.harness.tools.execute(call))
        await asyncio.wait_for(requested.wait(), timeout=10)
        unsubscribe()
        return task, approvals[-1]

    async def test_approval_parameter_change_is_rejected_and_original_saved(self):
        read = await self.read_sample()
        original_patch = self.patch_for("examples/tool-check/sample.txt")
        call = self.call("apply_patch", {"patch": original_patch})
        task, approval_id = await self.approval_task(call)
        call["arguments"]["patch"] = self.patch_for("examples/tool-check/sample.txt", after="value=changed-after-approval")
        await self.harness.approve(approval_id, True)
        result = await task
        self.assertIn(result["status"], {"error", "denied"})
        self.assertIn("参数发生变化", result["error"])
        self.assertFalse(self.harness.tools.pending)
        events = self.harness.store.read_events("events.jsonl")
        decided = [event for event in events if event["event_type"] == "approval.decided"][-1]
        self.assertEqual(decided["payload"]["arguments"]["patch"], original_patch)
        self.assertEqual((self.code_root / "examples/tool-check/sample.txt").read_text(encoding="utf-8"), "value=before\n")

    async def test_cancelled_approval_has_terminal_record(self):
        read = await self.read_sample()
        call = self.call("apply_patch", {"patch": self.patch_for("examples/tool-check/sample.txt")})
        task, approval_id = await self.approval_task(call)
        self.harness.tools.cancel_approvals()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertNotIn(approval_id, self.harness.tools.pending)
        records = self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")
        self.assertEqual(records[-1]["status"], "cancelled")
        self.assertEqual(records[-1]["call_id"], call["id"])
        self.assertEqual(self.harness.usage.summary()["tool_calls"], 2)
        self.assertEqual((self.code_root / "examples/tool-check/sample.txt").read_text(encoding="utf-8"), "value=before\n")

    @unittest.skipIf(os.name == "nt", "受限调查命令在 Linux 环境执行")
    async def test_read_only_commands_need_no_modification_approval(self):
        self.settings.approval_policy = "deny"
        self.assertTrue(self.harness.tools.is_read_only("bash"))
        self.assertFalse(self.harness._requires_observation([self.call("bash", {"command": "ls core"})]))
        listed = await self.harness.tools.execute(self.call("bash", {"command": "ls core"}))
        self.assertEqual(listed["status"], "success")
        self.assertEqual(listed["data"]["return_code"], 0)
        self.assertIn("sample.txt", listed["data"]["stdout"])
        working = await self.harness.tools.execute(self.call("bash", {"command": "pwd"}))
        self.assertEqual(working["status"], "success")
        self.assertEqual(working["data"]["stdout"].strip(), str(self.code_root))
        for command in ("ls /", "stat " + str(self.outside), "ls core; pwd", "ls --dereference core", "rm core/sample.txt"):
            result = await self.harness.tools.execute(self.call("bash", {"command": command}))
            self.assertIn(result["status"], {"error", "denied"}, command)
        self.assertEqual((self.code_root / "core/sample.txt").read_text(encoding="utf-8"), "value=before\n")
        self.assertEqual(self.outside.read_text(encoding="utf-8"), "outside-preserved\n")
        events = self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl")
        self.assertFalse(any(event["event_type"].startswith("approval.") for event in events))
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    @unittest.skipIf(os.name == "nt", "受限调查命令在 Linux 环境执行")
    async def test_scoped_command_reads_only_instance_directory(self):
        allowed = await self.harness.tools.execute(self.call("bash", {"command": "ls core"}))
        self.assertEqual(allowed["status"], "success")
        self.assertEqual(allowed["data"]["return_code"], 0)
        self.assertIn("sample.txt", allowed["data"]["stdout"])
        for command in ("ls /", "stat " + str(self.outside), "ls .env", "ls core; pwd", "ls --dereference core"):
            result = await self.harness.tools.execute(self.call("bash", {"command": command}))
            self.assertIn(result["status"], {"error", "denied"}, command)
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_read_evidence_does_not_read_session_or_other_instance(self):
        for reference in (self.harness.session_relative_path + "/session.json", "instances/another-instance/runs/RUN-0001/verification.json"):
            result = await self.harness.tools.execute(self.call("read_evidence", {"reference": reference}))
            self.assertIn(result["status"], {"error", "denied"})
            self.assertIn("只能读取", result["error"])

    async def test_invalid_schema_never_executes_patch(self):
        result = await self.harness.tools.execute(self.call("apply_patch", {"patch": self.patch_for("examples/tool-check/sample.txt"), "unexpected": True}))
        self.assertIn(result["status"], {"error", "denied"})
        self.assertEqual((self.code_root / "examples/tool-check/sample.txt").read_text(encoding="utf-8"), "value=before\n")
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_patch_requires_reading_in_current_task(self):
        patch = self.patch_for("examples/tool-check/sample.txt")
        self.assertEqual((await self.harness.tools.execute(self.call("apply_patch", {"patch": patch})))["status"], "error")
        await self.read_sample()
        await self.harness._finish("completed", "已完成文件读取检查")
        self.harness._ensure_task("检查新任务的补丁读取依据", kind="repair")
        result = await self.harness.tools.execute(self.call("apply_patch", {"patch": patch}))
        self.assertEqual(result["status"], "error")
        self.assertIn("修改依据缺失", result["error"])
        self.assertEqual((self.code_root / "examples/tool-check/sample.txt").read_text(), "value=before\n")

    async def test_patch_binding_survives_real_session_restore(self):
        read = await self.read_sample()
        task = json.loads(json.dumps(self.harness.task))
        await self.harness._save_session_async()
        instance_id = self.harness.current_instance["instance_id"]
        await self.harness.close()
        self.harness = AgentHarness(self.settings)
        await self.harness.restore_instance(instance_id)
        self.assertEqual(self.harness.task, task)
        result = await self.harness.tools.execute(self.call("apply_patch", {"patch": self.patch_for("examples/tool-check/sample.txt")}))
        self.assertEqual(result["status"], "success", result.get("error"))
        record = self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl")[-1]
        self.assertEqual(record["approval_context"]["read_evidence_refs"]["examples/tool-check/sample.txt"], "tool:" + read["operation_id"])
        self.assertEqual(self.harness.usage.summary()["model_calls"], 0)

    async def test_patch_approval_rechecks_bound_file_without_explicit_hash(self):
        read = await self.read_sample()
        call = self.call("apply_patch", {"patch": self.patch_for("examples/tool-check/sample.txt")})
        active, approval_id = await self.approval_task(call)
        bound = self.harness.tools.pending[approval_id]["record"]["resolved_operation"]
        self.assertEqual(bound["expected_versions"]["examples/tool-check/sample.txt"], read["data"]["file_version"])
        target = self.code_root / "examples/tool-check/sample.txt"
        target.write_bytes(b"value=external\n")
        await self.harness.approve(approval_id, True)
        result = await active
        self.assertEqual(result["status"], "error")
        self.assertIn("文件已变化", result["error"])
        self.assertEqual(target.read_bytes(), b"value=external\n")
        self.assertEqual(self.harness.state.get("changes"), [])

    async def test_new_patch_file_requires_formal_absence_read(self):
        path = "examples/tool-check/new.txt"
        patch = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+created\n"
        self.assertEqual((await self.harness.tools.execute(self.call("apply_patch", {"patch": patch})))["status"], "error")
        read = await self.harness.tools.execute(self.call("read_file", {"path": path}))
        self.assertEqual(read["status"], "success")
        self.assertFalse(read["data"]["exists"])
        self.assertEqual(read["data"]["file_version"], "absent")
        target = self.code_root / path
        target.write_bytes(b"external\n")
        self.assertEqual((await self.harness.tools.execute(self.call("apply_patch", {"patch": patch})))["status"], "error")
        self.assertEqual(target.read_bytes(), b"external\n")
        target.unlink()
        result = await self.harness.tools.execute(self.call("apply_patch", {"patch": patch}))
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual((self.code_root / path).read_bytes(), b"created\n")

    async def test_direct_patch_rejects_incorrect_explicit_version(self):
        read = await self.read_sample()
        path = "examples/tool-check/sample.txt"
        with self.assertRaisesRegex(PermissionError, "expected_versions"):
            await CodeTools(self.harness).apply_patch(self.patch_for(path), {path: read["data"]["file_version"] + "0"})
        self.assertEqual((self.code_root / path).read_bytes(), b"value=before\n")


if __name__ == "__main__":
    unittest.main()
