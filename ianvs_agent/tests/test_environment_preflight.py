# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings
from ianvs_agent.runtime.environment import EnvironmentManager
from ianvs_agent.runtime.instances import InstanceManager, file_manifest
from ianvs_agent.runtime.processes import recorded_process
from ianvs_agent.storage.local_store import LocalStore


class EnvironmentPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.project = Path(__file__).resolve().parents[2]
        artifact = Path(os.environ.get("IANVS_ENVIRONMENT_PREFLIGHT_ROOT", self.project / "workspace/environment-preflight-tests"))
        control = LocalStore(artifact)
        self.case_root = artifact / control.allocate_id("preflight-case")
        source = self.case_root / "source"
        self.benchmark = "examples/Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning/benchmarkingjob.yaml"
        self.invalid_name = str(Path(self.benchmark).parent / "requirement.txt").replace("\\", "/")
        for relative in (self.benchmark, self.invalid_name, "requirements.txt", "resources/third_party/sedna-0.6.0.1-py3-none-any.whl", ".github/workflows/validator/data/example_inventory.yaml"):
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.project / relative, destination)
        self.settings = AgentSettings(project_root=source, state_root=self.case_root / "state", python_executable=sys.executable)
        self.store = LocalStore(self.settings.state_root)
        self.events = EventBus(self.store, task_id="environment-preflight-test")
        self.instance = InstanceManager(self.settings, self.store, self.events).create(self.benchmark)
        self.root = Path(self.instance["code_root"])
        self.manager = EnvironmentManager(self.settings, self.store, self.events)

    def declare_invalid(self):
        self.instance["inventory_entry"]["requirements_file"] = self.invalid_name
        self.store.write(f"instances/{self.instance['instance_id']}/instance.json", self.instance)

    def saved_plan(self):
        saved = self.store.read(f"instances/{self.instance['instance_id']}/instance.json")
        return self.store.read(saved["environment_operation_ref"])

    async def test_actual_invalid_declaration_precedes_venv_and_retries_after_changes(self):
        self.declare_invalid()
        before = copy.deepcopy(self.instance)
        with self.assertRaisesRegex(ValueError, "dashscope") as failure:
            await self.manager.prepare(self.instance)
        plan = self.saved_plan()
        self.assertEqual(plan["status"], "invalid")
        self.assertIn("plan_id=" + plan["plan_id"], str(failure.exception))
        self.assertIn("evidence_ref=" + plan["evidence_ref"], str(failure.exception))
        self.assertFalse(Path(plan["venv_root"]).exists())
        self.assertFalse((self.store.root / "environments").exists())
        self.assertEqual(plan["requirements_file_versions"][self.invalid_name], plan["file_versions"][self.invalid_name])
        self.assertEqual(plan["file_versions"], file_manifest(plan["snapshot_root"]))
        operation = self.store.read(plan["parse_operation_ref"])
        self.assertEqual(operation["exit_code"], 0)
        self.assertFalse(operation["isolation"]["network"])
        self.assertEqual(operation["readonly_mappings"], [{"source": plan["snapshot_root"], "destination": plan["code_root"]}])
        self.assertTrue(Path(operation["log_ref"]).is_file())
        self.assertEqual(self.store.read(plan["parse_result_ref"]), plan["dependencies"])
        for key in ("environment", "status", "needs_verification", "verification"):
            self.assertEqual(self.instance.get(key), before.get(key))
        with self.assertRaisesRegex(ValueError, "dashscope"):
            await self.manager.prepare(self.instance)
        retried = self.saved_plan()
        self.assertNotEqual(retried["plan_id"], plan["plan_id"])
        self.assertEqual(retried["status"], "invalid")
        self.assertEqual(self.store.read(plan["evidence_ref"]), plan)
        self.assertFalse(Path(retried["venv_root"]).exists())
        self.assertNotIn(self.instance["instance_id"], self.manager._preparing)
        self.assertIsNone(self.manager._active_task)
        path = self.root / self.invalid_name
        path.write_text(path.read_text(encoding="utf-8").replace("dashscope=1.24.5", "dashscope==1.24.5"), encoding="utf-8")
        changed = await self.manager._preflight_prepare(self.instance, None)
        self.assertNotEqual(changed["plan_id"], plan["plan_id"])
        self.assertEqual(changed["status"], "planned")
        self.assertTrue(changed["dependencies"]["passed"])
        self.assertFalse(Path(changed["venv_root"]).exists())
        self.assertNotEqual(changed["requirements_file_versions"][self.invalid_name], plan["requirements_file_versions"][self.invalid_name])
        self.assertEqual(self.store.read(plan["evidence_ref"]), plan)

    async def test_local_find_links_directory_can_be_created_and_rechecked_with_same_file_manifest(self):
        self.instance["inventory_entry"]["requirements_file"] = "dependencies.txt"
        (self.root / "dependencies.txt").write_text("--find-links ./wheels\npackaging\n", encoding="utf-8")
        before = file_manifest(self.root)
        with self.assertRaisesRegex(ValueError, "find-links"):
            await self.manager.prepare(self.instance)
        failed = self.saved_plan()
        self.assertEqual(failed["status"], "invalid")
        (self.root / "wheels").mkdir()
        self.assertEqual(file_manifest(self.root), before)
        retried = await self.manager._preflight_prepare(self.instance, None)
        self.assertNotEqual(retried["plan_id"], failed["plan_id"])
        self.assertTrue(retried["dependencies"]["passed"])
        self.assertFalse(self.store.read(retried["parse_operation_ref"])["isolation"]["network"])
        self.assertFalse(Path(retried["venv_root"]).exists())
        self.assertEqual(self.store.read(failed["evidence_ref"]), failed)
        self.assertIn("wheels", retried["source_directories"])
        self.assertTrue((Path(retried["snapshot_root"]) / "wheels").is_dir())
        (self.root / "wheels").rmdir()
        self.assertEqual(file_manifest(self.root), before)
        with self.assertRaisesRegex(ValueError, "目录"):
            self.manager.revalidate(self.instance, retried["plan_id"])
        (self.root / "wheels").mkdir()
        (Path(retried["snapshot_root"]) / "wheels").rmdir()
        with self.assertRaisesRegex(RuntimeError, "快照目录"):
            self.manager.revalidate(self.instance, retried["plan_id"])

    async def test_snapshot_excludes_protected_and_symlink_directories(self):
        outside = self.case_root / "outside"
        outside.mkdir()
        (outside / "external.txt").write_text("外部文件", encoding="utf-8")
        (self.root / "outside-link").symlink_to(outside, target_is_directory=True)
        hidden = self.root / "workspace"
        hidden.mkdir()
        (hidden / "hidden.txt").write_text("受保护文件", encoding="utf-8")
        empty = self.root / "empty-input"
        empty.mkdir()
        plan = await self.manager._preflight_prepare(self.instance, None)
        self.assertTrue(plan["dependencies"]["passed"])
        self.assertIn("empty-input", plan["source_directories"])
        self.assertNotIn("outside-link", plan["source_directories"])
        self.assertNotIn("workspace", plan["source_directories"])
        snapshot = Path(plan["snapshot_root"])
        self.assertTrue((snapshot / "empty-input").is_dir())
        self.assertFalse((snapshot / "outside-link").exists())
        self.assertFalse((snapshot / "workspace").exists())

    async def test_plan_without_directory_versions_requires_new_approval_plan(self):
        plan = self.manager.preview_prepare(self.instance)
        plan.pop("source_directories")
        self.store.write(plan["evidence_ref"], plan)
        with self.assertRaisesRegex(ValueError, "缺少目录版本记录"):
            self.manager.revalidate(self.instance, plan["plan_id"])
        changed = self.manager.preview_prepare(self.instance)
        self.assertNotEqual(changed["plan_id"], plan["plan_id"])
        self.assertIn("source_directories", changed)
        self.assertFalse(Path(changed["venv_root"]).exists())

    async def test_undeclared_example_file_is_excluded_from_actual_preparation_inputs(self):
        plan = await self.manager._preflight_prepare(self.instance, None)
        self.assertTrue(plan["dependencies"]["passed"])
        self.assertEqual(plan["request"]["requirements_files"], ["requirements.txt"])
        self.assertNotIn(self.invalid_name, plan["requirements_file_versions"])
        self.assertFalse(Path(plan["venv_root"]).exists())
        self.assertFalse(self.store.read(plan["parse_operation_ref"])["isolation"]["network"])

    async def test_recursive_requirement_and_constraint_versions_are_bound_before_install(self):
        self.instance["inventory_entry"]["requirements_file"] = "dependencies/requirements.txt"
        directory = self.root / "dependencies"
        directory.mkdir()
        (directory / "requirements.txt").write_text("-r nested.txt\n-c constraints.txt\n", encoding="utf-8")
        (directory / "nested.txt").write_text("packaging>=24\n", encoding="utf-8")
        (directory / "constraints.txt").write_text("packaging==26.3\n", encoding="utf-8")
        plan = await self.manager._preflight_prepare(self.instance, None)
        for name in ("requirements.txt", "nested.txt", "constraints.txt"):
            relative = "dependencies/" + name
            self.assertEqual(plan["requirements_file_versions"][relative], plan["file_versions"][relative])
        self.assertTrue(any(item["constraint"] for item in plan["dependencies"]["requirements"]))
        self.assertEqual(self.manager.revalidate(self.instance, plan["plan_id"]), plan)
        (directory / "constraints.txt").write_text("packaging==25.0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "文件版本"):
            self.manager.revalidate(self.instance, plan["plan_id"])
        self.assertFalse(Path(plan["venv_root"]).exists())

    async def test_cancellation_during_real_parser_retains_process_and_plan_records(self):
        self.instance["inventory_entry"]["requirements_file"] = "long-requirements.txt"
        (self.root / "long-requirements.txt").write_text("packaging\n" * 100000, encoding="utf-8")
        task = asyncio.create_task(self.manager.prepare(self.instance))
        operation = None
        try:
            for _ in range(5000):
                reference = self.instance.get("environment_operation_ref")
                plan = self.store.read(reference, {}) if reference else {}
                operation = self.store.read(plan["parse_operation_ref"], {}) if plan.get("parse_operation_ref") else {}
                if operation.get("status") == "running":
                    break
                if task.done():
                    await task
                    self.fail("真实解析进程未进入运行状态")
                await asyncio.sleep(0.002)
            self.assertEqual(operation.get("status"), "running")
            self.assertIsNotNone(recorded_process(operation))
            self.assertTrue(await self.manager.cancel())
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        plan = self.saved_plan()
        self.assertEqual(plan["status"], "cancelled")
        operation = self.store.read(plan["parse_operation_ref"])
        self.assertEqual(operation["status"], "cancelled")
        self.assertIsNone(recorded_process(operation))
        self.assertFalse(Path(plan["venv_root"]).exists())
        self.assertNotIn(self.instance["instance_id"], self.manager._preparing)
        self.assertIsNone(self.manager._active_task)
        self.assertIsNone(self.manager.recover(self.instance))

    async def test_cancellation_waits_for_real_snapshot_without_background_state_changes(self):
        inputs = self.root / "staging-inputs"
        inputs.mkdir()
        for index in range(32):
            (inputs / f"input-{index:03}.bin").write_bytes(b"x" * 1024 * 1024)
        plans = self.store.path(f"instances/{self.instance['instance_id']}/environment-plans")
        task = asyncio.create_task(self.manager.prepare(self.instance))
        try:
            for _ in range(1000):
                if list(plans.glob("*/input-snapshot")):
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(list(plans.glob("*/input-snapshot")))
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in plans.glob("*.json")]
        self.assertEqual(len(records), 1)
        plan = records[0]
        self.assertEqual(plan["status"], "cancelled")
        self.assertNotIn("parse_operation_ref", plan)
        self.assertFalse(Path(plan["venv_root"]).exists())
        self.assertEqual(len(list((Path(plan["snapshot_root"]) / "staging-inputs").glob("*.bin"))), 32)
        saved = self.store.path(f"instances/{self.instance['instance_id']}/instance.json")
        content = saved.read_bytes()
        await asyncio.sleep(0.1)
        self.assertEqual(saved.read_bytes(), content)

    @unittest.skipUnless(os.environ.get("IANVS_RUNTIME_EVIDENCE_ROOT"), "需要已有真实 ready 环境记录，只读保留环境与验证状态")
    async def test_actual_ready_environment_and_verification_survive_syntax_failure(self):
        original_root = Path(os.environ["IANVS_RUNTIME_EVIDENCE_ROOT"])
        instance_id = os.environ.get("IANVS_RUNTIME_EVIDENCE_INSTANCE", "instance-0003")
        original_path = original_root / f"instances/{instance_id}/instance.json"
        original_bytes = original_path.read_bytes()
        original = json.loads(original_bytes)
        self.assertEqual(original["environment"]["status"], "ready")
        environment_path = original_root / f"environments/{original['environment']['environment_id']}/environment.json"
        environment_bytes = environment_path.read_bytes()
        self.instance["environment"] = copy.deepcopy(original["environment"])
        for key in ("status", "needs_verification", "verification"):
            if key in original:
                self.instance[key] = copy.deepcopy(original[key])
        self.declare_invalid()
        before = copy.deepcopy(self.instance)
        with self.assertRaisesRegex(ValueError, "依赖声明无法解析"):
            await self.manager.prepare(self.instance)
        saved = self.store.read(f"instances/{self.instance['instance_id']}/instance.json")
        for key in ("environment", "status", "needs_verification", "verification"):
            self.assertEqual(saved.get(key), before.get(key))
        self.assertFalse((self.store.root / "environments").exists())
        self.assertEqual(original_path.read_bytes(), original_bytes)
        self.assertEqual(environment_path.read_bytes(), environment_bytes)


if __name__ == "__main__":
    unittest.main()
