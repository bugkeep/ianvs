# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import psutil

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings
from ianvs_agent.runtime.environment import EnvironmentManager
from ianvs_agent.runtime.instance_deletion import InstanceDeletionManager
from ianvs_agent.runtime.instances import InstanceManager
from ianvs_agent.storage.local_store import LocalStore


class InstanceDeletionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        project = Path(__file__).resolve().parents[2]
        parent = Path(os.environ.get("IANVS_INSTANCE_DELETION_TEST_ROOT", project / "workspace/ianvs-agent/test-artifacts/instance-deletion"))
        self.root = parent / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        self.source = self.root / "source"
        shutil.copytree(project / "examples/llm_simple_qa", self.source / "examples/llm_simple_qa", ignore=shutil.ignore_patterns("workspace", "__pycache__"))
        self.settings = AgentSettings(self.source, self.root / "state", python_executable=sys.executable)
        self.store = LocalStore(self.settings.state_root)
        self.events = EventBus(self.store)
        self.manager = InstanceManager(self.settings, self.store, self.events)
        self.first = self.manager.create("examples/llm_simple_qa/benchmarkingjob.yaml")
        self.second = self.manager.create("examples/llm_simple_qa/benchmarkingjob.yaml")

    async def test_delete_all_real_venvs_and_records_preserves_other_instance_and_shared_files(self):
        environment = EnvironmentManager(self.settings, self.store, self.events)
        older = await environment.prepare(self.first, install=False)
        current = await environment.prepare(self.manager.get(self.first["instance_id"]), install=False)
        self.assertNotEqual(older["environment_id"], current["environment_id"])
        for value in (older, current):
            probe = subprocess.run([value["python_executable"], "-c", "print('independent environment')"], capture_output=True, text=True, check=True)
            self.assertEqual(probe.stdout.strip(), "independent environment")
        identity = self.first["instance_id"]
        self.store.write(f"instances/{identity}/sessions/session-0001/state.json", {"status": "interrupted"})
        self.store.write(f"instances/{identity}/changes/change-0001/change.json", {"changes": []})
        self.store.write("last_session.json", {"instance_id": identity, "path": f"instances/{identity}/sessions/session-0001"})
        self.store.write("quarantine/test-private.json", {"retained": True})
        shared = {str(path): path.read_bytes() for folder in (self.store.root / "original-goals", self.store.root / "quarantine", self.source, self.store.root / "instances" / self.second["instance_id"]) for path in folder.rglob("*") if path.is_file()}
        counters = self.store.read("counters.json")
        events = (self.store.root / "events.jsonl").read_bytes()
        preview = self.manager.preview_deletion(identity)
        self.assertEqual(preview["environment_ids"], sorted([older["environment_id"], current["environment_id"]]))
        result = self.manager.delete(identity)
        self.assertEqual(result["status"], "deleted")
        self.assertFalse((self.store.root / "instances" / identity).exists())
        for value in (older, current):
            self.assertFalse(Path(value["venv_root"]).parent.exists())
        self.assertEqual(self.store.read("last_session.json"), {})
        self.assertEqual(self.store.read("counters.json"), counters)
        self.assertEqual((self.store.root / "events.jsonl").read_bytes(), events)
        for name, original in shared.items():
            self.assertEqual(Path(name).read_bytes(), original)
        restarted = InstanceManager(self.settings, LocalStore(self.store.root), EventBus(self.store))
        self.assertEqual([item["instance_id"] for item in restarted.list_instances()], [self.second["instance_id"]])
        self.assertEqual(restarted.recover_pending_deletions(), [])
        self.assertEqual(restarted.delete(identity), result)

    async def test_foreign_last_session_is_preserved(self):
        pointer = {"instance_id": self.second["instance_id"], "path": "second-session"}
        self.store.write("last_session.json", pointer)
        self.manager.delete(self.first["instance_id"])
        self.assertEqual(self.store.read("last_session.json"), pointer)

    async def test_real_run_and_operation_processes_prevent_deletion(self):
        identity = self.first["instance_id"]
        for marker, value, reference in (
            ("IANVS_RUN_ID", "run-live", f"instances/{identity}/runs/run-live/run.json"),
            ("IANVS_OPERATION_ID", "operation-live", f"instances/{identity}/environment-commands/operation-live/command.json"),
            ("IANVS_OPERATION_ID", "patch-live", f"instances/{identity}/changes/patch-live/apply-operation.json"),
            ("IANVS_OPERATION_ID", "resolve-live", f"instances/{identity}/environment-plans/resolve-live/resolve-operation.json"),
            ("IANVS_OPERATION_ID", "parse-live", f"instances/{identity}/environment-plans/parse-live/parse/operation.json"),
            ("IANVS_OPERATION_ID", "inspect-live", f"instances/{identity}/environment-checks/inspect-live/operation.json"),
        ):
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=self.first["code_root"], env={**os.environ, marker: value})
            try:
                record = {"instance_id": identity, "pid": process.pid, "process_created_at": psutil.Process(process.pid).create_time(), "status": "running", "run_id" if marker == "IANVS_RUN_ID" else "operation_id": value}
                self.store.write(reference, record)
                with self.assertRaises(RuntimeError):
                    self.manager.delete(identity)
                self.assertTrue((self.store.root / "instances" / identity).is_dir())
            finally:
                process.terminate()
                process.wait(timeout=10)
        self.assertEqual(self.manager.delete(identity)["status"], "deleted")

    async def test_restart_finishes_actual_staged_directories(self):
        identity = self.first["instance_id"]
        deletion = InstanceDeletionManager(self.manager)
        preview = deletion.preview(identity)
        reference = preview["deletion_ref"]
        record = {"schema_version": 1, "instance_id": identity, "name": preview["name"], "environment_ids": [], "status": "deleting", "targets": deletion._targets(identity, [])}
        self.store.write(reference, record)
        target = record["targets"][0]
        staged = self.store.root / target["staged_ref"]
        staged.parent.mkdir(parents=True)
        os.replace(self.store.root / target["source_ref"], staged)
        restarted = InstanceManager(self.settings, LocalStore(self.store.root), EventBus(self.store))
        results = restarted.recover_pending_deletions()
        self.assertEqual(results[0]["status"], "deleted")
        self.assertFalse(staged.exists())
        self.assertEqual(self.store.read(reference)["status"], "deleted")

    async def test_restart_continues_actual_partial_cleanup_without_instance_metadata(self):
        identity = self.first["instance_id"]
        deletion = InstanceDeletionManager(self.manager)
        preview = deletion.preview(identity)
        record = {"schema_version": 1, "instance_id": identity, "name": preview["name"], "environment_ids": [], "status": "deleting", "targets": deletion._targets(identity, [])}
        record["targets"][0]["status"] = "staged"
        self.store.write(preview["deletion_ref"], record)
        staged = self.store.root / record["targets"][0]["staged_ref"]
        staged.parent.mkdir(parents=True)
        os.replace(self.store.root / record["targets"][0]["source_ref"], staged)
        (staged / "instance.json").unlink()
        self.assertTrue((staged / "code").is_dir())
        self.assertTrue(self.manager.preview_deletion(identity)["retry"])
        restarted = InstanceManager(self.settings, LocalStore(self.store.root), EventBus(self.store))
        self.assertEqual(restarted.recover_pending_deletions()[0]["status"], "deleted")
        self.assertFalse(staged.exists())

    async def test_retry_preview_rejects_actual_active_process(self):
        identity = self.first["instance_id"]
        deletion = InstanceDeletionManager(self.manager)
        record = {"schema_version": 1, "instance_id": identity, "name": identity, "environment_ids": [], "status": "failed", "targets": deletion._targets(identity, [])}
        self.store.write(f"instance-deletions/{identity}/deletion.json", record)
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=self.first["code_root"], env={**os.environ, "IANVS_RUN_ID": "run-preview-live"})
        try:
            self.store.write(f"instances/{identity}/runs/run-preview-live/run.json", {"run_id": "run-preview-live", "instance_id": identity, "pid": process.pid, "process_created_at": psutil.Process(process.pid).create_time(), "status": "running"})
            with self.assertRaises(RuntimeError):
                self.manager.preview_deletion(identity)
        finally:
            process.terminate()
            process.wait(timeout=10)
        self.assertTrue(self.manager.preview_deletion(identity)["retry"])

    async def test_actual_move_collision_is_retryable_and_recovery_does_not_stop_other_instances(self):
        identity = self.first["instance_id"]
        deletion = InstanceDeletionManager(self.manager)
        record = {"schema_version": 1, "instance_id": identity, "name": identity, "environment_ids": [], "status": "prepared", "targets": deletion._targets(identity, [])}
        reference = f"instance-deletions/{identity}/deletion.json"
        self.store.write(reference, record)
        staged = self.store.root / record["targets"][0]["staged_ref"]
        staged.parent.mkdir(parents=True)
        staged.write_text("真实文件冲突", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            deletion.delete(identity)
        self.assertEqual(self.store.read(reference)["status"], "failed")
        self.assertEqual(deletion.recover()[0]["status"], "failed")
        self.assertEqual(self.manager.get(self.second["instance_id"])["instance_id"], self.second["instance_id"])
        staged.unlink()
        self.assertEqual(deletion.delete(identity)["status"], "deleted")

    async def test_symlink_root_and_invalid_identity_are_rejected(self):
        identity = self.first["instance_id"]
        outside = self.root / "outside"
        outside.mkdir()
        marker = outside / "retained.txt"
        marker.write_text("保留", encoding="utf-8")
        code = self.store.root / "instances" / identity
        moved = self.root / "saved-instance"
        os.replace(code, moved)
        try:
            code.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            os.replace(moved, code)
            self.skipTest("当前 Windows 未授权创建实际符号链接：" + str(error))
        try:
            with self.assertRaises(PermissionError):
                self.manager.delete(identity)
            self.assertEqual(marker.read_text(encoding="utf-8"), "保留")
        finally:
            code.unlink()
            os.replace(moved, code)
        with self.assertRaises(ValueError):
            self.manager.delete("../outside")

    async def test_environment_with_conflicting_owner_is_rejected(self):
        identity = self.first["instance_id"]
        instance = self.manager.get(identity)
        instance["environment"] = {"environment_id": "environment-conflict"}
        self.store.write(f"instances/{identity}/instance.json", instance)
        self.store.write("environments/environment-conflict/environment.json", {"environment_id": "environment-conflict", "instance_id": self.second["instance_id"]})
        with self.assertRaises(ValueError):
            self.manager.delete(identity)
        self.assertTrue((self.store.root / "instances" / identity).is_dir())
        self.assertTrue(self.store.path("environments/environment-conflict/environment.json").is_file())

    async def test_failed_candidate_environment_and_declared_partial_venv_are_deleted(self):
        identity = self.first["instance_id"]
        self.settings.python_executable = str(self.root / "missing-python")
        environment = EnvironmentManager(self.settings, self.store, self.events)
        with self.assertRaises((OSError, RuntimeError)):
            await environment.prepare(self.first, install=False)
        failed = self.manager.get(identity)["environment"]
        self.assertEqual(failed["status"], "failed")
        candidate = self.store.allocate_id("environment")
        self.store.write(f"instances/{identity}/environment-plans/plan-interrupted.json", {"plan_id": "plan-interrupted", "instance_id": identity, "candidate_environment_id": candidate})
        partial = self.store.root / "environments" / candidate / "venv"
        partial.mkdir(parents=True)
        (partial / "partial.txt").write_text("实际中断时留下的目录", encoding="utf-8")
        result = self.manager.delete(identity)
        self.assertEqual(result["environment_ids"], sorted([failed["environment_id"], candidate]))
        self.assertFalse(partial.exists())
        self.assertFalse((self.store.root / "environments" / failed["environment_id"]).exists())

    async def test_internal_symlink_does_not_remove_external_content(self):
        outside = self.root / "outside"
        outside.mkdir()
        marker = outside / "retained.txt"
        marker.write_text("保留外部内容", encoding="utf-8")
        link = Path(self.first["code_root"]) / "external-link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest("当前 Windows 未授权创建实际符号链接：" + str(error))
        self.manager.delete(self.first["instance_id"])
        self.assertEqual(marker.read_text(encoding="utf-8"), "保留外部内容")


if __name__ == "__main__":
    unittest.main()
