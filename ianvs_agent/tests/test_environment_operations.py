# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import psutil

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings
from ianvs_agent.runtime.environment import EnvironmentManager
from ianvs_agent.runtime.instances import InstanceManager
from ianvs_agent.storage.local_store import LocalStore
from ianvs_agent.tools.operations import check_summary


class EnvironmentOperationsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.project = Path(__file__).resolve().parents[2]
        control_root = Path(os.environ.get("IANVS_ENVIRONMENT_OPERATIONS_ROOT", self.project / "workspace/environment-operations-tests"))
        control = LocalStore(control_root)
        self.case_root = control_root / control.allocate_id("operation-case")
        source = self.case_root / "source"
        benchmark = "examples/Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning/benchmarkingjob.yaml"
        for relative in (benchmark, ".github/workflows/validator/data/example_inventory.yaml"):
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.project / relative, destination)
        self.settings = AgentSettings(project_root=source, state_root=self.case_root / "state", python_executable=sys.executable)
        self.store = LocalStore(self.settings.state_root)
        self.events = EventBus(self.store, task_id="environment-operations-test")
        self.instance = InstanceManager(self.settings, self.store, self.events).create(benchmark)
        self.root = Path(self.instance["code_root"])
        wheel_root = os.environ.get("IANVS_ENVIRONMENT_WHEEL_ROOT")
        if wheel_root:
            wheel = Path(wheel_root) / "packaging-26.3-py3-none-any.whl"
            destination = self.root / "wheels" / wheel.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(wheel, destination)
        self.manager = EnvironmentManager(self.settings, self.store, self.events)
        self.environment = await self.manager.prepare(self.instance, install=False)

    def dependency_file(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    async def healthy_install(self):
        wheel_root = os.environ.get("IANVS_ENVIRONMENT_WHEEL_ROOT")
        if not wheel_root:
            self.skipTest("需要真实 packaging26.3 wheel 缓存，请设置 IANVS_ENVIRONMENT_WHEEL_ROOT")
        wheel = Path(wheel_root) / "packaging-26.3-py3-none-any.whl"
        self.assertTrue(wheel.is_file())
        destination = self.root / "wheels" / wheel.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(wheel, destination)
        self.dependency_file("dependencies/requirements.txt", "--no-index\n--find-links ./wheels\npackaging==26.3\n")
        plan = await self.manager.plan_install(self.instance, requirements_files=["dependencies/requirements.txt"], packages=["packaging==26.3"], modules=["packaging"])
        environment = await self.manager.install(self.instance, plan["plan_id"])
        self.assertEqual(environment["status"], "ready")
        self.assertTrue(environment["postcheck"]["passed"])
        self.assertEqual(next(item["version"] for item in environment["packages"] if item["name"].lower() == "packaging"), "26.3")
        return environment

    async def test_inspection_records_actual_missing_module_and_task_identity(self):
        check = await self.manager.inspect(self.instance, modules=["json", "ianvs_environment_dependency_that_does_not_exist"])
        self.assertFalse(check["passed"])
        self.assertTrue(check["pip_check"]["passed"])
        self.assertTrue(check["imports"][0]["passed"])
        self.assertFalse(check["imports"][1]["passed"])
        self.assertEqual(check["environment_id"], self.environment["environment_id"])
        self.assertEqual(check["task_id"], "environment-operations-test")
        self.assertEqual(self.store.read(check["evidence_ref"]), check)
        self.assertEqual(check["isolation"]["backend"], "bubblewrap")
        self.assertTrue(check["isolation"]["private_home"])
        self.assertEqual(check["log_ref"], str(self.store.path(check["evidence_ref"]).parent / "worker-output/output.log"))
        self.assertNotIn(str(self.store.path(check["evidence_ref"]).parent), check["isolation"]["writable_paths"])

    async def test_prepare_plan_is_stable_and_executes_bound_candidate(self):
        plan = self.manager.preview_prepare(self.instance)
        self.assertEqual(self.manager.preview_prepare(self.instance), plan)
        self.assertEqual(self.manager.revalidate(self.instance, plan["plan_id"]), plan)
        prepared = await self.manager.prepare(self.instance, install=False, plan_id=plan["plan_id"])
        self.assertEqual(prepared["environment_id"], plan["candidate_environment_id"])
        self.assertEqual(prepared["steps"][0]["command"], plan["commands"][0]["command"])
        self.assertEqual(self.store.read(plan["evidence_ref"])["status"], "completed")

    async def test_cancel_plan_waits_for_actual_snapshot_and_persists_cancellation(self):
        inputs = self.root / "staging-inputs"
        inputs.mkdir()
        for index in range(64):
            (inputs / f"input-{index:03}.bin").write_bytes(b"x" * 1024 * 1024)
        plans = self.store.path(f"instances/{self.instance['instance_id']}/environment-plans")
        operation = asyncio.create_task(self.manager.plan_install(self.instance, packages=["packaging==26.3"]))
        for _ in range(1000):
            if list(plans.glob("*/input-snapshot")):
                break
            await asyncio.sleep(0.001)
        self.assertTrue(list(plans.glob("*/input-snapshot")), "真实计划快照线程未启动")
        self.assertFalse(operation.done(), "应在快照准备阶段取消")
        operation.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await operation
        records = [json.loads(path.read_text(encoding="utf-8")) for path in plans.glob("*.json")]
        self.assertEqual(len(records), 1)
        plan = records[0]
        self.assertEqual(plan["status"], "cancelled")
        self.assertTrue(plan["completed_at"])
        self.assertNotIn("parse_operation_ref", plan)
        self.assertEqual(len(list((Path(plan["snapshot_root"]) / "staging-inputs").glob("*.bin"))), 64)
        self.assertNotIn(self.instance["instance_id"], self.manager._preparing)
        self.assertIsNone(self.manager._active_task)
        saved = self.store.path(f"instances/{self.instance['instance_id']}/instance.json").read_bytes()
        await asyncio.sleep(0.1)
        self.assertEqual(self.store.path(f"instances/{self.instance['instance_id']}/instance.json").read_bytes(), saved)
        self.assertIsNone(self.manager.recover(self.instance))

    async def test_recursive_requirements_constraints_and_invalid_actual_example(self):
        if not os.environ.get("IANVS_ENVIRONMENT_WHEEL_ROOT"):
            self.skipTest("需要真实 wheel 缓存完成 pip 解析")
        self.dependency_file("dependencies/requirements.txt", "--no-index\n--find-links ./wheels\n-r nested.txt\n-c constraints.txt\n")
        self.dependency_file("dependencies/nested.txt", "packaging>=24\n")
        self.dependency_file("dependencies/constraints.txt", "packaging==26.3\n")
        plan = await self.manager.plan_install(self.instance, requirements_files=["dependencies/requirements.txt"])
        self.assertTrue(plan["requires_install"])
        self.assertEqual(set(plan["requirements_file_versions"]), {"dependencies/requirements.txt", "dependencies/nested.txt", "dependencies/constraints.txt"})
        self.assertTrue(any(item["constraint"] for item in plan["dependencies"]["requirements"]))
        self.assertIn("--dry-run", plan["resolve_operation"]["command"])
        self.assertEqual(plan["resolved_packages"], {"packaging": "==26.3"})
        self.assertTrue(self.store.path(plan["constraints_ref"]).is_file())
        self.assertIn("wheels/packaging-26.3-py3-none-any.whl", plan["file_versions"])
        constraints_path = self.store.path(plan["constraints_ref"])
        constraints_path.write_text("packaging==25.0\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "正式解析材料"):
            self.manager.revalidate(self.instance, plan["plan_id"])
        actual = self.project / "examples/Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning/requirement.txt"
        shutil.copy2(actual, self.root / "invalid-requirements.txt")
        with self.assertRaises(ValueError) as failure:
            await self.manager.plan_install(self.instance, requirements_files=["invalid-requirements.txt"])
        self.assertIn("dashscope", str(failure.exception))
        self.assertIn("plan_id=environment-plan-", str(failure.exception))
        self.assertIn("evidence_ref=instances/", str(failure.exception))
        plans = list((self.settings.state_root / f"instances/{self.instance['instance_id']}/environment-plans").glob("*.json"))
        invalid = [json.loads(path.read_text(encoding="utf-8")) for path in plans if json.loads(path.read_text(encoding="utf-8"))["status"] == "invalid"]
        self.assertEqual(len(invalid), 1)
        self.assertTrue(any("dashscope" in item["error"] for item in invalid[0]["dependencies"]["errors"]))

    async def test_install_persists_actual_generation_packages_checks_and_reuses(self):
        environment = await self.healthy_install()
        self.assertEqual(environment["generation"], self.environment["generation"] + 1)
        self.assertTrue(self.instance["needs_verification"])
        plan = self.store.read(f"instances/{self.instance['instance_id']}/environment-plans/{environment['plan_id']}.json")
        self.assertEqual(plan["status"], "completed")
        self.assertEqual(plan["postcheck_ref"], environment["postcheck"]["evidence_ref"])
        reloaded = self.store.read(f"instances/{self.instance['instance_id']}/instance.json")
        manager = EnvironmentManager(self.settings, self.store, self.events)
        self.assertIsNone(manager.recover(reloaded))
        reused = await manager.prepare(reloaded)
        self.assertEqual(reused["environment_id"], environment["environment_id"])
        self.assertEqual(reused["packages"], environment["packages"])

    async def test_failed_candidate_keeps_actual_healthy_environment(self):
        healthy = await self.healthy_install()
        old_python = Path(healthy["python_executable"])
        plan = await self.manager.plan_install(self.instance, modules=["ianvs_environment_dependency_that_does_not_exist"])
        with self.assertRaises(RuntimeError) as failure:
            await self.manager.install(self.instance, plan["plan_id"])
        self.assertIn("ianvs_environment_dependency_that_does_not_exist", str(failure.exception))
        self.assertIn("plan_id=" + plan["plan_id"], str(failure.exception))
        self.assertIn("evidence_ref=", str(failure.exception))
        self.assertEqual(self.instance["environment"]["environment_id"], healthy["environment_id"])
        self.assertEqual(self.instance["environment"]["status"], "ready")
        self.assertTrue(old_python.is_file())
        failed = self.store.read(f"environments/{plan['candidate_environment_id']}/environment.json")
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["postcheck"]["passed"])
        self.assertTrue(failed["postcheck"]["pip_check"]["passed"])
        self.assertEqual(self.store.read(plan["evidence_ref"])["status"], "failed")

    async def test_readiness_preserves_original_modules_and_recovers_after_real_import_repair(self):
        healthy = await self.healthy_install()
        scope_path = self.store.path(healthy["readiness_ref"])
        original_scope = scope_path.read_bytes()
        scope = self.store.read(healthy["readiness_ref"])
        self.assertEqual(scope["request"]["modules"], ["packaging"])
        self.assertEqual(scope["postcheck_ref"], healthy["postcheck"]["evidence_ref"])
        module = next(Path(healthy["venv_root"]).glob("lib/python*/site-packages/packaging/__init__.py"))
        original_module = module.read_bytes()
        module.write_text("raise ImportError('readiness regression: original module unavailable')\n", encoding="utf-8")
        failed = await self.manager.inspect(self.instance, requirements_files=[], modules=[])
        self.assertEqual(failed["request"]["modules"], ["packaging"])
        self.assertEqual(failed["request"]["requirements_files"], ["dependencies/requirements.txt"])
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["package_drift"])
        self.assertFalse(failed["readiness"]["passed"])
        self.assertEqual(healthy["status"], "modified")
        failed_bytes = self.store.path(failed["evidence_ref"]).read_bytes()
        module.write_bytes(original_module)
        reloaded = self.store.read(f"instances/{self.instance['instance_id']}/instance.json")
        manager = EnvironmentManager(self.settings, self.store, self.events)
        restored = await manager.inspect(reloaded)
        self.assertTrue(restored["passed"])
        self.assertTrue(restored["readiness"]["passed"])
        self.assertEqual(restored["request"]["modules"], ["packaging"])
        self.assertEqual(reloaded["environment"]["status"], "ready")
        self.assertEqual(reloaded["environment"]["generation"], healthy["generation"])
        self.assertEqual(reloaded["environment"]["check_failure_ref"], failed["evidence_ref"])
        self.assertEqual(reloaded["environment"]["readiness_recovery_ref"], restored["evidence_ref"])
        self.assertEqual(scope_path.read_bytes(), original_scope)
        self.assertEqual(self.store.path(failed["evidence_ref"]).read_bytes(), failed_bytes)

    async def test_additional_import_probe_retains_original_readiness_and_failed_evidence(self):
        healthy = await self.healthy_install()
        scope_path = self.store.path(healthy["readiness_ref"])
        original_scope = scope_path.read_bytes()
        probe = "ianvs_environment_dependency_that_does_not_exist"
        failed = await self.manager.inspect(self.instance, modules=[probe])
        self.assertFalse(failed["passed"])
        self.assertTrue(failed["readiness"]["passed"])
        self.assertEqual(failed["request"]["modules"], ["packaging", probe])
        self.assertEqual(healthy["status"], "ready")
        self.assertFalse(healthy["current_check"]["passed"])
        checked = await self.manager.inspect(self.instance, requirements_files=[], modules=[])
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["request"]["modules"], ["packaging"])
        self.assertEqual(healthy["status"], "ready")
        self.assertEqual(scope_path.read_bytes(), original_scope)
        self.assertFalse(self.store.read(failed["evidence_ref"])["passed"])

    async def test_empty_inspection_cannot_remove_original_constraints_or_changed_declarations(self):
        await self.healthy_install()
        constraints = self.dependency_file("dependencies/constraints.txt", "packaging==26.3\n")
        plan = await self.manager.plan_install(self.instance, constraints_files=["dependencies/constraints.txt"])
        healthy = await self.manager.install(self.instance, plan["plan_id"])
        empty_request = {"requirements_files": [], "constraints_files": [], "packages": [], "modules": [], "index_url": None}
        checked = await self.manager.inspect(self.instance, request=empty_request)
        self.assertTrue(checked["passed"])
        for key in ("requirements_files", "constraints_files", "packages", "modules"):
            self.assertEqual(checked["request"][key], plan["request"][key])
        constraints.write_text("packaging==25.0\n", encoding="utf-8")
        changed = await self.manager.inspect(self.instance, request=empty_request)
        self.assertFalse(changed["passed"])
        self.assertFalse(changed["readiness"]["passed"])
        self.assertFalse(changed["package_drift"])
        self.assertEqual(healthy["status"], "modified")
        again = await self.manager.inspect(self.instance, requirements_files=[], modules=[])
        self.assertFalse(again["passed"])
        self.assertEqual(healthy["status"], "modified")

    async def test_additional_requirements_failures_preserve_original_readiness(self):
        healthy = await self.healthy_install()
        scope_path = self.store.path(healthy["readiness_ref"])
        original_scope = scope_path.read_bytes()
        probes = (("extra-missing.txt", "ianvs-readiness-probe-that-does-not-exist==1.0\n"), ("extra-invalid.txt", "packaging==\n"))
        for name, content in probes:
            with self.subTest(requirements=name):
                self.dependency_file(name, content)
                checked = await self.manager.inspect(self.instance, requirements_files=[name], modules=[])
                self.assertFalse(checked["passed"])
                self.assertTrue(checked["readiness"]["passed"])
                self.assertTrue(checked["readiness_dependencies"]["passed"])
                self.assertTrue(all(item.get("satisfied", True) for item in checked["readiness_dependencies"]["requirements"]))
                self.assertEqual(healthy["status"], "ready")
                self.assertEqual(checked["request"]["requirements_files"], ["dependencies/requirements.txt", name])
                self.assertEqual(check_summary(checked)["readiness"], checked["readiness"])
                self.assertEqual(check_summary(checked)["request"], checked["request"])
                self.assertEqual(scope_path.read_bytes(), original_scope)
                self.assertEqual(self.store.read(checked["evidence_ref"]), checked)
        self.assertTrue(checked["dependencies"]["errors"])
        original = await self.manager.inspect(self.instance)
        self.assertTrue(original["passed"])
        self.assertEqual(healthy["status"], "ready")

    async def test_empty_inspection_cannot_restore_actual_package_drift(self):
        healthy = await self.healthy_install()
        python = healthy["python_executable"]
        await self.manager._run_step(self.instance, healthy, "dependency_fault_injection", [python, "-m", "pip", "uninstall", "--yes", "packaging"], self.root, python, self.case_root / "readiness-package-removal.log", writable_paths=[healthy["venv_root"]])
        for _ in range(2):
            check = await self.manager.inspect(self.instance, requirements_files=[], modules=[])
            self.assertFalse(check["passed"])
            self.assertTrue(check["package_drift"])
            self.assertFalse(check["readiness"]["passed"])
            self.assertEqual(check["request"]["modules"], ["packaging"])
            self.assertEqual(healthy["status"], "modified")
        self.assertNotIn("readiness_recovery_ref", healthy)

    async def test_legacy_environment_without_recorded_scope_requires_controlled_preparation(self):
        healthy = await self.healthy_install()
        # 使用真实安装结果验证旧格式环境缺少可用范围字段时的恢复规则。
        healthy.pop("readiness_ref")
        healthy.pop("readiness_fingerprint")
        self.manager._save(self.instance, healthy)
        failed = await self.manager.inspect(self.instance, modules=["ianvs_environment_dependency_that_does_not_exist"])
        self.assertFalse(failed["passed"])
        self.assertEqual(healthy["status"], "modified")
        check = await self.manager.inspect(self.instance, modules=["packaging"])
        self.assertTrue(check["passed"])
        self.assertFalse(check["readiness"]["passed"])
        self.assertIsNone(check["readiness"]["scope_ref"])
        self.assertEqual(healthy["status"], "modified")
        self.assertNotIn("readiness_recovery_ref", healthy)

    async def test_damaged_readiness_scope_invalidates_environment_and_allows_new_candidate(self):
        healthy = await self.healthy_install()
        scope = self.store.read(healthy["readiness_ref"])
        scope["request"]["modules"] = []
        self.store.write(healthy["readiness_ref"], scope)
        damaged_bytes = self.store.path(healthy["readiness_ref"]).read_bytes()
        with self.assertRaisesRegex(RuntimeError, "原始可用范围"):
            await self.manager.inspect(self.instance)
        self.assertEqual(healthy["status"], "modified")
        failed_ref = healthy["check_failure_ref"]
        failed = self.store.read(failed_ref)
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["passed"])
        self.assertEqual(healthy["current_check"]["evidence_ref"], failed_ref)
        self.assertNotIn("command", failed)
        failure_bytes = self.store.path(failed_ref).read_bytes()
        plan = await self.manager.plan_install(self.instance)
        repaired = await self.manager.install(self.instance, plan["plan_id"])
        self.assertEqual(repaired["status"], "ready")
        self.assertEqual(repaired["generation"], healthy["generation"] + 1)
        self.assertNotEqual(repaired["readiness_ref"], healthy["readiness_ref"])
        self.assertEqual(self.manager._readiness_scope(repaired)["request"]["modules"], ["packaging"])
        self.assertTrue((await self.manager.inspect(self.instance))["readiness"]["passed"])
        self.assertEqual(self.store.path(healthy["readiness_ref"]).read_bytes(), damaged_bytes)
        self.assertEqual(self.store.path(failed_ref).read_bytes(), failure_bytes)

    async def test_modified_environment_is_retained_until_candidate_postcheck_passes(self):
        healthy = await self.healthy_install()
        python = healthy["python_executable"]
        await self.manager._run_step(self.instance, healthy, "dependency_fault_injection", [python, "-m", "pip", "uninstall", "--yes", "packaging"], self.root, python, self.case_root / "modified-environment.log", writable_paths=[healthy["venv_root"]])
        observed = await self.manager.inspect(self.instance, modules=["packaging"])
        self.assertTrue(observed["package_drift"])
        self.assertEqual(self.instance["environment"]["status"], "modified")
        original_id = healthy["environment_id"]
        plan = await self.manager.plan_install(self.instance, modules=["ianvs_environment_dependency_that_does_not_exist"])
        with self.assertRaises(RuntimeError):
            await self.manager.install(self.instance, plan["plan_id"])
        failed = self.store.read(f"environments/{plan['candidate_environment_id']}/environment.json")
        self.assertTrue(failed["candidate"])
        self.assertEqual(failed["status"], "failed")
        saved = self.store.read(f"instances/{self.instance['instance_id']}/instance.json")
        self.assertEqual(saved["environment"]["environment_id"], original_id)
        self.assertEqual(saved["environment"]["status"], "modified")
        self.assertNotIn("pending_environment_id", saved)
        self.assertEqual(saved["last_environment_attempt"]["environment_id"], failed["environment_id"])
        self.assertFalse(await self.manager.cancel(saved))
        repaired_plan = await self.manager.plan_install(self.instance, modules=["packaging"])
        repaired = await self.manager.install(self.instance, repaired_plan["plan_id"])
        self.assertTrue(repaired["postcheck"]["passed"])
        self.assertFalse(repaired["candidate"])
        self.assertNotEqual(repaired["environment_id"], original_id)
        self.assertEqual(self.instance["environment"]["environment_id"], repaired["environment_id"])
        self.assertNotIn("pending_environment_id", self.instance)

    async def test_missing_interpreter_preflight_keeps_real_failure_evidence(self):
        python = Path(self.environment["python_executable"])
        python.unlink()
        log = self.case_root / "missing-interpreter.log"
        with self.assertRaises(FileNotFoundError):
            await self.manager._run_step(self.instance, self.environment, "missing_interpreter", [str(python), "-c", "print('not executed')"], self.root, str(python), log)
        step = self.environment["failed_step"]
        self.assertEqual(step["phase"], "preflight")
        self.assertFalse(step["command_executed"])
        self.assertIsNone(step["exit_code"])
        self.assertIsNone(step.get("pid"))
        self.assertTrue(log.is_file())
        self.assertIn(str(python), log.read_text(encoding="utf-8"))
        self.assertIn("FileNotFoundError", log.read_text(encoding="utf-8"))
        with self.assertRaises(FileNotFoundError):
            await self.manager.inspect(self.instance, modules=["json"])
        checks = self.store.path(f"instances/{self.instance['instance_id']}/environment-checks")
        check = json.loads(next(checks.glob("*/check.json")).read_text(encoding="utf-8"))
        self.assertEqual(check["status"], "failed")
        self.assertFalse(check["passed"])
        self.assertEqual(check["phase"], "preflight")
        self.assertFalse(check["command_executed"])
        self.assertIsNone(check["exit_code"])
        self.assertTrue(self.store.path(check["operation_ref"]).is_file())
        self.assertTrue(Path(check["log_ref"]).is_file())
        saved = self.store.read(f"instances/{self.instance['instance_id']}/instance.json")
        self.assertEqual(saved["environment_check_ref"], check["evidence_ref"])
        self.assertEqual(saved["environment"]["status"], "modified")
        self.assertFalse(saved["environment"]["current_check"]["passed"])
        self.assertTrue(saved["needs_verification"])

    async def test_approval_rejects_changed_input_and_cross_instance_path(self):
        if not os.environ.get("IANVS_ENVIRONMENT_WHEEL_ROOT"):
            self.skipTest("需要真实 wheel 缓存完成 pip 解析")
        dependency = self.dependency_file("dependencies.txt", "--no-index\n--find-links ./wheels\npackaging==26.3\n")
        plan = await self.manager.plan_install(self.instance, requirements_files=["dependencies.txt"])
        dependency.write_text("packaging==25.0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "文件版本"):
            self.manager.revalidate(self.instance, plan["plan_id"])
        with self.assertRaises(PermissionError):
            await self.manager.plan_install(self.instance, requirements_files=["../../outside.txt"])
        with self.assertRaises(ValueError):
            self.manager.revalidate(self.instance, "../environment-plan-0001")
        self.assertFalse(Path(plan["venv_root"]).exists())

    async def test_same_package_requirement_replaces_prior_explicit_pin(self):
        await self.healthy_install()
        request = self.manager._request(self.instance, packages=["Packaging>=26.3"])
        self.assertEqual(request["packages"], ["Packaging>=26.3"])
        plan = await self.manager.plan_install(self.instance, packages=["Packaging>=26.3"])
        self.assertEqual(plan["request"]["packages"], ["Packaging>=26.3"])
        self.assertEqual(plan["resolved_packages"], {"packaging": "==26.3"})

    async def test_source_is_readonly_and_cancelled_process_is_finalized(self):
        source = self.dependency_file("source-protection.txt", "protected\n")
        python = self.environment["python_executable"]
        command = [python, "-c", "from pathlib import Path; Path('source-protection.txt').write_text('changed')"]
        with self.assertRaises(RuntimeError):
            await self.manager._run_step(self.instance, self.environment, "source_write_denied", command, self.root, python, self.case_root / "source-denied.log")
        self.assertEqual(source.read_text(encoding="utf-8"), "protected\n")
        operation = asyncio.create_task(self.manager._run_step(self.instance, self.environment, "cancellation", [python, "-c", "import time; time.sleep(60)"], self.root, python, self.case_root / "cancellation.log"))
        for _ in range(100):
            if self.environment["steps"][-1].get("name") == "cancellation" and self.environment["steps"][-1].get("pid"):
                break
            await asyncio.sleep(0.05)
        pid = self.environment["steps"][-1]["pid"]
        operation.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await operation
        step = self.environment["steps"][-1]
        self.assertEqual(step["status"], "cancelled")
        self.assertTrue(step["completed_at"])
        self.assertTrue(not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE)
        saved = self.store.read(f"environments/{self.environment['environment_id']}/environment.json")
        self.assertEqual(saved["steps"][-1]["status"], "cancelled")

    async def test_installed_package_drift_invalidates_environment(self):
        healthy = await self.healthy_install()
        python = healthy["python_executable"]
        await self.manager._run_step(self.instance, healthy, "dependency_fault_injection", [python, "-m", "pip", "uninstall", "--yes", "packaging"], self.root, python, self.case_root / "dependency-removal.log", writable_paths=[healthy["venv_root"]])
        check = await self.manager.inspect(self.instance, modules=["packaging"])
        self.assertFalse(check["passed"])
        self.assertTrue(check["package_drift"])
        self.assertEqual(self.instance["environment"]["status"], "modified")
        self.assertTrue(self.instance["needs_verification"])

    @unittest.skipUnless(os.environ.get("IANVS_INTERRUPTED_ENVIRONMENT_INSTANCE"), "需要实际中断安装的正式实例记录")
    async def test_recover_actual_interrupted_install_in_independent_store(self):
        original_path = Path(os.environ["IANVS_INTERRUPTED_ENVIRONMENT_INSTANCE"])
        original_bytes = original_path.read_bytes()
        original = json.loads(original_bytes)
        environment = original["environment"]
        self.assertEqual(environment["status"], "preparing")
        source_store = original_path.parents[2]
        environment_path = source_store / f"environments/{environment['environment_id']}/environment.json"
        environment_bytes = environment_path.read_bytes()
        self.store.write(f"instances/{original['instance_id']}/instance.json", original)
        self.store.write(f"environments/{environment['environment_id']}/environment.json", json.loads(environment_bytes))
        plan_id = environment["plan_id"]
        plan_reference = f"instances/{original['instance_id']}/environment-plans/{plan_id}.json"
        plan_bytes = (source_store / plan_reference).read_bytes()
        self.store.write(plan_reference, json.loads(plan_bytes))
        recovered = EnvironmentManager(self.settings, self.store, self.events).recover(original)
        self.assertEqual(recovered["status"], "interrupted")
        self.assertTrue(recovered["completed_at"])
        self.assertEqual(self.store.read(plan_reference)["status"], "interrupted")
        self.assertEqual(original_path.read_bytes(), original_bytes)
        self.assertEqual(environment_path.read_bytes(), environment_bytes)
        self.assertEqual((source_store / plan_reference).read_bytes(), plan_bytes)


if __name__ == "__main__":
    unittest.main()
