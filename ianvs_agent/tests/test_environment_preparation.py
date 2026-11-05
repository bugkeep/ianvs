# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import hashlib
import os
import shutil
import sys
import unittest
from pathlib import Path

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings
from ianvs_agent.runtime.environment import EnvironmentManager
from ianvs_agent.runtime.instances import InstanceManager
from ianvs_agent.storage.local_store import LocalStore


class EnvironmentPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.project = Path(__file__).resolve().parents[2]
        control_root = Path(os.environ.get("IANVS_ENVIRONMENT_TEST_ROOT", self.project / "workspace/environment-tests"))
        control = LocalStore(control_root)
        self.case_root = control_root / control.allocate_id("preparation-case")
        source = self.case_root / "source"
        self.benchmark = "examples/Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning/benchmarkingjob.yaml"
        for relative in (self.benchmark, str(Path(self.benchmark).parent / "requirement.txt"), ".github/workflows/validator/data/example_inventory.yaml"):
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.project / relative, target)
        self.settings = AgentSettings(project_root=source, state_root=self.case_root / "state", python_executable=sys.executable)
        self.store = LocalStore(self.settings.state_root)
        self.events = EventBus(self.store)
        self.instance = InstanceManager(self.settings, self.store, self.events).create(self.benchmark)
        self.manager = EnvironmentManager(self.settings, self.store, self.events)

    async def test_created_venv_reports_example_preparation_gaps(self):
        environment = await self.manager.prepare(self.instance, install=False)
        self.assertEqual(environment["status"], "created")
        self.assertTrue(Path(environment["python_executable"]).is_file())
        self.assertEqual(environment["steps"][0]["name"], "create_venv")
        self.assertEqual(environment["steps"][0]["status"], "completed")
        scope = environment["preparation_scope"]
        self.assertEqual(scope["core_dependencies"], "not_installed")
        self.assertEqual(scope["requirements_file"], None)
        self.assertEqual(scope["undeclared_requirements"], [str(Path(self.benchmark).parent / "requirement.txt").replace("\\", "/")])
        self.assertEqual(scope["configuration_status"], "incomplete")
        self.assertEqual(scope["model_data_status"], "not_verified")
        self.assertTrue(any("testenv" in warning for warning in environment["warnings"]))
        kinds = [event["event_type"] for event in self.store.read_events("events.jsonl")]
        self.assertIn("environment.step_started", kinds)
        self.assertIn("environment.step_completed", kinds)
        self.assertIn("environment.completed", kinds)

    async def test_missing_local_dependency_retains_preflight_evidence_without_venv(self):
        with self.assertRaisesRegex(ValueError, "requirements.txt"):
            await self.manager.prepare(self.instance)
        saved = self.store.read(f"instances/{self.instance['instance_id']}/instance.json")
        self.assertIsNone(saved.get("environment"))
        plan = self.store.read(saved["environment_operation_ref"])
        self.assertEqual(plan["status"], "invalid")
        self.assertFalse(plan["dependencies"]["passed"])
        self.assertTrue(any("requirements.txt" in item["error"] for item in plan["dependencies"]["errors"]))
        self.assertFalse(Path(plan["venv_root"]).exists())
        operation = self.store.read(plan["parse_operation_ref"])
        self.assertEqual(operation["status"], "completed")
        self.assertFalse(operation["isolation"]["network"])
        self.assertTrue(Path(operation["log_ref"]).is_file())
        self.assertEqual(self.store.read(plan["parse_result_ref"]), plan["dependencies"])
        kinds = [event["event_type"] for event in self.store.read_events("events.jsonl")]
        self.assertNotIn("environment.started", kinds)
        self.assertNotIn("environment.step_started", kinds)

    async def test_completed_local_pip_command_has_persistent_trace(self):
        environment = await self.manager.prepare(self.instance, install=False)
        log = self.case_root / "pip-list.json"
        result = await self.manager._run_step(self.instance, environment, "inspect_packages", [environment["python_executable"], "-m", "pip", "list", "--format=json", "--disable-pip-version-check"], self.instance["code_root"], environment["python_executable"], log)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(any(package["name"] == "pip" for package in json.loads(log.read_text(encoding="utf-8"))))
        saved = self.store.read(f"environments/{environment['environment_id']}/environment.json")
        self.assertEqual(saved["steps"][-1]["log_ref"], str(log))

    @unittest.skipUnless(os.environ.get("IANVS_READY_ENVIRONMENT_ROOT"), "需要已有真实环境记录，只读检查已安装依赖")
    async def test_reuse_of_recorded_ready_environment_preserves_verification_state(self):
        original_root = Path(os.environ["IANVS_READY_ENVIRONMENT_ROOT"])
        original_path = original_root / "instances/instance-0004/instance.json"
        original = json.loads(original_path.read_text(encoding="utf-8"))
        environment = original["environment"]
        self.assertEqual(environment["status"], "ready")
        environment_path = original_root / f"environments/{environment['environment_id']}/environment.json"
        digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (original_path, environment_path)}
        code_root = Path(self.instance["code_root"])
        for relative in ("requirements.txt", "resources/third_party/sedna-0.6.0.1-py3-none-any.whl"):
            target = code_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(original["code_root"]) / relative, target)
        self.instance["inventory_entry"] = original["inventory_entry"]
        self.instance["environment"] = environment
        self.settings.python_executable = environment["base_python"]
        self.assertEqual(self.manager._dependency_fingerprint(self.instance), environment["dependency_fingerprint"])
        before_status = self.instance["status"]
        before_needs_verification = self.instance["needs_verification"]
        reused = await self.manager.prepare(self.instance)
        self.assertEqual(reused["environment_id"], environment["environment_id"])
        self.assertEqual(reused["preparation_scope"]["core_dependencies"], "installed")
        self.assertIn("2 项", reused["summary"])
        self.assertEqual(self.instance["status"], before_status)
        self.assertEqual(self.instance["needs_verification"], before_needs_verification)
        self.assertTrue((self.settings.state_root / f"environments/{environment['environment_id']}/packages-check.json").is_file())
        for path, digest in digests.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
