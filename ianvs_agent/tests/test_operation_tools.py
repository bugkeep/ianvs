# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
import unittest
from pathlib import Path

import yaml

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.runtime.processes import recorded_command
from ianvs_agent.storage.local_store import LocalStore


class OperationToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        project = Path(__file__).resolve().parents[2]
        base = Path(os.environ.get("IANVS_OPERATION_TOOL_TEST_ROOT", project / "workspace/operation-tool-tests"))
        self.root = base / LocalStore(base).allocate_id("tool-case")
        source = self.root / "project"
        example = source / "examples/operation"
        example.mkdir(parents=True)
        (source / "README.md").write_text("# 环境操作检查\n", encoding="utf-8")
        (example / "README.md").write_text("# Example\n", encoding="utf-8")
        (example / "benchmarkingjob.yaml").write_text(yaml.safe_dump({"benchmarkingjob": {"name": "operation", "workspace": "workspace/output", "testenv": "examples/operation/testenv.yaml", "test_object": []}}), encoding="utf-8")
        (example / "testenv.yaml").write_text(yaml.safe_dump({"testenv": {"metrics": []}}), encoding="utf-8")
        profile = ModelProfile("操作工具检查", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3")
        settings = AgentSettings(source, self.root / "state", python_executable=sys.executable, profiles={"ark": profile}, approval_policy="auto_scoped")
        self.harness = AgentHarness(settings)
        await self.harness.create_instance("examples/operation/benchmarkingjob.yaml")
        self.harness._ensure_task("检查当前实例环境与具体操作审批", kind="maintenance")
        self.code = Path(self.harness.current_instance["code_root"])

    async def asyncTearDown(self):
        await self.harness.close()

    async def execute(self, name, arguments):
        return await self.harness.tools.execute({"id": "check-" + name, "name": name, "arguments": arguments})

    def environment(self):
        instance = self.harness.current_instance
        environment = {"environment_id": "environment-check", "instance_id": instance["instance_id"], "generation": 1, "status": "ready", "python_executable": sys.executable}
        instance["environment"] = environment
        self.harness.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        self.harness.store.write("environments/environment-check/environment.json", environment)

    async def test_manual_approval_rejects_source_change(self):
        self.environment()
        self.harness.settings.approval_policy = "manual"
        task = asyncio.create_task(self.execute("execute_environment_command", {"argv": ["python", "-c", "print('operation')"]}))
        for _ in range(200):
            if self.harness.tools.pending:
                break
            await asyncio.sleep(0.02)
        self.assertTrue(self.harness.tools.pending)
        approval_id, item = next(iter(self.harness.tools.pending.items()))
        resolved = item["record"]["resolved_operation"]
        self.assertEqual(resolved["python_executable"], sys.executable)
        self.assertIn("source_revision", resolved)
        (self.code / "examples/operation/README.md").write_text("# 修改后的输入\n", encoding="utf-8")
        self.harness.tools.approve(approval_id, True)
        result = await task
        self.assertEqual(result["status"], "error")
        self.assertIn("发生变化", result["error"])
        self.assertEqual(list(self.harness.settings.state_root.glob("instances/*/environment-commands/*/command.json")), [])
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_install_action_permissions_and_apply_parameters(self):
        self.assertTrue(self.harness.tools.is_read_only("install_dependencies", {"action": "plan"}))
        self.assertFalse(self.harness.tools.is_read_only("install_dependencies", {"action": "apply"}))
        result = await self.execute("install_dependencies", {"action": "apply", "plan_id": "environment-plan-0001", "packages": ["packaging"]})
        self.assertEqual(result["status"], "error")
        self.assertIn("仅接受", result["error"])
        missing = await self.execute("install_dependencies", {})
        self.assertEqual(missing["status"], "error")
        self.assertIn("required property", missing["error"])
        self.harness._update_task_progress([missing])
        self.assertFalse(self.harness._maintenance_is_complete())
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_credentials_store_only_authorized_references(self):
        name = "IANVS_OPERATION_CHECK_CREDENTIAL"
        self.harness.settings.runtime_credentials = {name: name}
        value = "private-operation-check-" + self.root.name
        os.environ[name] = value
        try:
            result = await self.execute("configure_runtime_environment", {"credential_names": [name]})
            self.assertEqual(result["status"], "success", result.get("error"))
            instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
            self.assertEqual(instance["credential_bindings"], {name: name})
            self.assertTrue(instance["needs_verification"])
            for path in self.harness.settings.state_root.rglob("*"):
                if path.is_file():
                    self.assertNotIn(value.encode(), path.read_bytes(), str(path))
        finally:
            os.environ.pop(name)
        refused = await self.execute("configure_runtime_environment", {"credential_names": ["UNAUTHORIZED_KEY"]})
        self.assertEqual(refused["status"], "error")
        self.assertIn("尚未获", refused["error"])

    async def test_environment_evidence_ownership(self):
        instance_id = self.harness.current_instance["instance_id"]
        reference = f"instances/{instance_id}/environment-plans/environment-plan-0001.json"
        self.harness.store.write(reference, {"instance_id": instance_id, "evidence_ref": reference, "status": "planned"})
        result = await self.execute("read_evidence", {"reference": reference})
        self.assertEqual(result["status"], "success", result.get("error"))
        self.harness.store.write(reference, {"instance_id": "instance-foreign", "evidence_ref": reference})
        refused = await self.execute("read_evidence", {"reference": reference})
        self.assertEqual(refused["status"], "error")
        self.assertIn("只能读取", refused["error"])

    async def test_registered_actual_pip_report_is_readable_only_by_owner(self):
        instance_id = self.harness.current_instance["instance_id"]
        environment_id = self.harness.store.allocate_id("environment")
        report_ref = f"environments/{environment_id}/pip-output/install-report.json"
        report = self.harness.store.path(report_ref)
        report.parent.mkdir(parents=True)
        await recorded_command([sys.executable, "-m", "pip", "install", "--dry-run", "--no-index", "--report", str(report), "packaging"], self.code, sys.executable, report.parent / "pip.log")
        environment_ref = f"environments/{environment_id}/environment.json"
        environment = {"environment_id": environment_id, "instance_id": instance_id, "pip_report_ref": report_ref}
        self.harness.store.write(environment_ref, environment)
        result = await self.execute("read_evidence", {"reference": report_ref})
        self.assertEqual(result["status"], "success", result.get("error"))
        environment["instance_id"] = "instance-foreign"
        self.harness.store.write(environment_ref, environment)
        result = await self.execute("read_evidence", {"reference": report_ref})
        self.assertEqual(result["status"], "error")

    async def test_failed_inspect_exposes_registered_operation_and_log(self):
        self.harness.settings.python_executable = str(self.root / "missing-python")
        result = await self.execute("inspect_environment", {})
        self.assertEqual(result["status"], "error")
        instance = self.harness.instances.get(self.harness.current_instance["instance_id"])
        check = self.harness.store.read(instance["environment_check_ref"])
        self.assertEqual(check["phase"], "preflight")
        self.assertFalse(check["passed"])
        self.assertIsNone(check["exit_code"])
        for reference in (check["operation_ref"], check["log_ref"]):
            result = await self.execute("read_evidence", {"reference": reference})
            self.assertEqual(result["status"], "success", result.get("error"))


if __name__ == "__main__":
    unittest.main()
