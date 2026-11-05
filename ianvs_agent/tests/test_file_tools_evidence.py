# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import AgentSettings, ModelProfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("IANVS_FILE_TOOL_VALIDATION_STATE"), "需要明确指定已有真实实例与验证证据目录")
class ValidationEvidenceToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.source = Path(os.environ["IANVS_FILE_TOOL_VALIDATION_STATE"])
        self.instance_id = os.environ.get("IANVS_FILE_TOOL_VALIDATION_INSTANCE", "instance-0002")
        self.root = PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/validation-tools" / uuid.uuid4().hex[:12]
        base = Path("instances") / self.instance_id
        source_record = self.source / base / "instance.json"
        self.assertTrue(source_record.is_file(), str(source_record))
        instance = json.loads(source_record.read_text(encoding="utf-8"))
        self.assertEqual(instance["environment"]["status"], "ready")
        destination = self.root / base / "instance.json"
        destination.parent.mkdir(parents=True)
        shutil.copy2(source_record, destination)
        shutil.copy2(self.source / "counters.json", self.root / "counters.json")
        self.references = set()
        current = (instance.get("verification") or {}).get("static_report_ref")
        if current:
            self.references.add(current)
        for path in (self.source / base / "runs").glob("*/verification.json"):
            target = self.root / path.relative_to(self.source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            reference = json.loads(path.read_text(encoding="utf-8")).get("static_report_ref")
            if reference:
                self.references.add(reference)
        self.assertTrue(self.references, "正式实例或 Run 必须包含真实 static_report_ref")
        for reference in self.references:
            target = self.root / reference
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source / reference, target)
        profile = ModelProfile("验证证据工具检查", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3")
        self.settings = AgentSettings(PROJECT_ROOT, self.root, python_executable=sys.executable, profiles={"ark": profile}, approval_policy="auto_scoped")
        self.harness = AgentHarness(self.settings)
        await self.harness.restore_instance(self.instance_id)

    async def asyncTearDown(self):
        await self.harness.close()

    async def execute(self, name, arguments):
        result = await self.harness.tools.execute({"id": "call-" + uuid.uuid4().hex[:12], "name": name, "arguments": arguments})
        return result, self.harness.store.read(result["full_output_ref"]) if result.get("truncated") else result["data"]

    async def test_formal_instance_and_run_validation_reports_are_readable(self):
        for reference in sorted(self.references):
            result, data = await self.execute("read_evidence", {"reference": reference, "end_line": 12})
            self.assertEqual(result["status"], "success", result.get("error"))
            self.assertEqual(data["reference"], reference)
            expected = (self.source / reference).read_text(encoding="utf-8").splitlines()
            self.assertEqual(data["content"], "\n".join(f"{index + 1}: {line}" for index, line in enumerate(expected[:12])))
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)

    async def test_actual_validator_tool_registers_only_its_report(self):
        result, validation = await self.execute("validate_instance", {})
        self.assertEqual(result["status"], "success", result.get("error"))
        reference = validation["report_ref"]
        self.assertNotIn(reference, self.references)
        result, data = await self.execute("read_evidence", {"reference": reference, "end_line": 10})
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(data["reference"], reference)
        self.assertTrue(data["content"].startswith("1: {"))
        entry = str(Path(reference).with_name("entry.json"))
        result, _ = await self.execute("read_evidence", {"reference": entry})
        self.assertEqual(result["status"], "error")
        unregistered = f"instances/{self.instance_id}/validations/validation-unregistered/report.json"
        self.harness.store.path(unregistered).parent.mkdir(parents=True)
        shutil.copy2(self.root / reference, self.root / unregistered)
        result, _ = await self.execute("read_evidence", {"reference": unregistered})
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
