# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import os
import shutil
import sys
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class OriginalGoalEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        artifacts = Path(os.environ.get("IANVS_ORIGINAL_GOAL_EVIDENCE_TEST_ROOT", PROJECT_ROOT / "workspace/original-goal-evidence-tests"))
        self.root = artifacts / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        self.source = self.root / "source"
        shutil.copytree(PROJECT_ROOT / "examples/llm_simple_qa", self.source / "examples/llm_simple_qa", ignore=shutil.ignore_patterns("__pycache__"))
        settings = replace(load_settings(PROJECT_ROOT, load_credentials=False), project_root=self.source, state_root=self.root / "state", python_executable=sys.executable)
        self.harness = AgentHarness(settings)
        self.metric_path = "examples/llm_simple_qa/testenv/acc.py"

    async def asyncTearDown(self):
        self.assertEqual(self.harness.usage_summary()["model_calls"], 0)
        await self.harness.close()

    async def create(self):
        await self.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
        self.harness._ensure_task("读取当前实例创建时的指标源码")
        goal = self.harness.original_goals.load(self.harness.current_instance)
        binding = next(item for item in goal["bindings"] if item["path"] == self.metric_path)
        return goal, binding

    async def read(self, reference, **parameters):
        return await self.harness.tools.execute({"id": "goal-evidence-" + uuid.uuid4().hex[:8], "name": "read_evidence", "arguments": {"reference": reference, **parameters}})

    async def test_bound_original_source_text_and_pagination_survive_current_edits(self):
        goal, binding = await self.create()
        original = (self.source / self.metric_path).read_bytes()
        (Path(self.harness.current_instance["code_root"]) / self.metric_path).write_text("current = 'changed'\n", encoding="utf-8")
        (self.source / self.metric_path).write_text("source = 'changed'\n", encoding="utf-8")
        result = await self.read(binding["object_ref"], start_line=16, end_line=20)
        self.assertEqual(result["status"], "success", result)
        data = result["data"]
        lines = original.decode("utf-8").splitlines()
        self.assertEqual(data["content"], "\n".join(f"{index + 1}: {lines[index]}" for index in range(15, 20)))
        self.assertEqual(data["path"], self.metric_path)
        self.assertEqual(data["source_kind"], "creation_snapshot")
        self.assertEqual(data["file_version"], hashlib.sha256(original).hexdigest())
        self.assertEqual(data["object_ref"], binding["object_ref"])
        self.assertEqual(data["original_goal_ref"], goal["evidence_ref"])
        self.assertEqual(data["total_lines"], len(lines))
        self.assertTrue(data["truncated"])
        self.assertEqual(data["next_line"], 21)
        tail = await self.read(binding["object_ref"], start_line=21, end_line=2000)
        self.assertFalse(tail["data"]["truncated"])
        rejected = await self.read(binding["object_ref"], start_line=0)
        self.assertEqual(rejected["status"], "error")

    async def test_foreign_instance_and_unbound_creation_objects_are_denied(self):
        first_goal, _ = await self.create()
        first_id = self.harness.current_instance["instance_id"]
        with (self.source / self.metric_path).open("a", encoding="utf-8") as stream:
            stream.write("\n# 第二个实例的独立源码版本。\n")
        _, foreign = await self.create()
        self.harness.select_instance(first_id)
        rejected = await self.read(foreign["object_ref"])
        self.assertEqual(rejected["status"], "error")
        self.assertIn("已绑定", rejected["error"])
        source = self.harness.store.read(first_goal["source_manifest_ref"])
        unbound = source["file_objects"]["examples/llm_simple_qa/README.md"]
        rejected = await self.read(unbound["object_ref"])
        self.assertEqual(rejected["status"], "error")
        self.assertIn("已绑定", rejected["error"])

    async def test_corrupted_object_and_unnormalized_references_are_rejected(self):
        _, binding = await self.create()
        reference = binding["object_ref"]
        for invalid in (reference.replace("/objects/", "/objects/../objects/"), reference.replace("/", "\\"), str(self.harness.store.path(reference))):
            result = await self.read(invalid)
            self.assertEqual(result["status"], "error", result)
        target = self.harness.store.path(reference)
        target.write_bytes(target.read_bytes() + b"\n")
        rejected = await self.read(reference)
        self.assertEqual(rejected["status"], "error")
        self.assertIn("损坏", rejected["error"])

    async def test_bound_binary_is_rejected_and_bom_text_uses_existing_decoder(self):
        metric = self.source / self.metric_path
        original = metric.read_text(encoding="utf-8")
        metric.write_bytes(original.encode("utf-8") + b"\x00")
        _, binary = await self.create()
        rejected = await self.read(binary["object_ref"])
        self.assertEqual(rejected["status"], "error")
        self.assertIn("binary", rejected["error"])
        metric.write_bytes(original.encode("utf-16"))
        _, text = await self.create()
        read = await self.read(text["object_ref"], start_line=16, end_line=18)
        self.assertEqual(read["status"], "success", read)
        self.assertEqual(read["data"]["encoding"], "utf-16")
        self.assertEqual(read["data"]["content"], "\n".join(f"{index + 1}: {original.splitlines()[index]}" for index in range(15, 18)))

    async def test_task_goal_identity_is_checked_and_unbound_task_uses_instance_goal(self):
        _, binding = await self.create()
        self.harness.task["original_goal"] = {**self.harness.task["original_goal"], "goal_digest": "0" * 64}
        rejected = await self.read(binding["object_ref"])
        self.assertEqual(rejected["status"], "error")
        self.assertIn("正式版本", rejected["error"])
        self.harness.task.pop("original_goal")
        read = await self.read(binding["object_ref"])
        self.assertEqual(read["status"], "success", read)

    async def test_task_bound_goal_version_controls_object_access_after_source_recovery(self):
        await self.create()
        original = (self.source / self.metric_path).read_bytes()
        instance = copy.deepcopy(self.harness.current_instance)
        for key in ("original_goal_ref", "original_source_ref", "goal_id", "goal_digest"):
            instance.pop(key, None)
        for root in (self.source, Path(instance["code_root"])):
            (root / self.metric_path).write_text("changed = True\n", encoding="utf-8")
        self.harness.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        old_goal = self.harness.original_goals.resolve(instance)
        self.assertEqual(old_goal["status"], "source_missing")
        self.harness._bind_original_goal(resolve=False)
        (self.source / self.metric_path).write_bytes(original)
        new_goal = self.harness.original_goals.resolve(instance)
        self.assertEqual(new_goal["status"], "complete")
        self.harness.current_instance = self.harness.instances.get(instance["instance_id"])
        binding = next(item for item in new_goal["bindings"] if item["path"] == self.metric_path)
        self.assertEqual(self.harness.task["original_goal"]["evidence_ref"], old_goal["evidence_ref"])
        rejected = await self.read(binding["object_ref"])
        self.assertEqual(rejected["status"], "error")
        self.assertIn("已绑定", rejected["error"])
        self.harness.task.pop("original_goal")
        read = await self.read(binding["object_ref"])
        self.assertEqual(read["status"], "success", read)
        self.assertEqual(read["data"]["original_goal_ref"], new_goal["evidence_ref"])

    async def test_only_current_registered_completion_contract_is_readable(self):
        await self.create()

        async def declare():
            result = await self.harness.tools.execute({"id": "declare-" + uuid.uuid4().hex[:8], "name": "declare_completion_contract", "arguments": {"requirements": [{"id": "answer", "kind": "answer", "description": "说明指标来源"}], "answer_text": "指标来源保存在当前实例的创建归档中。"}})
            self.assertEqual(result["status"], "success", result)
            return self.harness.completion.contract()["evidence_ref"]

        previous = await declare()
        current = await self.read(previous)
        self.assertEqual(current["status"], "success", current)
        self.assertIn("说明指标来源", current["data"]["content"])
        self.harness.state["status"] = "completed"
        self.harness._ensure_task("说明另一个当前问题")
        reference = await declare()
        self.assertNotEqual(reference, previous)
        rejected = await self.read(previous)
        self.assertEqual(rejected["status"], "error")
        current = await self.read(reference)
        self.assertEqual(current["status"], "success", current)


if __name__ == "__main__":
    unittest.main()
