# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
import unittest
import uuid
from pathlib import Path

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.context.compactor import complete_interactions, latest_tool_interaction
from ianvs_agent.tools.filesystem import CodeTools


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RepairContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        capture = os.environ.get("IANVS_REPAIR_CAPTURE")
        if not capture:
            self.skipTest("设置 IANVS_REPAIR_CAPTURE 后读取真实失败会话记录")
        self.source = Path(capture)
        if self.source.is_dir():
            self.source = self.source / "session_snapshot.json"
        self.source_bytes = self.source.read_bytes()
        self.snapshot = json.loads(self.source_bytes)
        self.original = copy.deepcopy(self.snapshot)
        self.task, self.state, self.history = self.snapshot["task"], self.snapshot["state"], self.snapshot["history"]
        self.root = PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/repair-context" / uuid.uuid4().hex
        profile = ModelProfile("真实记录请求准备检查", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3")
        settings = AgentSettings(PROJECT_ROOT, self.root, profiles={"ark": profile})
        self.harness = AgentHarness(settings)
        self.harness.current_instance = json.loads((self.source.parent / "instance.json").read_text(encoding="utf-8"))
        self.harness.session = copy.deepcopy(self.snapshot["session"])
        self.harness.task, self.harness.state = self.task, self.state
        self.harness.session_relative_path = f"instances/{self.task['instance_id']}/sessions/{self.harness.session['session_id']}"
        self.harness.events.bind(instance_id=self.task["instance_id"], session_id=self.harness.session["session_id"], task_id=self.task["task_id"])
        self.harness._connect_provider()
        self.context, self.provider, self.usage = self.harness.context, self.harness.provider, self.harness.usage

    async def asyncTearDown(self):
        if hasattr(self, "provider"):
            await self.provider.close()

    def _check_original_records(self):
        self.assertEqual(self.snapshot, self.original)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), hashlib.sha256(self.source_bytes).digest())
        self.assertEqual(self.usage.summary()["model_calls"], 0)
        self.assertIsNone(self.provider.client)

    def _check_state_view(self, prepared):
        task = json.loads(prepared["messages"][1]["content"].split("\n", 1)[1])
        view = json.loads(prepared["messages"][2]["content"].split("\n", 1)[1])
        facts, hypotheses = self.state["facts"], self.state["hypotheses"]
        indices = sorted(set(range(min(5, len(facts)))) | set(range(max(0, len(facts) - 15), len(facts))))
        self.assertEqual(task, self.task)
        self.assertEqual(task["goal"], self.original["task"]["goal"])
        self.assertEqual(task["acceptance"], self.original["task"]["acceptance"])
        self.assertEqual(view["facts"], [facts[index] for index in indices])
        self.assertEqual(view["hypotheses"], hypotheses[-10:])
        for key, value in self.state.items():
            if key not in {"facts", "hypotheses"}:
                self.assertEqual(view[key], value, key)
        selection = view["context_selection"]
        self.assertEqual(selection["facts_total"], len(facts))
        self.assertEqual(selection["facts_omitted"], len(facts) - len(indices))
        self.assertEqual(selection["hypotheses_total"], len(hypotheses))
        self.assertEqual(selection["hypotheses_omitted"], max(0, len(hypotheses) - 10))
        self.assertEqual(self.harness.store.read(selection["full_state_ref"]), self.state)
        self.assertEqual(view["verification"], self.state["verification"])
        task_reference = self.context.session_path + f"/tool-output/TASK-{self.task['task_id']}.json"
        self.assertEqual(self.harness.store.read(task_reference), self.task)
        return selection

    def _captured_progress(self, task):
        records = [json.loads(line) for line in (self.source.parent / "tools.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in records:
            self.harness.store.append(self.context.session_path + "/tools.jsonl", record)
        self.harness.task = task
        self.harness.state = copy.deepcopy(self.state)
        self.harness._update_repair_progress(records)
        return self.harness.state

    async def test_captured_plan_keeps_program_state_and_original_acceptance(self):
        prepared = await self.context.prepare(self.task, self.state, self.history, self.harness.tools.definitions())
        self._check_state_view(prepared)
        self.assertIsNone(prepared["checkpoint_id"])
        self.assertIn("允许在任务授权范围内修复引发异常的源码实现", prepared["messages"][0]["content"])
        self.assertIn("phase=verification_pending", prepared["messages"][-1]["content"])
        self.assertEqual(json.loads(prepared["messages"][2]["content"].split("\n", 1)[1]).get("repair_progress"), self.state.get("repair_progress"))
        self._check_original_records()

    async def test_captured_observe_keeps_current_tool_pair_and_readable_full_state(self):
        blocks = complete_interactions(self.history)
        current = next(block for block in reversed(blocks) if block[0].get("tool_calls"))
        prepared = await self.context.prepare(self.task, self.state, current, [], role="observe")
        selection = self._check_state_view(prepared)
        outbound = [message for message in prepared["messages"] if message.get("role") in {"assistant", "tool"}]
        complete_interactions(outbound)
        self.assertEqual(outbound, latest_tool_interaction(current))
        evidence = await CodeTools(self.harness).read_evidence(selection["full_state_ref"], 1, 200)
        self.assertEqual(evidence["reference"], selection["full_state_ref"])
        self.assertIn('"status"', evidence["content"])
        self.assertIn("只返回新增或修正的事实和假设", prepared["messages"][0]["content"])
        self._check_original_records()

    def test_captured_short_fact_ranges_do_not_repeat_overlap(self):
        for count in (0, 3, 18, 21):
            with self.subTest(fact_count=count):
                state = copy.deepcopy(self.state)
                state["facts"] = state["facts"][:count]
                view = self.context._current_state_view(self.task, state)
                self.assertEqual(len(view["facts"]), min(count, 20))
                if count <= 20:
                    self.assertEqual(view["facts"], state["facts"])
                self.assertEqual(self.harness.store.read(view["context_selection"]["full_state_ref"]), state)
        self._check_original_records()

    async def test_captured_read_versions_and_program_progress_are_preserved(self):
        state = self._captured_progress(copy.deepcopy(self.task))
        progress = state["repair_progress"]
        self.assertTrue(progress["files_read"])
        frozen = self.task["acceptance"]["file_identities"]
        for item in progress["files_read"]:
            if item["path"] in frozen:
                self.assertEqual(item["matches_frozen_identity"], item["file_version"] == frozen[item["path"]])
        prepared = await self.context.prepare(self.task, state, [], self.harness.tools.definitions())
        view = json.loads(prepared["messages"][2]["content"].split("\n", 1)[1])
        self.assertEqual(view["repair_progress"], progress)
        self.assertEqual(view["verification"], self.state["verification"])
        self.assertEqual(self.harness.store.read(view["context_selection"]["full_state_ref"]), state)
        self._check_original_records()

    async def test_captured_large_task_sends_read_identities_and_keeps_full_acceptance(self):
        capture = os.environ.get("IANVS_REPAIR_TASK_CAPTURE")
        if not capture:
            self.skipTest("设置 IANVS_REPAIR_TASK_CAPTURE 后读取完整正式任务清单")
        source = Path(capture)
        if source.is_dir():
            source = source / "task.json"
        original_bytes = source.read_bytes()
        task = json.loads(original_bytes)
        manifest_capture = os.environ.get("IANVS_REPAIR_MANIFEST_CAPTURE")
        if manifest_capture:
            manifest_source = Path(manifest_capture)
            baseline = json.loads(manifest_source.read_text(encoding="utf-8"))
            trial_source = manifest_source.parents[5] / "trial.json"
            trial = json.loads(trial_source.read_text(encoding="utf-8"))
            case = next(case for case in trial["cases"] if case["instance_id"] == task["instance_id"])
            self.assertEqual(baseline["run_id"], case["baseline_run_id"])
            self.assertEqual(baseline["status"], "completed")
            # 身份范围采用正式试验驱动的健康基线清单与允许修改路径。
            task["acceptance"]["file_identities"] = {path: digest for path, digest in baseline["source_manifest"].items() if path != case["target"]}
            self.harness.store.write("capture-sources.json", {"task": str(source), "baseline_run": str(manifest_source), "trial": str(trial_source), "allowed_modified_file": case["target"]})
        original_task = copy.deepcopy(task)
        identities = task["acceptance"]["file_identities"]
        self.assertGreater(len(identities), 24)
        state = self._captured_progress(copy.deepcopy(self.task))
        prepared = await self.context.prepare(task, state, [], self.harness.tools.definitions())
        view = json.loads(prepared["messages"][1]["content"].split("\n", 1)[1])
        state_view = json.loads(prepared["messages"][2]["content"].split("\n", 1)[1])
        expected = set(list(identities)[:8]) | {item["path"] for item in state["repair_progress"]["files_read"] if item["path"] in identities}
        self.assertEqual(set(view["acceptance"]["file_identities"]), expected)
        self.assertEqual(view["acceptance"]["file_identities"], {path: identities[path] for path in expected})
        for key, value in task.items():
            if key != "acceptance":
                self.assertEqual(view[key], value, key)
        for key, value in task["acceptance"].items():
            if key != "file_identities":
                self.assertEqual(view["acceptance"][key], value, key)
        selection = view["context_selection"]
        self.assertEqual(selection["file_identities_total"], len(identities))
        self.assertEqual(selection["included"], len(expected))
        self.assertEqual(selection["omitted"], len(identities) - len(expected))
        self.assertEqual(self.harness.store.read(selection["full_task_ref"]), original_task)
        self.assertEqual(state_view["repair_progress"], state["repair_progress"])
        self.assertEqual(state_view["verification"], self.state["verification"])
        evidence = await CodeTools(self.harness).read_evidence(selection["full_task_ref"], 1, 200)
        self.assertEqual(evidence["reference"], selection["full_task_ref"])
        self.assertIn('"acceptance"', evidence["content"])
        compaction = self.context._compaction_messages(task, state, [], None)
        compact_view = json.loads(compaction[1]["content"].split("\n", 1)[1])
        self.assertEqual(compact_view["task"], view)
        self.assertEqual(compact_view["state"], state_view)
        self.assertEqual(task, original_task)
        self.assertEqual(source.read_bytes(), original_bytes)
        self._check_original_records()


if __name__ == "__main__":
    unittest.main()
