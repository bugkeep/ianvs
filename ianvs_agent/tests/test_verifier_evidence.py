# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import csv
import hashlib
import os
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import yaml
import psutil

from ianvs_agent.agent.events import EventBus
from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.runtime.instances import file_manifest, source_files
from ianvs_agent.runtime.packages import package_identity
from ianvs_agent.runtime.runner import Runner
from ianvs_agent.runtime.verifier import Verifier
from ianvs_agent.storage.local_store import LocalStore


@unittest.skipUnless(os.environ.get("IANVS_RUNTIME_EVIDENCE_ROOT"), "需要已有真实 Ianvs Run，设置 IANVS_RUNTIME_EVIDENCE_ROOT 后执行")
class RealRunVerifierTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[2]
        original = LocalStore(Path(os.environ["IANVS_RUNTIME_EVIDENCE_ROOT"]))
        instance_id = os.environ.get("IANVS_RUNTIME_EVIDENCE_INSTANCE", "instance-0001")
        run_id = os.environ.get("IANVS_RUNTIME_EVIDENCE_RUN", "run-0002")
        cls.original_instance = original.read(f"instances/{instance_id}/instance.json")
        cls.original_run = original.read(f"instances/{instance_id}/runs/{run_id}/run.json")
        cls.original_counters = original.read("counters.json")
        if cls.original_instance is None or cls.original_run is None:
            raise RuntimeError("真实实例或 Run 证据不存在")
        if cls.original_run["status"] != "completed" or cls.original_run["exit_code"] != 0:
            raise RuntimeError("该组回归需要正常完成的真实基线 Run")
        cls.control = LocalStore(Path(os.environ.get("IANVS_VERIFIER_TEST_ROOT", cls.project / "workspace/verifier-evidence-tests")))

    async def asyncSetUp(self):
        replay_id = self.control.allocate_id("evidence-copy")
        profile = ModelProfile("真实证据回归", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3")
        self.settings = AgentSettings(project_root=self.project, state_root=self.control.root / replay_id, profiles={"ark": profile})
        self.store = LocalStore(self.settings.state_root)
        self.store.write("counters.json", self.original_counters)
        self.events = EventBus(self.store)
        self.instance = copy.deepcopy(self.original_instance)
        self.run = copy.deepcopy(self.original_run)
        instance_root = self.settings.state_root / f"instances/{self.instance['instance_id']}"
        code_root = instance_root / "code"
        source_root = Path(self.original_instance["code_root"])
        for source in source_files(source_root):
            destination = code_root / source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        # 演示实例可能已注入故障；复制仓库中与真实基线完全一致的原 benchmark。
        benchmark = self.instance["benchmark_path"]
        original_benchmark = self.project / benchmark
        digest = hashlib.sha256(original_benchmark.read_bytes()).hexdigest()
        self.assertEqual(digest, self.original_run["source_manifest"][benchmark])
        shutil.copy2(original_benchmark, code_root / benchmark)
        self.assertEqual(file_manifest(code_root), self.original_run["source_manifest"])
        self.instance["code_root"] = str(code_root)
        self.instance["verification"] = None
        self.instance["last_run_id"] = self.run["run_id"]
        self.instance["environment"] = copy.deepcopy(self.run["environment_snapshot"])
        original_output = Path(self.original_run["output_dir"])
        output = instance_root / f"runs/{self.run['run_id']}/output"
        shutil.copytree(original_output, output)
        self.run["cwd"] = str(code_root)
        self.run["output_dir"] = str(output)
        self.run["artifacts"] = [str(output / Path(value).relative_to(original_output)) for value in self.run["artifacts"]]
        self.rank_path = None
        for event in self.run["events"]:
            if event["event_type"] == "rank.completed":
                for name in ("all_rank_file", "selected_rank_file"):
                    event["payload"][name] = str(output / Path(event["payload"][name]).relative_to(original_output))
                self.rank_path = Path(event["payload"]["all_rank_file"])
        for path in output.rglob("*.csv"):
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            for row in rows:
                row["url"] = str(output / Path(row["url"]).relative_to(original_output))
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        self.store.write("original-run.json", self.original_run)
        self.store.write("provenance.json", {"source_run_id": self.run["run_id"], "source_started_at": self.run["started_at"], "source_evidence_root": os.environ["IANVS_RUNTIME_EVIDENCE_ROOT"], "location_rebased": True})
        self._save()
        self.verifier = Verifier(self.settings, self.store, self.events)

    def _save(self):
        self.store.write(f"instances/{self.instance['instance_id']}/instance.json", self.instance)
        self.store.write(f"instances/{self.instance['instance_id']}/runs/{self.run['run_id']}/run.json", self.run)

    def _acceptance(self, seconds):
        boundary = (datetime.fromisoformat(self.run["started_at"]) + timedelta(seconds=seconds)).isoformat()
        return {"repair_task_id": "repair-boundary-test", "repair_started_at": boundary, "latest_patch_at": boundary, "metric_thresholds": {"acc": {"min": 1.0}}}

    async def test_real_run_after_repair_boundary_passes(self):
        result = await self.verifier.verify(self.instance, self.run, self._acceptance(-1))
        self.assertTrue(result["passed"], result["reasons"])
        self.assertTrue(result["repair_completed"], result["reasons"])

    async def test_real_package_order_does_not_change_environment_identity(self):
        packages = self.run["environment_snapshot"]["packages"]
        packages.reverse()
        self._save()
        result = await self.verifier.verify(self.instance, self.run, self._acceptance(-1))
        self.assertTrue(result["passed"], result["reasons"])
        self.assertTrue(result["repair_completed"], result["reasons"])
        identity = package_identity(packages)
        self.assertNotEqual(identity, package_identity(packages[:-1]))
        self.assertNotEqual(identity, package_identity([*packages, packages[0]]))
        harness = AgentHarness(self.settings)
        unchanged = await asyncio.to_thread(harness.instances.check_changes, self.instance["instance_id"], False)
        self.assertFalse(unchanged["environment_changed"])
        self.assertFalse(unchanged["needs_verification"])
        self.assertFalse(unchanged["verification"]["stale"])
        await harness.close()

    async def test_old_real_run_cannot_complete_new_repair(self):
        result = await self.verifier.verify(self.instance, self.run, self._acceptance(1))
        self.assertTrue(result["passed"], result["reasons"])
        self.assertFalse(result["repair_completed"])

    async def test_real_run_at_latest_patch_boundary_is_not_fresh(self):
        acceptance = self._acceptance(-1)
        acceptance["latest_patch_at"] = self.run["started_at"]
        result = await self.verifier.verify(self.instance, self.run, acceptance)
        self.assertTrue(result["passed"], result["reasons"])
        self.assertFalse(result["repair_completed"])

    async def test_missing_actual_rank_artifact_fails(self):
        self.rank_path.unlink()
        result = await self.verifier.verify(self.instance, self.run)
        self.assertFalse(result["passed"])
        self.assertFalse(result["results_complete"])

    async def test_polluted_actual_rank_metric_fails(self):
        with self.rank_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]["acc"] = "0.0"
        with self.rank_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result = await self.verifier.verify(self.instance, self.run)
        self.assertFalse(result["passed"])
        self.assertFalse(result["results_complete"])

    async def test_required_skipped_real_validator_check_fails(self):
        self.instance["inventory_entry"]["path"] = str(Path(self.instance["benchmark_path"]).parent / "testalgorithms")
        self._save()
        static = await self.verifier.validate(self.instance)
        self.assertTrue(static["passed"], static["checks"])
        skipped = [item["name"] for item in static["checks"] if item["status"] == "SKIP"]
        self.assertTrue(skipped, "真实 validator 没有产生 SKIP，不能验证该场景")
        result = await self.verifier.verify(self.instance, self.run, {"required_checks": [skipped[0]]})
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks_passed"])

    async def test_empty_required_metrics_cannot_remove_frozen_metric(self):
        with self.rank_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = [{name: value for name, value in row.items() if name != "acc"} for row in reader]
            fields = [name for name in reader.fieldnames if name != "acc"]
        with self.rank_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        result = await self.verifier.verify(self.instance, self.run, {"required_metrics": []})
        self.assertFalse(result["passed"])
        self.assertFalse(result["results_complete"])

    async def test_real_run_from_another_instance_is_rejected(self):
        self.instance["instance_id"] = "different-instance"
        self.store.write("instances/different-instance/instance.json", self.instance)
        with self.assertRaises(ValueError):
            await self.verifier.verify(self.instance, self.run)

    async def test_private_model_conditions_cannot_pass_with_real_qwen_run(self):
        acceptance = self._acceptance(-1)
        acceptance.update(required_model_identity="company-private-model", model_authorization="required")
        result = await self.verifier.verify(self.instance, self.run, acceptance)
        self.assertTrue(result["execution_completed"])
        self.assertTrue(result["results_complete"])
        self.assertFalse(result["checks_passed"])
        self.assertFalse(result["passed"])
        self.assertFalse(result["repair_completed"])
        self.assertEqual(result["unsupported_acceptance_fields"], ["model_authorization", "required_model_identity"])
        unsupported = [item for item in result["checks"] if item.get("acceptance_field")]
        self.assertEqual({item["status"] for item in unsupported}, {"SKIP"})
        self.assertEqual(result["acceptance"], acceptance)

    async def test_replaced_metric_source_is_rejected(self):
        root = Path(self.instance["code_root"])
        testenv_path = root / "examples/llm_simple_qa/testenv/testenv.yaml"
        testenv = yaml.safe_load(testenv_path.read_text(encoding="utf-8"))
        old_metric = root / testenv["testenv"]["metrics"][0]["url"]
        alternate = old_metric.with_name("alternate_acc.py")
        shutil.copy2(old_metric, alternate)
        testenv["testenv"]["metrics"][0]["url"] = alternate.relative_to(root).as_posix()
        testenv_path.write_text(yaml.safe_dump(testenv), encoding="utf-8")
        result = await self.verifier.verify(self.instance, self.run)
        checks = {item["name"]: item["status"] for item in result["checks"]}
        self.assertEqual(checks["原始评测指标实现保持一致"], "PASS")
        self.assertEqual(checks["原始指标名称与实现来源保持一致"], "FAIL")
        self.assertFalse(result["passed"])

    async def test_replaced_algorithm_configuration_is_rejected(self):
        root = Path(self.instance["code_root"])
        benchmark_path = root / self.instance["benchmark_path"]
        benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
        algorithm = benchmark["benchmarkingjob"]["test_object"]["algorithms"][0]
        original = root / algorithm["url"]
        alternate = original.with_name("alternate_algorithm.yaml")
        shutil.copy2(original, alternate)
        algorithm["url"] = alternate.relative_to(root).as_posix()
        benchmark_path.write_text(yaml.safe_dump(benchmark), encoding="utf-8")
        result = await self.verifier.verify(self.instance, self.run)
        checks = {item["name"]: item["status"] for item in result["checks"]}
        self.assertEqual(checks["原始算法目标与配置来源保持一致"], "FAIL")
        self.assertFalse(result["passed"])

    async def test_replaced_frozen_dataset_source_is_rejected(self):
        root = Path(self.instance["code_root"])
        testenv_path = root / "examples/llm_simple_qa/testenv/testenv.yaml"
        testenv = yaml.safe_load(testenv_path.read_text(encoding="utf-8"))
        original = root / testenv["testenv"]["dataset"]["test_data"]
        alternate = original.with_name("alternate_data.jsonl")
        shutil.copy2(original, alternate)
        testenv["testenv"]["dataset"]["test_data"] = alternate.relative_to(root).as_posix()
        testenv_path.write_text(yaml.safe_dump(testenv), encoding="utf-8")
        result = await self.verifier.verify(self.instance, self.run)
        checks = {item["name"]: item["status"] for item in result["checks"]}
        self.assertEqual(checks["冻结的原始数据来源保持一致"], "FAIL")
        self.assertFalse(result["passed"])

    async def test_equivalent_reference_paths_keep_goal_bindings(self):
        root = Path(self.instance["code_root"])
        testenv_path = root / "examples/llm_simple_qa/testenv/testenv.yaml"
        testenv = yaml.safe_load(testenv_path.read_text(encoding="utf-8"))
        metric = testenv["testenv"]["metrics"][0]
        metric["url"] = (root / metric["url"]).resolve().as_posix()
        for name, value in testenv["testenv"]["dataset"].items():
            testenv["testenv"]["dataset"][name] = str((root / value).resolve())
        testenv_path.write_text(yaml.safe_dump(testenv), encoding="utf-8")
        benchmark_path = root / self.instance["benchmark_path"]
        benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
        algorithm = benchmark["benchmarkingjob"]["test_object"]["algorithms"][0]
        algorithm["url"] = (root / algorithm["url"]).resolve().as_posix()
        benchmark_path.write_text(yaml.safe_dump(benchmark), encoding="utf-8")
        result = await self.verifier.verify(self.instance, self.run)
        checks = {item["name"]: item["status"] for item in result["checks"]}
        for name in ("原始指标名称与实现来源保持一致", "原始算法目标与配置来源保持一致", "冻结的原始数据来源保持一致"):
            self.assertEqual(checks[name], "PASS")
        self.assertFalse(result["passed"], "改变源码后必须新 Run，路径等效不能替代重新执行")

    async def test_missing_original_paths_allow_repair_with_frozen_baseline(self):
        benchmark = yaml.safe_load(self.instance["task_spec"]["benchmark_text"])
        benchmark["benchmarkingjob"]["test_object"]["algorithms"][0]["url"] = "examples/llm_simple_qa/missing-algorithm.yaml"
        testenv = yaml.safe_load(self.instance["task_spec"]["testenv_text"])
        testenv["testenv"]["dataset"]["test_data"] = "dataset/llm_simple_qa/missing-data.jsonl"
        self.instance["task_spec"]["benchmark_text"] = yaml.safe_dump(benchmark)
        self.instance["task_spec"]["testenv_text"] = yaml.safe_dump(testenv)
        self._save()
        acceptance = {"metric_thresholds": {"acc": {"min": 1.0}}, "file_identities": {name: value for name, value in self.original_run["source_manifest"].items() if name.startswith(("examples/llm_simple_qa/", "dataset/"))}}
        frozen = copy.deepcopy(acceptance)
        result = await self.verifier.verify(self.instance, self.run, acceptance)
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(result["acceptance"], frozen)

    async def test_new_cancelled_run_invalidates_old_passed_verification(self):
        previous = await self.verifier.verify(self.instance, self.run)
        self.assertTrue(previous["passed"], previous["reasons"])
        runner = Runner(self.settings, self.store, self.events)
        started = asyncio.Event()
        self.events.subscribe(lambda event: started.set() if event["event_type"] == "run.started" else None)
        pending = asyncio.create_task(runner.run(self.instance))
        await asyncio.wait_for(started.wait(), timeout=10)
        path = f"instances/{self.instance['instance_id']}/instance.json"
        during = self.store.read(path)
        self.assertTrue(during["verification"]["stale"])
        self.assertTrue(during["needs_verification"])
        self.assertNotEqual(during["last_run_id"], previous["run_id"])
        await runner.cancel()
        cancelled = await asyncio.wait_for(pending, timeout=30)
        self.assertEqual(cancelled["status"], "cancelled")
        after = self.store.read(path)
        self.assertTrue(after["verification"]["stale"])
        self.assertEqual(after["verification"]["superseded_by_run_id"], cancelled["run_id"])
        self.assertEqual(after["status"], "needs_verification")
        historical = self.store.read(f"instances/{self.instance['instance_id']}/runs/{previous['run_id']}/verification.json")
        self.assertTrue(historical["passed"])
        self.assertFalse(historical["stale"])
        reviewed = await self.verifier.verify(self.instance, self.run)
        self.assertTrue(reviewed["passed"], reviewed["reasons"])
        self.assertTrue(reviewed["stale"])
        after_review = self.store.read(path)
        self.assertTrue(after_review["verification"]["stale"])
        self.assertEqual(after_review["last_run_id"], cancelled["run_id"])
        self.assertTrue(after_review["needs_verification"])
        harness = AgentHarness(self.settings)
        try:
            harness.select_instance(self.instance["instance_id"])
            harness._ensure_task("回溯历史 Run，同时保留最新取消运行的正式状态")
            harness.state["last_run_id"] = cancelled["run_id"]
            harness.state["verification"] = copy.deepcopy(after_review["verification"])
            harness._save_session()
            self.assertFalse(harness._repair_is_complete())
            reviewed_tool = await harness.tools.execute({"id": "historical-run-review", "name": "verify_run", "arguments": {"run_id": self.run["run_id"]}})
            self.assertEqual(reviewed_tool["status"], "success")
            self.assertTrue(reviewed_tool["data"]["passed"])
            self.assertTrue(reviewed_tool["data"]["stale"])
            self.assertTrue(harness.state["verification"]["stale"])
            self.assertEqual(harness.state["last_run_id"], cancelled["run_id"])
            self.assertTrue(harness.verification_report()["stale"])
            self.assertFalse(harness._repair_is_complete())
            self.assertEqual(harness.usage.summary()["model_calls"], 0)
        finally:
            await harness.close()

    async def test_harness_cancel_after_real_core_events_persists_finished_run(self):
        previous = await self.verifier.verify(self.instance, self.run)
        self.assertTrue(previous["passed"], previous["reasons"])
        harness = AgentHarness(self.settings)
        started = asyncio.Event()
        harness.events.subscribe(lambda event: started.set() if event["event_type"] == "run.stage_changed" and event["payload"].get("stage") == "configuration_load" else None)
        operation = None
        try:
            harness.select_instance(self.instance["instance_id"])
            operation = asyncio.create_task(harness.run_current())
            await asyncio.wait_for(started.wait(), timeout=60)
            current = harness.instances.get(self.instance["instance_id"])
            active = harness.runner.get_run(current["instance_id"], current["last_run_id"])
            self.assertIsNotNone(active["pid"])
            self.assertTrue(psutil.pid_exists(active["pid"]))
            await harness.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(operation, timeout=30)
            saved = harness.runner.get_run(current["instance_id"], current["last_run_id"])
            self.assertEqual(saved["status"], "cancelled")
            self.assertTrue(saved["interrupted"])
            self.assertIsNotNone(saved["completed_at"])
            self.assertIsNotNone(saved["exit_code"])
            self.assertFalse(psutil.pid_exists(saved["pid"]))
            actual = harness.instances.get(current["instance_id"])
            self.assertTrue(actual["verification"]["stale"])
            self.assertEqual(actual["verification"]["superseded_by_run_id"], saved["run_id"])
            self.assertTrue(actual["needs_verification"])
            self.assertEqual(harness.usage.summary()["model_calls"], 0)
        finally:
            await harness.close()
            if operation is not None:
                await asyncio.gather(operation, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
