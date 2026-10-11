# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import yaml

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings
from ianvs_agent.runtime.goal_checks import compare_goal, original_metrics
from ianvs_agent.runtime.instances import InstanceManager, configuration_evidence, file_manifest
from ianvs_agent.runtime.original_goal import OriginalGoalStore
from ianvs_agent.storage.local_store import LocalStore


class OriginalGoalTests(unittest.TestCase):
    def setUp(self):
        self.project = Path(__file__).resolve().parents[2]
        artifact = Path(os.environ.get("IANVS_ORIGINAL_GOAL_TEST_ROOT", self.project / "workspace/original-goal-tests"))
        control = LocalStore(artifact)
        self.case_root = artifact / control.allocate_id("goal-case")
        self.source = self.case_root / "source"
        inventory = ".github/workflows/validator/data/example_inventory.yaml"
        destination = self.source / inventory
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.project / inventory, destination)
        self.settings = AgentSettings(self.source, self.case_root / "state", python_executable=sys.executable)
        self.store = LocalStore(self.settings.state_root)
        self.manager = InstanceManager(self.settings, self.store, EventBus(self.store))
        self.goals = OriginalGoalStore(self.settings, self.store)

    def copy_example(self, relative="examples/llm_simple_qa"):
        shutil.copytree(self.project / relative, self.source / relative, ignore=shutil.ignore_patterns("__pycache__", "workspace", "*.pyc"))
        return relative

    def create(self, relative="examples/llm_simple_qa", filename="benchmarkingjob.yaml"):
        return self.manager.create(relative + "/" + filename)

    def legacy(self, instance):
        instance = copy.deepcopy(instance)
        for key in ("original_goal_ref", "original_source_ref", "goal_id", "goal_digest"):
            instance.pop(key, None)
        self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        return instance

    def test_creation_archives_exact_bytes_and_survives_source_edits(self):
        self.copy_example()
        instance = self.create()
        original = copy.deepcopy(instance["task_spec"])
        goal = self.goals.load(instance)
        self.assertEqual(goal["status"], "complete")
        self.assertEqual(goal["goal_digest"], instance["goal_digest"])
        source = self.store.read(goal["source_manifest_ref"])
        self.assertEqual(set(source["file_objects"]), set(instance["source_manifest"]))
        benchmark = goal["documents"]["benchmark"]
        path = Path(instance["code_root"]) / benchmark["path"]
        object_path = self.store.path(benchmark["object_ref"])
        self.assertFalse(object_path.is_relative_to(Path(instance["code_root"])))
        self.assertEqual(object_path.read_bytes(), path.read_bytes())
        if os.name == "posix":
            self.assertNotEqual(object_path.stat().st_ino, path.stat().st_ino)
        path.write_text("changed: true\n", encoding="utf-8")
        (self.source / benchmark["path"]).write_text("changed: true\n", encoding="utf-8")
        loaded = OriginalGoalStore(self.settings, LocalStore(self.store.root)).load(instance)
        self.assertEqual(loaded, goal)
        self.assertEqual(self.goals.resolve(instance), goal)
        self.assertEqual(instance["task_spec"], original)
        self.assertEqual(file_manifest(object_path.parent)[object_path.name], benchmark["sha256"])

    def test_exact_objects_are_deduplicated_across_new_instances(self):
        self.copy_example()
        first = self.create()
        objects = sorted(self.store.root.glob("original-goals/objects/*/*.bin"))
        sizes = {path: path.stat().st_size for path in objects}
        second = self.create()
        self.assertEqual({path: path.stat().st_size for path in self.store.root.glob("original-goals/objects/*/*.bin")}, sizes)
        self.assertNotEqual(first["goal_id"], second["goal_id"])
        self.assertEqual(self.goals.load(first)["documents"]["benchmark"]["object_ref"], self.goals.load(second)["documents"]["benchmark"]["object_ref"])

    def test_real_broken_references_keep_original_text_and_full_model_requirements(self):
        relative = self.copy_example("examples/Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning")
        instance = self.create(relative)
        self.assertEqual(instance["task_spec"]["status"], "incomplete")
        original = copy.deepcopy(instance["task_spec"])
        goal = self.goals.load(instance)
        self.assertEqual(goal["status"], "complete", goal["unresolved"])
        for field in ("benchmarkingjob/testenv", "benchmarkingjob/test_object/algorithms/0/url", "testenv/metrics/0/url"):
            item = next(item for item in goal["bindings"] if item["field"] == field)
            self.assertEqual(item["binding_basis"], "approved_unique_example_suffix")
            self.assertEqual(item["sha256"], instance["source_manifest"][item["path"]])
        algorithm = goal["requirements"]["algorithms"][0]["document"]["algorithm"]
        expected = yaml.safe_load((self.source / relative / "testalgorithms/perception_reasoning.yaml").read_text(encoding="utf-8"))["algorithm"]
        self.assertEqual(algorithm, expected)
        self.assertEqual(instance["task_spec"], original)
        self.assertIn("模型加载", goal["limitations"][0])

    def test_multiple_same_example_full_suffix_candidates_require_binding(self):
        relative = self.copy_example("examples/Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning")
        nested = self.source / relative / "conflict/testenv/testenv.yaml"
        nested.parent.mkdir(parents=True)
        shutil.copy2(self.source / relative / "testenv/testenv.yaml", nested)
        instance = self.create(relative)
        goal = self.goals.load(instance)
        self.assertEqual(goal["status"], "needs_binding")
        problem = next(item for item in goal["unresolved"] if item["field"] == "benchmarkingjob/testenv")
        self.assertEqual(problem["reason"], "ambiguous_reference")
        self.assertEqual(len(problem["candidates"]), 2)
        self.assertEqual(self.goals.resolve(instance)["goal_id"], goal["goal_id"])

    def test_complete_suffix_preserves_real_reid_and_tracking_subdirectories(self):
        relative = self.copy_example("examples/MOT17/multiedge_inference_bench/pedestrian_tracking")
        instance = self.create(relative, "reid_job.yaml")
        goal = self.goals.load(instance)
        self.assertEqual(goal["documents"]["testenv"]["path"], relative + "/testenv/reid/testenv.yaml")
        manifest = instance["source_manifest"]
        outside = "examples/other/testenv/reid/testenv.yaml"
        manifest = {**manifest, outside: manifest[relative + "/testenv/reid/testenv.yaml"]}
        declared = "./examples/missing/testenv/reid/testenv.yaml"
        candidates, basis = self.goals._candidates(instance, declared, "testenv", manifest)
        self.assertEqual(candidates, [relative + "/testenv/reid/testenv.yaml"])
        self.assertEqual(basis, "approved_unique_example_suffix")

    def test_legacy_source_root_requires_exact_creation_bytes(self):
        self.copy_example()
        instance = self.legacy(self.create())
        benchmark = instance["benchmark_path"]
        (Path(instance["code_root"]) / benchmark).write_text("changed: true\n", encoding="utf-8")
        goal = self.goals.resolve(instance)
        self.assertEqual(goal["status"], "complete")
        self.assertEqual(goal["documents"]["benchmark"]["source_kind"], "legacy_source_file")
        self.assertEqual(goal["documents"]["benchmark"]["sha256"], instance["source_manifest"][benchmark])
        self.assertEqual(self.goals.resolve(instance)["goal_id"], goal["goal_id"])

    def test_module_source_is_bound_as_code_and_can_change_without_freezing_model_artifacts(self):
        self.copy_example()
        instance = self.create()
        goal = self.goals.load(instance)
        name = "examples/llm_simple_qa/testalgorithms/gen/basemodel.py"
        binding = next(item for item in goal["bindings"] if item["path"] == name)
        self.assertIn("/modules/", binding["field"])
        self.assertNotIn(name, {item["path"] for item in goal["artifact_bindings"]})
        target = Path(instance["code_root"]) / name
        target.write_bytes(target.read_bytes() + b"\n# Source repair verification\n")
        self.assertEqual(self.goals.load(instance), goal)
        self.assertNotEqual(hashlib.sha256(target.read_bytes()).hexdigest(), binding["sha256"])
        checks = compare_goal(instance, goal, configuration_evidence(instance["code_root"], instance["benchmark_path"]), file_manifest(instance["code_root"]))
        self.assertTrue(all(checks.values()), checks)

    def test_unique_suffix_module_url_repair_retains_same_algorithm_goal(self):
        relative = self.copy_example("examples/Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning")
        instance = self.create(relative)
        goal = self.goals.load(instance)
        root = Path(instance["code_root"])
        benchmark = copy.deepcopy(goal["documents"]["benchmark"]["document"])
        benchmark["benchmarkingjob"]["testenv"] = goal["documents"]["testenv"]["path"]
        algorithm = goal["documents"]["algorithms"][0]
        benchmark["benchmarkingjob"]["test_object"]["algorithms"][0]["url"] = algorithm["path"]
        (root / instance["benchmark_path"]).write_text(yaml.safe_dump(benchmark), encoding="utf-8")
        document = copy.deepcopy(algorithm["document"])
        for index, module in enumerate(document["algorithm"]["modules"]):
            binding = next(item for item in goal["bindings"] if item["field"] == f"algorithms/0/algorithm/modules/{index}/url")
            module["url"] = binding["path"]
        (root / algorithm["path"]).write_text(yaml.safe_dump(document), encoding="utf-8")
        testenv = copy.deepcopy(goal["documents"]["testenv"]["document"])
        for index, metric in enumerate(testenv["testenv"]["metrics"]):
            metric["url"] = next(item["path"] for item in goal["bindings"] if item["field"] == f"testenv/metrics/{index}/url")
        (root / goal["documents"]["testenv"]["path"]).write_text(yaml.safe_dump(testenv), encoding="utf-8")
        evidence = configuration_evidence(root, instance["benchmark_path"])
        self.assertEqual(evidence["status"], "complete")
        checks = compare_goal(instance, goal, evidence, file_manifest(root))
        self.assertTrue(all(checks.values()), checks)

    def test_existing_model_directory_binds_complete_creation_file_set(self):
        self.copy_example()
        directory = self.source / "local_model"
        directory.mkdir()
        (directory / "config.json").write_text('{"model_type": "bert", "hidden_size": 8}\n', encoding="utf-8")
        shutil.copy2(self.project / "examples/llm_simple_qa/testenv/acc.py", directory / "model_notes.py")
        path = self.source / "examples/llm_simple_qa/testalgorithms/gen/gen_algorithm.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["algorithm"]["model_path"] = "local_model"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        instance = self.create()
        goal = self.goals.load(instance)
        members = [item for item in goal["artifact_bindings"] if item.get("artifact_directory") == "local_model"]
        self.assertEqual({item["path"] for item in members}, {"local_model/config.json", "local_model/model_notes.py"})
        root = Path(instance["code_root"])
        evidence = configuration_evidence(root, instance["benchmark_path"])
        self.assertTrue(all(compare_goal(instance, goal, evidence, file_manifest(root)).values()))
        (root / "local_model/config.json").write_text('{"model_type": "bert", "hidden_size": 16}\n', encoding="utf-8")
        self.assertFalse(compare_goal(instance, goal, evidence, file_manifest(root))["创建时已有模型与数据文件保持一致"])
        shutil.copy2(directory / "config.json", root / "local_model/config.json")
        (root / "local_model/extra.json").write_text("{}\n", encoding="utf-8")
        self.assertFalse(compare_goal(instance, goal, evidence, file_manifest(root))["创建时已有模型与数据文件保持一致"])
        self.assertEqual(self.goals.load(instance), goal)

    def test_arbitrary_algorithm_string_path_is_preserved_without_becoming_model_artifact(self):
        self.copy_example()
        path = self.source / "examples/llm_simple_qa/testalgorithms/gen/gen_algorithm.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["algorithm"]["description"] = "examples/llm_simple_qa/testalgorithms/gen/basemodel.py"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        instance = self.create()
        goal = self.goals.load(instance)
        self.assertEqual(goal["requirements"]["algorithms"][0]["document"], document)
        self.assertFalse(any(item["field"].endswith("/description") for item in goal["artifact_bindings"]))

    def test_existing_model_directory_rejects_added_symbolic_members(self):
        self.copy_example()
        directory = self.source / "local_model"
        directory.mkdir()
        shutil.copy2(self.project / "examples/llm_simple_qa/testenv/acc.py", directory / "model_notes.py")
        path = self.source / "examples/llm_simple_qa/testalgorithms/gen/gen_algorithm.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["algorithm"]["model_path"] = "local_model"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        instance = self.create()
        goal = self.goals.load(instance)
        root = Path(instance["code_root"])
        evidence = configuration_evidence(root, instance["benchmark_path"])
        manifest = file_manifest(root)
        target = root / "local_model/model_notes.py"
        links = ((target, False), (root / "examples/llm_simple_qa/testenv", True), (root / "unavailable-model-member", False))
        for index, (destination, is_directory) in enumerate(links):
            with self.subTest(destination=destination, is_directory=is_directory):
                link = root / "local_model" / f"additional-{index}"
                try:
                    link.symlink_to(destination, target_is_directory=is_directory)
                except OSError as error:
                    if os.name == "nt" and error.winerror in (5, 1314):
                        self.skipTest("Windows 当前权限不允许创建符号链接")
                    raise
                try:
                    self.assertEqual(file_manifest(root), manifest)
                    checks = compare_goal(instance, goal, evidence, file_manifest(root))
                    self.assertFalse(checks["创建时已有模型与数据文件保持一致"])
                finally:
                    link.unlink()
                self.assertTrue(all(compare_goal(instance, goal, evidence, file_manifest(root)).values()))

    def test_initial_task_text_conflict_requires_explicit_binding_and_cannot_complete(self):
        self.copy_example()
        instance = self.legacy(self.create())
        parsed = yaml.safe_load(instance["task_spec"]["testenv_text"])
        parsed["testenv"]["dataset"]["test_data"] = "different-original-data.jsonl"
        instance["task_spec"]["testenv_text"] = yaml.safe_dump(parsed)
        goal = self.goals.resolve(instance)
        self.assertEqual(goal["status"], "needs_binding")
        self.assertTrue(any(item["reason"] == "initial_record_conflict" for item in goal["unresolved"]))
        self.assertEqual(self.goals.load(instance), goal)

    def test_initial_task_text_recovers_only_exact_creation_bytes(self):
        self.copy_example()
        names = ("examples/llm_simple_qa/benchmarkingjob.yaml", "examples/llm_simple_qa/testenv/testenv.yaml")
        for name in names:
            path = self.source / name
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        instance = self.legacy(self.create())
        for name in names:
            for root in (self.source, Path(instance["code_root"])):
                (root / name).write_text("changed: true\n", encoding="utf-8")
        goal = self.goals.resolve(instance)
        self.assertEqual(goal["status"], "complete", goal["unresolved"])
        for key in ("benchmark", "testenv"):
            self.assertEqual(goal["documents"][key]["source_kind"], "initial_task_text")
            self.assertEqual(goal["documents"][key]["sha256"], instance["source_manifest"][goal["documents"][key]["path"]])

    def test_normalized_initial_task_text_cannot_replace_missing_crlf_original_bytes(self):
        self.copy_example()
        name = "examples/llm_simple_qa/benchmarkingjob.yaml"
        path = self.source / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        instance = self.legacy(self.create())
        self.assertNotEqual(hashlib.sha256(instance["task_spec"]["benchmark_text"].encode("utf-8")).hexdigest(), instance["source_manifest"][name])
        for root in (self.source, Path(instance["code_root"])):
            (root / name).write_text("changed: true\n", encoding="utf-8")
        goal = self.goals.resolve(instance)
        self.assertEqual(goal["status"], "source_missing")
        self.assertIsNone(goal["documents"]["benchmark"])

    def test_unsupported_yaml_document_retains_source_without_original_metrics_exception(self):
        self.copy_example()
        path = self.source / "examples/llm_simple_qa/testenv/testenv.yaml"
        path.write_text("- unsupported-testenv-shape\n", encoding="utf-8")
        instance = self.create()
        goal = self.goals.load(instance)
        self.assertEqual(goal["status"], "unsupported", goal["unresolved"])
        self.assertIsNone(goal["documents"]["testenv"]["document"])
        self.assertEqual(original_metrics(goal), [])

    def test_real_grouped_metrics_remain_explicitly_unsupported(self):
        relative = self.copy_example("examples/PIPL/edge-cloud_collaborative_learning_bench")
        instance = self.create(relative)
        goal = self.goals.load(instance)
        self.assertNotEqual(goal["status"], "complete", goal["unresolved"])
        self.assertIsInstance(goal["requirements"]["metrics"], dict)
        self.assertTrue(any(item["reason"] == "unsupported_metrics" for item in goal["unresolved"]))

    def copy_builtin_metrics(self):
        name = "core/testcasecontroller/metrics/metrics.py"
        target = self.source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.project / name, target)

    def test_real_glfc_process_model_metric_is_protected_without_becoming_rank_metric(self):
        relative = self.copy_example("examples/cifar100/fci_ssl/glfc")
        self.copy_builtin_metrics()
        instance = self.create(relative)
        goal = self.goals.load(instance)
        self.assertEqual(goal["status"], "complete", goal["unresolved"])
        expected = {item["name"] for item in original_metrics(goal)}
        self.assertEqual(expected, {"forget_rate", "task_avg_acc"})
        metric = next(item for item in goal["metric_sources"] if item["field"] == "testenv/model_eval/model_metric/url")
        self.assertEqual(metric["name"], "accuracy")
        self.assertEqual(metric["metric_kind"], "custom_metric")
        root = Path(instance["code_root"])
        target = root / metric["path"]
        target.write_bytes(target.read_bytes() + b"\n# Process metric source changed\n")
        checks = compare_goal(instance, goal, configuration_evidence(root, instance["benchmark_path"]), file_manifest(root))
        self.assertFalse(checks["原始评测指标实现保持一致"])

    def test_real_semantic_segmentation_process_metric_unique_suffix_is_bound(self):
        relative = self.copy_example("examples/Cloud_Robotics/singletask_learning_bench/Semantic_Segmentation")
        self.copy_builtin_metrics()
        instance = self.create(relative)
        goal = self.goals.load(instance)
        self.assertEqual(goal["status"], "source_missing", goal["unresolved"])
        self.assertEqual(goal["unresolved"], [{"field": "benchmarkingjob/test_object/algorithms/0/url", "declared": "./examples/cloud-robotics/single_task_learning/testalgorithm/rfnet/algorithm.yaml", "reason": "creation_source_missing", "candidates": []}])
        metric = next(item for item in goal["metric_sources"] if item["field"] == "testenv/model_eval/model_metric/url")
        self.assertEqual(metric["binding_basis"], "approved_unique_example_suffix")
        self.assertEqual(metric["path"], relative + "/testenv/accuracy.py")
        original = goal["documents"]["testenv"]["document"]["testenv"]["model_eval"]
        self.assertEqual(original["threshold"], 0)
        self.assertEqual(original["operator"], "<")
        self.assertEqual(original["model_metric"]["mode"], "no-inference")
        self.assertNotEqual(metric["declared"], next(item["declared"] for item in goal["metric_sources"] if item["field"] == "testenv/metrics/0/url"))

    def test_invalid_model_metric_type_is_preserved_as_unsupported(self):
        self.copy_example()
        path = self.source / "examples/llm_simple_qa/testenv/testenv.yaml"
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        value["testenv"]["model_eval"] = {"model_metric": ["accuracy"]}
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        instance = self.create()
        goal = self.goals.load(instance)
        self.assertEqual(goal["status"], "unsupported")
        self.assertTrue(any(item["field"] == "testenv/model_eval/model_metric" and item["reason"] == "unsupported_metrics" for item in goal["unresolved"]))

    def test_missing_legacy_source_can_recover_new_version_without_rewriting_old_goal(self):
        self.copy_example()
        instance = self.legacy(self.create())
        name = "examples/llm_simple_qa/testenv/acc.py"
        original = (self.source / name).read_bytes()
        for root in (self.source, Path(instance["code_root"])):
            (root / name).write_text("changed = True\n", encoding="utf-8")
        failed = self.goals.resolve(instance)
        self.assertEqual(failed["status"], "source_missing")
        failed_bytes = self.store.path(failed["evidence_ref"]).read_bytes()
        self.assertEqual(self.goals.resolve(instance)["goal_id"], failed["goal_id"])
        (self.source / name).write_bytes(original)
        recovered = self.goals.resolve(instance)
        self.assertEqual(recovered["status"], "complete")
        self.assertEqual(recovered["previous_goal_id"], failed["goal_id"])
        self.assertEqual(recovered["previous_goal_ref"], failed["evidence_ref"])
        self.assertNotEqual(recovered["goal_id"], failed["goal_id"])
        self.assertEqual(self.store.path(failed["evidence_ref"]).read_bytes(), failed_bytes)
        self.assertEqual(self.goals.load(instance, failed["evidence_ref"]), failed)
        self.assertTrue(instance["needs_verification"])

    def test_missing_creation_manifest_cannot_relabel_current_files_as_original(self):
        self.copy_example()
        instance = self.legacy(self.create())
        instance["source_manifest"] = {}
        goal = self.goals.resolve(instance)
        self.assertEqual(goal["status"], "source_missing")
        self.assertIsNone(goal["source_manifest_ref"])
        self.assertIsNone(goal["documents"]["benchmark"])
        self.assertTrue(any(item["reason"] == "creation_manifest_missing" for item in goal["unresolved"]))
        self.assertEqual(self.goals.load(instance), goal)

    def test_object_corruption_and_foreign_instance_references_fail_closed(self):
        self.copy_example()
        instance = self.create()
        goal = self.goals.load(instance)
        second = self.create()
        with self.assertRaisesRegex(PermissionError, "当前实例"):
            self.goals.load(second, goal["evidence_ref"])
        artifact = self.store.path(goal["documents"]["benchmark"]["object_ref"])
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "损坏"):
            self.goals.load(instance)
        with self.assertRaisesRegex(ValueError, "损坏"):
            self.goals.resolve(instance)

    def test_source_manifest_corruption_and_record_digest_mismatch_fail_closed(self):
        self.copy_example()
        instance = self.create()
        goal = self.goals.load(instance)
        path = self.store.path(goal["source_manifest_ref"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value["instance_id"] = "instance-invalid"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "身份"):
            self.goals.load(instance)
        self.store.write(goal["source_manifest_ref"], {**value, "instance_id": instance["instance_id"]})
        record = self.store.read(goal["evidence_ref"])
        record["requirements"]["dataset"] = {}
        self.store.write(goal["evidence_ref"], record)
        with self.assertRaisesRegex(ValueError, "摘要"):
            self.goals.load(instance)

    def test_scope_rejects_symlink_object_and_unnormalized_reference(self):
        self.copy_example()
        instance = self.create()
        for reference in ("../outside.json", "/outside.json", "C:/outside.json", "instances/../outside.json"):
            with self.subTest(reference=reference), self.assertRaises((PermissionError, ValueError)):
                self.goals.load(instance, reference)
        if os.name == "nt":
            return
        goal = self.goals.load(instance)
        path = self.store.path(goal["documents"]["benchmark"]["object_ref"])
        external = self.case_root / "external.bin"
        shutil.copy2(path, external)
        path.unlink()
        path.symlink_to(external)
        with self.assertRaisesRegex(PermissionError, "程序管理目录"):
            self.goals.load(instance)


if __name__ == "__main__":
    unittest.main()
