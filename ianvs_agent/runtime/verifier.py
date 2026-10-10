# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
import csv
import json
import math
import sys
from pathlib import Path
from datetime import datetime

import yaml

from .instances import configuration_evidence, file_manifest, utc_now, validate_identity
from .packages import package_identity
from .processes import recorded_command


SUPPORTED_ACCEPTANCE_FIELDS = {"original_benchmark_path", "required_metrics", "required_checks", "metric_thresholds", "file_identities", "repair_task_id", "repair_started_at", "latest_patch_at"}


class Verifier:
    def __init__(self, settings, store, events):
        self.settings, self.store, self.events = settings, store, events

    def _check_instance(self, instance):
        instance_id = validate_identity(instance["instance_id"], "实例")
        saved = self.store.read(f"instances/{instance_id}/instance.json")
        if saved is None or saved.get("instance_id") != instance_id:
            raise ValueError("验证实例没有对应的正式记录")
        fields = ("code_root", "source_manifest", "task_spec", "environment")
        if any(instance.get(field) != saved.get(field) for field in fields):
            raise ValueError("验证实例与正式源码、原始任务或环境记录不一致")

    async def validate(self, instance):
        self._check_instance(instance)
        python = sys.executable
        validation_id = self.store.allocate_id("validation")
        relative = f"instances/{instance['instance_id']}/validations/{validation_id}"
        entry = instance.get("inventory_entry") or {"name": instance["name"], "path": str(Path(instance["benchmark_path"]).parent), "benchmark_file": instance["benchmark_path"], "readme_file": instance.get("readme_path")}
        self.store.write(f"{relative}/entry.json", entry)
        directory = self.settings.state_root / relative
        helper = Path(__file__).with_name("validation_worker.py")
        await recorded_command([python, str(helper), "--repo-root", instance["code_root"], "--validator-root", str(self.settings.project_root), "--entry", str(directory / "entry.json"), "--report", str(directory / "report.json")], self.settings.project_root, python, directory / "validator.log", timeout=120)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        checks = [{**check, "source_kind": "static_validation", "evidence_ref": f"{relative}/report.json"} for example in report["examples"] for check in example["checks"]]
        required = {"Example directory exists", "benchmarkingjob.yaml exists", "YAML syntax", "Repository path references exist"}
        statuses = {check["name"]: check["status"] for check in checks}
        passed = all(statuses.get(name) == "PASS" for name in required) and not any(check["status"] in {"FAIL", "ERROR"} for check in checks)
        return {"passed": passed, "checks": checks, "report_ref": f"{relative}/report.json", "evidence_type": "real_static_validation"}

    @staticmethod
    def _metric_equal(value, text):
        try:
            if isinstance(value, (int, float)):
                number = float(text)
                return math.isfinite(number) and math.isfinite(value) and math.isclose(value, number, rel_tol=1e-9, abs_tol=1e-12)
            parsed = ast.literal_eval(text) if not isinstance(value, str) else text
            return parsed == value
        except (ValueError, SyntaxError, TypeError):
            return False

    def _original_metrics(self, instance):
        specification = instance.get("task_spec", {})
        if specification.get("status") in {"pending", "incomplete"} or not specification.get("testenv_text"):
            return []
        testenv = self._mapping(self._document(specification["testenv_text"]).get("testenv"))
        metrics = testenv.get("metrics")
        return metrics if isinstance(metrics, list) and all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in metrics) else []

    @staticmethod
    def _reference(root, value):
        return (Path(root) / value).resolve() if isinstance(value, str) and value else None

    @staticmethod
    def _mapping(value):
        return value if isinstance(value, dict) else {}

    def _document(self, text):
        try:
            return self._mapping(yaml.safe_load(text or "{}"))
        except yaml.YAMLError:
            return {}

    def _frozen_reference(self, instance, value, known_files):
        root = Path(instance["code_root"]).resolve()
        path = self._reference(root, value)
        return path is not None and path.is_relative_to(root) and path.relative_to(root).as_posix() in known_files

    def _goal_bindings(self, instance, run, acceptance, evidence, original_metrics):
        root = instance["code_root"]
        known_files = set(instance["source_manifest"]) | set(run.get("source_manifest") or {}) | set(acceptance.get("file_identities", {}))
        current_testenv = self._mapping(self._mapping(evidence.get("testenv")).get("testenv"))
        observed_metrics = current_testenv.get("metrics", [])
        metric_bindings = isinstance(observed_metrics, list)
        if metric_bindings:
            for expected in original_metrics:
                matching = [item for item in observed_metrics if isinstance(item, dict) and item.get("name") == expected["name"]]
                if len(matching) != 1 or self._reference(root, matching[0].get("url")) != self._reference(root, expected.get("url")):
                    metric_bindings = False
        specification = instance.get("task_spec", {})
        original_benchmark = self._document(specification.get("benchmark_text"))
        original_job = self._mapping(original_benchmark.get("benchmarkingjob"))
        current_job = self._mapping(self._mapping(evidence.get("benchmark")).get("benchmarkingjob"))
        expected_object = self._mapping(original_job.get("test_object"))
        current_object = self._mapping(current_job.get("test_object"))
        algorithm_bindings = expected_object.get("type") == current_object.get("type")
        observed_algorithms = current_object.get("algorithms", [])
        expected_algorithms = expected_object.get("algorithms", [])
        algorithm_bindings = algorithm_bindings and isinstance(expected_algorithms, list) and isinstance(observed_algorithms, list)
        if algorithm_bindings:
            for expected in expected_algorithms:
                if not isinstance(expected, dict) or not isinstance(expected.get("name"), str):
                    algorithm_bindings = False
                    continue
                matching = [item for item in observed_algorithms if isinstance(item, dict) and item.get("name") == expected["name"]]
                if len(matching) != 1 or self._frozen_reference(instance, expected.get("url"), known_files) and self._reference(root, matching[0].get("url")) != self._reference(root, expected.get("url")):
                    algorithm_bindings = False
        original_testenv = self._mapping(self._document(specification.get("testenv_text")).get("testenv"))
        expected_dataset = self._mapping(original_testenv.get("dataset"))
        current_dataset = current_testenv.get("dataset", {})
        dataset_bindings = isinstance(current_dataset, dict)
        if dataset_bindings:
            for name, value in expected_dataset.items():
                if self._frozen_reference(instance, value, known_files) and self._reference(root, current_dataset.get(name)) != self._reference(root, value):
                    dataset_bindings = False
        return metric_bindings, algorithm_bindings, dataset_bindings

    async def verify(self, instance, run, acceptance=None):
        self._check_instance(instance)
        instance_id = instance["instance_id"]
        run_id = validate_identity(run["run_id"], "运行")
        if run.get("instance_id") != instance_id:
            raise ValueError("本次 Run 不属于验证实例")
        saved = self.store.read(f"instances/{instance_id}/runs/{run_id}/run.json")
        if saved != run:
            raise ValueError("本次 Run 与正式运行记录不一致")
        acceptance = acceptance or {}
        reasons, checks = [], []
        unsupported = sorted(set(acceptance) - SUPPORTED_ACCEPTANCE_FIELDS)
        for name in unsupported:
            checks.append({"name": f"未支持的正式验收字段：{name}", "status": "SKIP", "acceptance_field": name, "message": "当前验证器没有实现该条件，不能声称原始任务已通过验收"})
        if unsupported:
            reasons.append(f"正式验收条件尚未实现独立检查：{', '.join(unsupported)}")
        benchmark_matches = "original_benchmark_path" not in acceptance or acceptance["original_benchmark_path"] == instance["benchmark_path"]
        if "original_benchmark_path" in acceptance:
            checks.append({"name": "原始 benchmark 路径保持一致", "status": "PASS" if benchmark_matches else "FAIL"})
            if not benchmark_matches:
                reasons.append("当前实例 benchmark 路径不符合冻结的原始任务")
        self.events.emit("verification.started", {"run_id": run["run_id"]}, instance_id=instance["instance_id"], run_id=run["run_id"])
        execution_completed = run["status"] == "completed" and run["exit_code"] == 0 and not run["timed_out"] and not run["interrupted"]
        if not execution_completed:
            reasons.append("本轮没有正常完成执行")
        original_metrics = self._original_metrics(instance)
        goal_available = bool(original_metrics) or bool(acceptance.get("required_metrics"))
        if not goal_available:
            reasons.append("冻结的原始验收目标缺失，需要明确原始指标与正确性要求")
        required_metrics = list(dict.fromkeys([item["name"] for item in original_metrics] + acceptance.get("required_metrics", [])))
        testcases = {item["testcase_id"] for item in run["testcases"]}
        metrics = run["metrics"]
        completed = {event["payload"]["testcase_id"] for event in run["events"] if event["event_type"] == "testcase.completed"}
        results_complete = bool(testcases) and completed == testcases and set(metrics) == testcases and all(set(required_metrics).issubset(values) for values in metrics.values())
        pipeline_completed = any(event["event_type"] == "run.pipeline_completed" for event in run["events"])
        results_complete = results_complete and pipeline_completed
        if not results_complete:
            reasons.append("当前 Run 缺少完整 TestCase、原始任务指标或流程结束证据")
        rank_events = [event for event in run["events"] if event["event_type"] == "rank.completed"]
        rank_checked = False
        if rank_events and results_complete:
            rank_path = Path(rank_events[-1]["payload"]["all_rank_file"]).resolve()
            output = Path(run["output_dir"]).resolve()
            if rank_path.is_relative_to(output) and str(rank_path) in run["artifacts"] and rank_path.is_file():
                with rank_path.open(encoding="utf-8", newline="") as source:
                    rows = list(csv.DictReader(source))
                row_ids = {Path(row["url"]).name for row in rows}
                rank_checked = len(rows) == len(testcases) and row_ids == testcases
                for row in rows:
                    case_id = Path(row["url"]).name
                    if not Path(row["url"]).resolve().is_relative_to(output):
                        rank_checked = False
                    if case_id not in metrics:
                        rank_checked = False
                        continue
                    for name in required_metrics:
                        if name not in row or not self._metric_equal(metrics[case_id][name], row[name]):
                            rank_checked = False
        if not rank_checked:
            results_complete = False
            reasons.append("本轮 Rank 产物与实际 TestCase、指标或输出目录不一致")
        static = await self.validate(instance)
        checks.extend(static["checks"])
        environment = instance["environment"]
        package_log = self.settings.state_root / f"instances/{instance['instance_id']}/runs/{run['run_id']}/verified-packages.json"
        await recorded_command([environment["python_executable"], "-m", "pip", "--disable-pip-version-check", "list", "--format=json"], instance["code_root"], environment["python_executable"], package_log)
        observed_packages = json.loads(package_log.read_text(encoding="utf-8"))
        pip_check_log = package_log.with_name("verified-pip-check.log")
        pip_check = await recorded_command([environment["python_executable"], "-m", "pip", "--disable-pip-version-check", "check"], instance["code_root"], environment["python_executable"], pip_check_log, check=False)
        snapshot = run.get("environment_snapshot") or {}
        environment_ready = environment.get("status") == "ready" and snapshot.get("status") == "ready" and pip_check["exit_code"] == 0
        checks.append({"name": "运行环境已准备且依赖检查通过", "status": "PASS" if environment_ready else "FAIL", "evidence_ref": pip_check_log.relative_to(self.store.root).as_posix()})
        if not environment_ready:
            reasons.append("当前或运行时环境尚未准备完成，或 pip check 没有通过")
        observed_identity = package_identity(observed_packages)
        environment_unchanged = (run["environment_id"] == environment["environment_id"]
                                 and environment["python_executable"] == snapshot.get("python_executable")
                                 and observed_identity is not None
                                 and observed_identity == package_identity(snapshot.get("packages")))
        checks.append({"name": "执行环境与运行时依赖保持一致", "status": "PASS" if environment_unchanged else "FAIL"})
        if not environment_unchanged:
            reasons.append("解释器或已安装依赖与本次运行环境不一致")
        current = await asyncio.to_thread(file_manifest, instance["code_root"])
        unchanged = current == run.get("source_manifest")
        checks.append({"name": "运行后源码与数据保持一致", "status": "PASS" if unchanged else "FAIL"})
        if not unchanged:
            reasons.append("运行后文件已经变化，当前结果需要重新验证")
        metric_identity = True
        for item in original_metrics:
            if item.get("url"):
                relative = str(item["url"]).removeprefix("./")
                if current.get(relative) != instance["source_manifest"].get(relative):
                    metric_identity = False
        checks.append({"name": "原始评测指标实现保持一致", "status": "PASS" if metric_identity else "FAIL"})
        if not metric_identity:
            reasons.append("评测指标实现与原始任务不一致")
        evidence = await asyncio.to_thread(configuration_evidence, instance["code_root"], instance["benchmark_path"])
        metric_bindings, algorithm_bindings, dataset_bindings = self._goal_bindings(instance, run, acceptance, evidence, original_metrics)
        for name, matches in (("原始指标名称与实现来源保持一致", metric_bindings), ("原始算法目标与配置来源保持一致", algorithm_bindings), ("冻结的原始数据来源保持一致", dataset_bindings)):
            checks.append({"name": name, "status": "PASS" if matches else "FAIL"})
            if not matches:
                reasons.append(name.replace("保持一致", "被替换"))
        protected = ["core/common/events.py", "core/testcasecontroller/testcase/testcase.py", "core/storymanager/rank/rank.py"]
        audit_identity = all(instance["source_manifest"].get(path) is not None and current.get(path) == instance["source_manifest"][path] for path in protected)
        checks.append({"name": "原始审计与评分生成代码保持一致", "status": "PASS" if audit_identity else "FAIL"})
        if not audit_identity:
            reasons.append("审计或评分生成代码与冻结身份不一致")
        checks.append({"name": "原始验收目标存在", "status": "PASS" if goal_available else "SKIP"})
        checks_passed = static["passed"] and unchanged and metric_identity and metric_bindings and algorithm_bindings and dataset_bindings and environment_ready and environment_unchanged and audit_identity and goal_available and benchmark_matches and not unsupported
        for name in acceptance.get("required_checks", []):
            matching = [check for check in checks if check["name"] == name]
            if not matching or not all(check["status"] == "PASS" for check in matching):
                checks_passed = False
                reasons.append(f"必需检查没有通过：{name}")
        goal_achieved = execution_completed and results_complete and checks_passed
        for name, limits in acceptance.get("metric_thresholds", {}).items():
            for case_id, values in metrics.items():
                value = values.get(name)
                within = isinstance(value, (int, float)) and math.isfinite(value) and value >= limits.get("min", -math.inf) and value <= limits.get("max", math.inf)
                if not within:
                    goal_achieved = False
                    reasons.append(f"TestCase {case_id} 的 {name} 没有满足原始验收条件")
        for path, digest in acceptance.get("file_identities", {}).items():
            if current.get(path) != digest:
                goal_achieved = False
                reasons.append(f"任务文件身份不一致：{path}")
        passed = goal_achieved
        repair_run_fresh = False
        if acceptance.get("repair_task_id"):
            boundaries = [acceptance.get("repair_started_at"), acceptance.get("latest_patch_at")]
            if boundaries[0]:
                started = datetime.fromisoformat(run["started_at"])
                repair_run_fresh = all(started > datetime.fromisoformat(boundary) for boundary in boundaries if boundary)
            checks.append({"name": "修复复验 Run 在任务与最新修改之后启动", "status": "PASS" if repair_run_fresh else "FAIL"})
            if not repair_run_fresh:
                reasons.append("当前 Run 早于修复任务或最新修改，不能证明本次修复完成")
        latest_instance = self.store.read(f"instances/{instance['instance_id']}/instance.json")
        current_run = latest_instance.get("last_run_id") == run["run_id"]
        if not current_run:
            reasons.append("该验证属于历史 Run，最新 Run 仍需独立验证")
        result = {"run_id": run["run_id"], "instance_id": instance["instance_id"], "passed": passed, "execution_completed": execution_completed, "results_complete": results_complete, "checks_passed": checks_passed, "goal_achieved": goal_achieved, "repair_completed": passed and bool(acceptance.get("repair_task_id")) and repair_run_fresh and current_run, "checks": checks, "reasons": reasons, "unsupported_acceptance_fields": unsupported, "evidence_type": "real_ianvs_run", "verified_at": utc_now(), "static_report_ref": static["report_ref"], "acceptance": acceptance, "stale": not current_run}
        relative = f"instances/{instance['instance_id']}/runs/{run['run_id']}/verification.json"
        self.store.write(relative, result)
        if current_run:
            instance["verification"] = result
            instance["needs_verification"] = not passed
            instance["status"] = "verified" if passed else "verification_failed"
            instance["current_manifest"] = current
            self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        else:
            instance.update(latest_instance)
        self.events.emit("verification.completed", result, instance_id=instance["instance_id"], run_id=run["run_id"])
        return result
