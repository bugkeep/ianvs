# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import yaml


def goal_summary(goal):
    if not goal:
        return {"status": "source_missing", "unresolved": [{"reason": "original_goal_not_registered"}]}
    result = {key: goal.get(key) for key in ("goal_id", "goal_digest", "evidence_ref", "status", "requirements", "supported_checks", "limitations")}
    unresolved = goal.get("unresolved", [])
    result.update(unresolved=unresolved[:20], unresolved_count=len(unresolved), unresolved_truncated=len(unresolved) > 20)
    return result


def original_metrics(goal):
    document = (goal.get("documents", {}).get("testenv") or {}).get("document") or {}
    testenv = document.get("testenv", {})
    metrics = testenv.get("metrics") if isinstance(testenv, dict) else None
    return metrics if isinstance(metrics, list) and all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in metrics) else []


def _directory_has_symbolic_links(root, directory):
    path = root
    for part in Path(directory).parts:
        path = path / part
        if path.is_symlink():
            return True
    if not path.is_dir():
        return False

    def fail(error):
        raise error

    for current, directories, files in os.walk(path, followlinks=False, onerror=fail):
        if any((Path(current) / name).is_symlink() for name in directories + files):
            return True
    return False


def compare_goal(instance, goal, evidence, current_manifest):
    root = Path(instance["code_root"]).resolve()
    documents = goal.get("documents") or {}
    paths = set(instance["source_manifest"])
    aliases = {}
    for binding in goal.get("bindings", []):
        aliases.setdefault(binding["declared"], set()).add(binding["path"])
        paths.add(binding["path"])

    def canonical(value, original=False):
        if isinstance(value, dict):
            return {key: canonical(item, original) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item, original) for item in value]
        if not isinstance(value, str):
            return value
        matches = aliases.get(value, set()) if original else set()
        if len(matches) == 1:
            return next(iter(matches))
        if "\x00" in value or "://" in value:
            return value
        path = (root / value).resolve()
        if path.is_relative_to(root):
            relative = path.relative_to(root).as_posix()
            if relative in paths:
                return relative
        return value

    def mapping(value):
        return value if isinstance(value, dict) else {}

    expected_benchmark = mapping((documents.get("benchmark") or {}).get("document"))
    expected_job = mapping(expected_benchmark.get("benchmarkingjob"))
    current_job = mapping(mapping(evidence.get("benchmark")).get("benchmarkingjob"))
    expected_testenv = mapping(mapping((documents.get("testenv") or {}).get("document")).get("testenv"))
    current_testenv = mapping(mapping(evidence.get("testenv")).get("testenv"))
    metrics_match = canonical(expected_testenv.get("metrics"), True) == canonical(current_testenv.get("metrics"))
    dataset_match = canonical(expected_testenv.get("dataset"), True) == canonical(current_testenv.get("dataset"))
    testenv_match = canonical(expected_testenv, True) == canonical(current_testenv)
    testenv_source_match = canonical(expected_job.get("testenv"), True) == canonical(current_job.get("testenv"))
    algorithm_match = canonical(expected_job.get("test_object"), True) == canonical(current_job.get("test_object"))
    ranking_match = canonical(mapping(expected_job.get("rank")).get("sort_by"), True) == canonical(mapping(current_job.get("rank")).get("sort_by"))
    for algorithm in documents.get("algorithms", []):
        path = (root / algorithm["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            algorithm_match = False
            continue
        try:
            current = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeError):
            algorithm_match = False
            continue
        if canonical(algorithm["document"], True) != canonical(current):
            algorithm_match = False
    metric_identity = all(current_manifest.get(item["path"]) == item["sha256"] for item in goal.get("metric_sources", []))
    artifact_identity = all(current_manifest.get(item["path"]) == item["sha256"] for item in goal.get("artifact_bindings", []))
    directories = {item["artifact_directory"] for item in goal.get("artifact_bindings", []) if item.get("artifact_directory")}
    for directory in directories:
        prefix = directory.rstrip("/") + "/"
        expected_files = {path: digest for path, digest in instance["source_manifest"].items() if path.startswith(prefix)}
        current_files = {path: digest for path, digest in current_manifest.items() if path.startswith(prefix)}
        artifact_identity = artifact_identity and expected_files == current_files and not _directory_has_symbolic_links(root, directory)
    return {
        "原始指标名称与实现来源保持一致": metrics_match,
        "原始算法目标与配置来源保持一致": algorithm_match,
        "冻结的原始数据来源保持一致": dataset_match,
        "原始测试环境参数保持一致": testenv_match,
        "原始测试环境配置来源保持一致": testenv_source_match,
        "原始排序指标与方向保持一致": ranking_match,
        "原始评测指标实现保持一致": metric_identity,
        "创建时已有模型与数据文件保持一致": artifact_identity,
    }
