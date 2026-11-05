# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shutil
import subprocess
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .packages import package_identity
from .processes import runtime_environment

IGNORED_DIRECTORIES = {".git", "workspace", "__pycache__", ".venv", "venv", "node_modules", ".agents", ".workbuddy", ".pytest_cache", ".idea"}
IGNORED_FILES = {".git", "AGENTS.md", "SHARED_AGENTS.md"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_identity(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError(f"{label}标识无效")
    return value


def source_files(root):
    root = Path(root)
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(name for name in folders if name not in IGNORED_DIRECTORIES and not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if name in IGNORED_FILES or name.startswith(".env") or name.endswith((".pyc", ".pyo")) or path.is_symlink():
                continue
            yield path


def _file_signature(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_ctime_ns, info.st_mtime_ns, info.st_size


def file_manifest(root, digest_cache=None):
    root = Path(root).resolve()
    # Windows 与 /mnt 挂载目录的时间信息不用于复用摘要。
    if os.name == "nt" or root.is_relative_to("/mnt"):
        digest_cache = None
    manifest = {}
    for path in source_files(root):
        cache_key = str(path)
        signature = _file_signature(path) if digest_cache is not None else None
        cached = digest_cache.get(cache_key) if digest_cache is not None else None
        settled = signature is not None and time.time_ns() - signature[2] > 1_000_000_000
        # 近期修改文件需要在一秒后重新读取，随后才开始复用摘要。
        if cached is not None and cached[0] == signature and cached[2] and settled:
            digest = cached[1]
        else:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest_cache is not None:
                if signature == _file_signature(path):
                    digest_cache[cache_key] = (signature, digest, settled)
                else:
                    digest_cache.pop(cache_key, None)
        manifest[path.relative_to(root).as_posix()] = digest
    return manifest


def manifest_changes(before, after):
    return sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))


def configuration_evidence(root, benchmark_path):
    root = Path(root).resolve()
    evidence = {"status": "incomplete", "benchmark_text": None, "testenv_text": None, "benchmark": None, "testenv": None, "issues": []}
    path = root / benchmark_path
    if not path.is_file():
        evidence["issues"].append(f"benchmark 文件缺失：{benchmark_path}")
        return evidence
    evidence["benchmark_text"] = path.read_text(encoding="utf-8")
    try:
        configuration = yaml.safe_load(evidence["benchmark_text"])
    except yaml.YAMLError as error:
        evidence["issues"].append(f"benchmark YAML 无法解析：{error}")
        return evidence
    evidence["benchmark"] = configuration
    job = configuration.get("benchmarkingjob") if isinstance(configuration, dict) else None
    if not isinstance(job, dict) or not isinstance(job.get("testenv"), str):
        evidence["issues"].append("benchmarkingjob/testenv 配置缺失或类型无效")
        return evidence
    testenv = (root / job["testenv"]).resolve()
    if not testenv.is_relative_to(root):
        evidence["issues"].append(f"testenv 配置不属于实例源码目录：{job['testenv']}")
        return evidence
    if not testenv.is_file():
        evidence["issues"].append(f"testenv 文件缺失：{job['testenv']}")
        return evidence
    evidence["testenv_text"] = testenv.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(evidence["testenv_text"])
    except yaml.YAMLError as error:
        evidence["issues"].append(f"testenv YAML 无法解析：{error}")
        return evidence
    evidence["testenv"] = parsed
    if not isinstance(parsed, dict) or not isinstance(parsed.get("testenv"), dict):
        evidence["issues"].append("testenv 配置结构无效")
        return evidence
    evidence["status"] = "complete"
    return evidence


class ExampleCatalog:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()

    def list_examples(self):
        inventory_path = self.project_root / ".github/workflows/validator/data/example_inventory.yaml"
        inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else {"examples": []}
        entries = {str(item.get("benchmark_file", "")).removeprefix("./"): item for item in inventory.get("examples", []) if item.get("benchmark_file")}
        benchmarks = set(entries)
        for path in (self.project_root / "examples").rglob("*.yaml"):
            if path.name.startswith("benchmarkingjob"):
                benchmarks.add(path.relative_to(self.project_root).as_posix())
        examples = []
        for benchmark_path in sorted(benchmarks):
            path = self.project_root / benchmark_path
            if not path.is_file():
                continue
            entry = entries.get(benchmark_path)
            readme = entry.get("readme_file") if entry else None
            if not readme:
                parents = [path.parent, *path.parent.parents]
                for parent in parents:
                    if parent == self.project_root:
                        break
                    matches = [item for item in parent.iterdir() if item.is_file() and item.name.lower() == "readme.md"]
                    if matches:
                        readme = matches[0].relative_to(self.project_root).as_posix()
                        break
            evidence = configuration_evidence(self.project_root, benchmark_path)
            examples.append({"benchmark_path": benchmark_path, "name": entry.get("name", path.parent.name) if entry else path.parent.name, "readme_path": readme, "inventory_entry": entry, "config_status": "valid" if evidence["status"] == "complete" else "invalid", "config_issues": evidence["issues"]})
        return examples


class InstanceManager:
    def __init__(self, settings, store, events):
        self.settings, self.store, self.events = settings, store, events
        self._digest_cache = {}

    def create(self, benchmark_path, name=None):
        benchmark_path = str(benchmark_path).removeprefix("./")
        matches = [item for item in ExampleCatalog(self.settings.project_root).list_examples() if item["benchmark_path"] == benchmark_path]
        if len(matches) != 1:
            raise ValueError("benchmark 路径没有对应的唯一示例")
        example = matches[0]
        instance_id = self.store.allocate_id("instance")
        relative = f"instances/{instance_id}"
        code_root = self.settings.state_root / relative / "code"
        code_root.mkdir(parents=True, exist_ok=False)
        original = Path(self.settings.project_root).resolve()
        instance = {"instance_id": instance_id, "name": name or instance_id, "code_root": str(code_root), **example, "environment": None, "status": "creating", "verification": None, "created_at": utc_now(), "source_root": str(original), "source_manifest": {}, "current_manifest": {}, "needs_verification": True, "source_updates": [], "task_spec": {"status": "pending", "benchmark_text": None, "testenv_text": None}}
        self.store.write(f"{relative}/instance.json", instance)
        try:
            for source in source_files(original):
                destination = code_root / source.relative_to(original)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            manifest = file_manifest(code_root, self._digest_cache)
            instance.update(source_manifest=manifest, current_manifest=manifest, task_spec=configuration_evidence(code_root, benchmark_path), status="created")
            if instance["task_spec"]["status"] != "complete":
                instance["status"] = "configuration_invalid"
            from .original_goal import OriginalGoalStore
            OriginalGoalStore(self.settings, self.store).capture(instance)
            self.store.write(f"{relative}/instance.json", instance)
        except Exception as error:
            instance.update(status="creation_failed", creation_error=str(error), creation_error_type=type(error).__name__)
            self.store.write(f"{relative}/instance.json", instance)
            raise
        self.events.emit("instance.created", {"name": instance["name"], "benchmark_path": benchmark_path}, instance_id=instance_id)
        return instance

    def list_instances(self):
        root = self.settings.state_root / "instances"
        return [self.store.read(path.relative_to(self.settings.state_root).as_posix()) for path in sorted(root.glob("*/instance.json"))] if root.exists() else []

    def get(self, instance_id):
        validate_identity(instance_id, "实例")
        instance = self.store.read(f"instances/{instance_id}/instance.json")
        if instance is None:
            raise KeyError("实例不存在")
        if instance.get("instance_id") != instance_id:
            raise ValueError("实例记录身份与请求不一致")
        return instance

    def preview_deletion(self, instance_id):
        from .instance_deletion import InstanceDeletionManager
        return InstanceDeletionManager(self).preview(instance_id)

    def delete(self, instance_id):
        from .instance_deletion import InstanceDeletionManager
        return InstanceDeletionManager(self).delete(instance_id)

    def recover_pending_deletions(self):
        from .instance_deletion import InstanceDeletionManager
        return InstanceDeletionManager(self).recover()

    def check_changes(self, instance_id, check_source=True):
        instance = self.get(instance_id)
        current = file_manifest(instance["code_root"], self._digest_cache)
        changed = manifest_changes(instance.get("current_manifest", {}), current)
        if check_source:
            upstream = file_manifest(instance["source_root"], self._digest_cache)
            instance["source_updates"] = manifest_changes(instance["source_manifest"], upstream)
            instance["source_updates_checked_at"] = utc_now()
        environment_changed = False
        environment = instance.get("environment") or {}
        if environment.get("status") == "ready":
            package_check = subprocess.run([environment["python_executable"], "-m", "pip", "--disable-pip-version-check", "list", "--format=json"], cwd=instance["code_root"], env=runtime_environment(environment["python_executable"]), capture_output=True, text=True, check=True, timeout=30)
            environment_changed = package_identity(json.loads(package_check.stdout)) != package_identity(environment["packages"])
        if changed or environment_changed:
            instance["needs_verification"] = True
            instance["status"] = "needs_verification"
            if instance.get("verification"):
                instance["verification"]["stale"] = True
            self.events.emit("instance.changed", {"changed_files": changed, "environment_changed": environment_changed}, instance_id=instance_id)
        instance["current_manifest"] = current
        instance["changed_files"] = changed
        instance["environment_changed"] = environment_changed
        self.store.write(f"instances/{instance_id}/instance.json", instance)
        return instance
