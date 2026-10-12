# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from pathlib import Path

from filelock import FileLock

from .instances import utc_now, validate_identity
from .processes import recorded_process
from .runner import Runner


class InstanceDeletionManager:
    def __init__(self, manager):
        self.manager = manager
        self.store = manager.store

    def _path(self, reference):
        path = self.store.root / reference
        if path.resolve() != path or not path.is_relative_to(self.store.root):
            raise PermissionError("删除目录必须位于程序管理的确切路径，不能包含符号链接或目录联接")
        return path

    @staticmethod
    def _environment_ids(instance):
        values = {instance.get("pending_environment_id")}
        for field in ("environment", "last_environment_attempt"):
            value = instance.get(field) or {}
            values.add(value.get("environment_id"))
        return {validate_identity(value, "环境") for value in values if value is not None}

    def _owned_environments(self, instance, root):
        identity = instance["instance_id"]
        identities = self._environment_ids(instance)
        for path in root.glob("environment-plans/*.json"):
            plan = self.store.read(path.relative_to(self.store.root).as_posix())
            if plan.get("instance_id") != identity or plan.get("plan_id") != path.stem:
                raise ValueError("环境计划身份与实例不一致")
            for field in ("candidate_environment_id", "base_environment_id", "reused_environment_id"):
                if plan.get(field):
                    identities.add(validate_identity(plan[field], "环境"))
        for path in (self.store.root / "environments").glob("*/environment.json"):
            value = self.store.read(path.relative_to(self.store.root).as_posix())
            if value.get("instance_id") == identity:
                if value.get("environment_id") != path.parent.name:
                    raise ValueError("环境记录身份与目录不一致")
                identities.add(validate_identity(path.parent.name, "环境"))
        for environment_id in identities:
            folder = self._path("environments/" + environment_id)
            record = self.store.read(f"environments/{environment_id}/environment.json")
            if record is not None and (record.get("instance_id") != identity or record.get("environment_id") != environment_id):
                raise ValueError("环境不属于待删除实例")
            if folder.exists() and not folder.is_dir():
                raise ValueError("受管理环境路径不是目录")
        for path in (self.store.root / "instances").glob("*/instance.json"):
            if path.parent.name == identity:
                continue
            other = self.store.read(path.relative_to(self.store.root).as_posix())
            if self._environment_ids(other) & identities:
                raise ValueError("待删除环境同时被其他实例引用")
        return sorted(identities)

    def _assert_inactive(self, instance_id, instance_root, environment_ids):
        for path in instance_root.glob("runs/*/run.json"):
            run = self.store.read(path.relative_to(self.store.root).as_posix())
            if Runner._owned_process(run) is not None:
                raise RuntimeError("当前实例仍有运行进程，请先取消或等待完成")
        patterns = (
            "environment-commands/*/command.json",
            "environment-checks/*/operation.json",
            "environment-plans/*/parse/operation.json",
            "environment-plans/*/resolve-operation.json",
            "changes/*/*-operation.json",
        )
        for pattern in patterns:
            for path in instance_root.glob(pattern):
                operation = self.store.read(path.relative_to(self.store.root).as_posix())
                if recorded_process(operation) is not None:
                    raise RuntimeError("当前实例仍有环境或补丁操作进程，请先取消或等待完成")
        for environment_id in environment_ids:
            environment = self.store.read(f"environments/{environment_id}/environment.json", {})
            if not environment:
                staged = f"instance-deletions/{instance_id}/data/environments/{environment_id}/environment.json"
                environment = self.store.read(staged, {})
            for operation in environment.get("steps", []):
                if recorded_process(operation) is not None:
                    raise RuntimeError("当前实例仍有环境准备进程，请先取消或等待完成")

    def preview(self, instance_id):
        validate_identity(instance_id, "实例")
        reference = f"instance-deletions/{instance_id}/deletion.json"
        previous = self.store.read(reference)
        if previous is not None:
            self._validate_record(instance_id, previous)
            if previous.get("status") != "deleted":
                self._assert_ownership(instance_id, previous)
                original = self._path("instances/" + instance_id)
                staged = self._path(f"instance-deletions/{instance_id}/data/instance")
                self._assert_inactive(instance_id, original if original.exists() else staged, previous["environment_ids"])
                return {**self._result(previous, reference), "paths": [str(self._path(item["source_ref"])) for item in previous["targets"]], "retry": True}
        root = self._path("instances/" + instance_id)
        instance = self.manager.get(instance_id)
        if not root.is_dir():
            raise ValueError("实例受管理目录缺失")
        if Path(instance["code_root"]) != root / "code":
            raise ValueError("实例源码目录与受管理目录不一致")
        environments = self._owned_environments(instance, root)
        self._assert_inactive(instance_id, root, environments)
        return {
            "instance_id": instance_id,
            "name": instance.get("name", instance_id),
            "environment_ids": environments,
            "paths": [str(root), *[str(self._path("environments/" + value)) for value in environments if self._path("environments/" + value).exists()]],
            "deletion_ref": f"instance-deletions/{instance_id}/deletion.json",
        }

    @staticmethod
    def _targets(instance_id, environment_ids):
        prefix = "instance-deletions/" + instance_id + "/data/"
        return [{"source_ref": "instances/" + instance_id, "staged_ref": prefix + "instance", "status": "pending"}, *[
            {"source_ref": "environments/" + value, "staged_ref": prefix + "environments/" + value, "status": "pending"}
            for value in environment_ids
        ]]

    def _validate_record(self, instance_id, record):
        identities = record.get("environment_ids")
        if record.get("instance_id") != instance_id or not isinstance(identities, list) or record.get("status") not in {"prepared", "deleting", "failed", "deleted"}:
            raise ValueError("删除操作身份无效")
        identities = [validate_identity(value, "环境") for value in identities]
        if identities != sorted(set(identities)):
            raise ValueError("删除操作环境清单无效")
        expected = self._targets(instance_id, identities)
        actual = record.get("targets")
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("删除操作目录清单无效")
        for item, target in zip(actual, expected):
            if any(item.get(field) != target[field] for field in ("source_ref", "staged_ref")) or item.get("status") not in {"pending", "staged", "deleted"}:
                raise ValueError("删除操作目录超出确切范围")
            self._path(item["source_ref"])
            self._path(item["staged_ref"])

    def _assert_ownership(self, instance_id, record):
        for target in record["targets"]:
            for field in ("source_ref", "staged_ref"):
                folder = self._path(target[field])
                if not folder.exists():
                    continue
                if not folder.is_dir():
                    raise FileExistsError("删除范围中的路径不是目录：" + target[field])
                if target["source_ref"].startswith("environments/"):
                    value = self.store.read(target[field] + "/environment.json")
                    if value is not None and (value.get("instance_id") != instance_id or value.get("environment_id") != Path(target["source_ref"]).name):
                        raise ValueError("删除环境的正式记录归属发生变化")
                else:
                    value = self.store.read(target[field] + "/instance.json")
                    partial_cleanup = field == "staged_ref" and target["status"] in {"staged", "deleted"} and not self._path(target["source_ref"]).exists()
                    if (value is None and not partial_cleanup) or (value is not None and value.get("instance_id") != instance_id):
                        raise ValueError("删除实例的正式记录身份发生变化")

    def _clear_pointer(self, instance_id):
        path = self._path("last_session.json")
        with FileLock(str(path) + ".lock"):
            value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if value.get("instance_id") == instance_id:
                self.store._write_atomic(path, {})

    def _resume(self, instance_id, record):
        self._validate_record(instance_id, record)
        reference = f"instance-deletions/{instance_id}/deletion.json"
        if record.get("status") == "deleted":
            return self._result(record, reference)
        record.update(status="deleting", updated_at=utc_now())
        record.pop("error", None)
        self.store.write(reference, record)
        try:
            self._assert_ownership(instance_id, record)
            original = self._path("instances/" + instance_id)
            staged = self._path(f"instance-deletions/{instance_id}/data/instance")
            checked_root = original if original.exists() else staged
            self._assert_inactive(instance_id, checked_root, record["environment_ids"])
            self._clear_pointer(instance_id)
            # 先移动全部受管理目录，再清理内容；中断后根据正式范围继续。
            for target in record["targets"]:
                source = self._path(target["source_ref"])
                destination = self._path(target["staged_ref"])
                if source.exists() and destination.exists():
                    raise FileExistsError("删除源目录与暂存目录同时存在，需要核查：" + target["source_ref"])
                if source.exists():
                    if target["status"] != "pending":
                        raise ValueError("已删除目录被重新创建，不能继续旧操作")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, destination)
                target["status"] = "staged" if destination.exists() else "deleted"
                self.store.write(reference, record)
            for target in record["targets"]:
                destination = self._path(target["staged_ref"])
                if destination.exists():
                    shutil.rmtree(destination)
                target["status"] = "deleted"
                self.store.write(reference, record)
            record.update(status="deleted", completed_at=utc_now())
            self.store.write(reference, record)
            return self._result(record, reference)
        except Exception as error:
            record.update(status="failed", error=str(error), error_type=type(error).__name__, updated_at=utc_now())
            self.store.write(reference, record)
            raise

    @staticmethod
    def _result(record, reference):
        return {key: record[key] for key in ("instance_id", "name", "status", "environment_ids")} | {"deletion_ref": reference}

    def delete(self, instance_id):
        validate_identity(instance_id, "实例")
        directory = self._path("instance-deletions/" + instance_id)
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(directory / "operation.lock"), timeout=0):
            reference = f"instance-deletions/{instance_id}/deletion.json"
            record = self.store.read(reference)
            if record is None:
                preview = self.preview(instance_id)
                record = {"schema_version": 1, **{key: preview[key] for key in ("instance_id", "name", "environment_ids")}, "status": "prepared", "created_at": utc_now(), "targets": self._targets(instance_id, preview["environment_ids"])}
                self.store.write(reference, record)
            return self._resume(instance_id, record)

    def recover(self):
        results = []
        for path in sorted((self.store.root / "instance-deletions").glob("*/deletion.json")):
            reference = path.relative_to(self.store.root).as_posix()
            try:
                record = self.store.read(reference)
                if record.get("status") != "deleted":
                    results.append(self.delete(path.parent.name))
            except Exception as error:
                results.append({"instance_id": path.parent.name, "status": "failed", "deletion_ref": reference, "error": str(error)})
        return results
