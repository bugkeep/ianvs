# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

from .instances import file_manifest, source_files, utc_now
from .processes import recorded_process, sandboxed_command, terminate_recorded_operation


class EnvironmentCommands:
    def __init__(self, settings, store, events):
        self.settings, self.store, self.events = settings, store, events
        self.active = {}
        self.recover()

    def recover(self):
        self.active = {}
        for path in self.settings.state_root.glob("instances/*/environment-commands/*/command.json"):
            relative = path.relative_to(self.store.root).as_posix()
            record = self.store.read(relative)
            if record.get("status") not in {"staging", "starting", "running", "recovery_required"}:
                continue
            process = recorded_process(record)
            record.update(status="recovery_required" if process else "interrupted", recovery_checked_at=utc_now())
            self.store.write(relative, record)
            if process:
                self.active[record["instance_id"]] = record

    def preview(self, instance, argv, cwd=".", timeout=300, writable_directories=None, network=False):
        if instance["instance_id"] in self.active:
            raise RuntimeError("当前实例有待核查的活动环境命令，不能再次执行")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and "\x00" not in item for item in argv):
            raise ValueError("环境命令必须使用非空参数列表")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 1800:
            raise ValueError("环境命令时限必须大于零且不超过 1800 秒")
        environment = instance.get("environment") or {}
        python = environment.get("python_executable")
        if environment.get("status") not in {"ready", "created", "modified"} or not python or not Path(python).is_file():
            raise RuntimeError("请先为当前实例创建 Python 环境")
        root = Path(instance["code_root"]).resolve()
        working = (root / cwd).resolve(strict=True)
        if not working.is_relative_to(root) or not working.is_dir():
            raise PermissionError("环境命令工作目录必须属于当前实例源码")
        command = list(argv)
        if command[0] in {"python", "python3"}:
            command[0] = python
        elif not Path(command[0]).is_absolute():
            command[0] = shutil.which(command[0])
        if not command[0] or not Path(command[0]).is_file():
            raise FileNotFoundError("指定环境命令不存在")
        manifest = file_manifest(root)
        generated = set(instance.get("generated_directories") or [])
        directories = []
        for name in writable_directories or []:
            destination = (root / name).resolve()
            if destination == root or not destination.is_relative_to(root):
                raise PermissionError("生成目录必须位于当前实例内部")
            relative = destination.relative_to(root).as_posix()
            if any(part.startswith(".env") or part in {".git", ".github", "core", "ianvs_agent", "workspace"} for part in Path(relative).parts):
                raise PermissionError("生成目录不能覆盖受保护的代码、记录或凭据")
            if relative not in generated and any(path == relative or path.startswith(relative + "/") for path in manifest):
                raise PermissionError("已有源码及输入文件只能通过对应的版本绑定工具修改")
            directories.append(relative)
        revision = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        return {"instance_id": instance["instance_id"], "environment_id": environment["environment_id"], "generation": environment.get("generation"), "python_executable": python, "dependency_fingerprint": environment.get("dependency_fingerprint"), "command": command, "cwd": str(working), "timeout_seconds": timeout, "writable_directories": sorted(set(directories)), "network": bool(network), "source_revision": revision, "source_file_count": len(manifest)}

    async def cancel(self):
        self.recover()
        for record in list(self.active.values()):
            await terminate_recorded_operation(record)
            record.update(status="cancelled", completed_at=utc_now(), interrupted=True)
            self.store.write(record["evidence_ref"], record)
        self.active = {}

    def _snapshot(self, root, snapshot, directories):
        for name in directories:
            (root / name).mkdir(parents=True, exist_ok=True)
        for path in source_files(root):
            target = snapshot / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for name in directories:
            (snapshot / name).mkdir(parents=True, exist_ok=True)
        return file_manifest(snapshot)

    async def execute(self, instance, argv, cwd=".", timeout=300, writable_directories=None, network=False, approved=None):
        plan = await asyncio.to_thread(self.preview, instance, argv, cwd, timeout, writable_directories, network)
        if approved is not None and approved != plan:
            raise PermissionError("审批后的命令、环境或源码版本发生变化")
        operation_id = self.store.allocate_id("environment-command")
        relative = f"instances/{instance['instance_id']}/environment-commands/{operation_id}"
        folder = self.store.path(relative)
        folder.mkdir(parents=True, exist_ok=False)
        snapshot = folder / "source"
        snapshot.mkdir()
        root = Path(instance["code_root"]).resolve()
        record = {"schema_version": 1, **plan, "operation_id": operation_id, "task_id": self.events.identities.get("task_id"), "status": "staging", "started_at": utc_now(), "evidence_ref": relative + "/command.json", "readonly_mappings": [{"source": str(snapshot), "destination": str(root)}]}

        def save(value):
            self.store.write(relative + "/command.json", value)
            self.events.emit("environment.command_progress", {key: value.get(key) for key in ("operation_id", "status", "pid", "log_ref")}, instance_id=instance["instance_id"])

        self.active[instance["instance_id"]] = record
        save(record)
        self.events.emit("environment.command_started", {"operation_id": operation_id, "command": plan["command"], "cwd": plan["cwd"]}, instance_id=instance["instance_id"])
        try:
            staging = asyncio.create_task(asyncio.to_thread(self._snapshot, root, snapshot, plan["writable_directories"]))
            try:
                copied = await asyncio.shield(staging)
            except asyncio.CancelledError:
                await staging
                raise
            if hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest() != plan["source_revision"]:
                raise PermissionError("命令执行前的源码版本发生变化，需要重新审批")
            await sandboxed_command(plan["command"], plan["cwd"], plan["python_executable"], folder / "output.log", timeout=timeout, readonly_paths=[root], writable_paths=[root / name for name in plan["writable_directories"]], network=network, operation=record, on_update=save, pip_timeout_seconds=self.settings.pip_timeout_seconds, pip_retries=self.settings.pip_retries)
        except (Exception, asyncio.CancelledError) as error:
            record.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=str(error) or type(error).__name__)
            raise
        finally:
            record["completed_at"] = utc_now()
            save(record)
            self.active.pop(instance["instance_id"], None)
            current = self.store.read(f"instances/{instance['instance_id']}/instance.json")
            current["last_environment_command"] = {key: record.get(key) for key in ("operation_id", "status", "evidence_ref", "log_ref", "completed_at")}
            if plan["writable_directories"]:
                current["generated_directories"] = sorted(set(current.get("generated_directories") or []) | set(plan["writable_directories"]))
                current.update(needs_verification=True, status="needs_verification")
                if current.get("verification"):
                    current["verification"]["stale"] = True
            self.store.write(f"instances/{instance['instance_id']}/instance.json", current)
            self.events.emit("environment.command_completed" if record["status"] == "completed" else "environment.command_failed", {"operation_id": operation_id, "status": record["status"], "evidence_ref": record["evidence_ref"], "log_ref": record.get("log_ref"), "error": record.get("error")}, instance_id=instance["instance_id"])
        public = {key: value for key, value in record.items() if key != "readonly_mappings"}
        public["stdout"] = (folder / "output.log").read_text(encoding="utf-8", errors="replace")[-12000:]
        return public
