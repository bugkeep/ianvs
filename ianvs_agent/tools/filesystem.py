# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

import patch_ng
import yaml
from charset_normalizer.utils import identify_sig_or_bom

from ianvs_agent.runtime.instances import configuration_evidence
from ianvs_agent.runtime.environment import environment_summary
from ianvs_agent.runtime.credentials import credential_status
from ianvs_agent.runtime.processes import sandboxed_command
from .operations import register_operations


SEARCH_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md", ".txt", ".json", ".jsonl", ".toml", ".sh"})
SEARCH_MAX_BYTES = 2_000_000
SEARCH_MAX_LIMIT = 1000
MANAGED_PATH_PARTS = frozenset({".git", ".env", "workspace", "environments", "sessions", "changes"})


class CodeTools:
    def __init__(self, harness):
        self.harness = harness

    @property
    def instance(self):
        if not self.harness.current_instance:
            raise ValueError("请先选择实例")
        return self.harness.current_instance

    @property
    def code_root(self):
        return Path(self.instance["code_root"]).resolve()

    @staticmethod
    def _path_restriction(relative):
        if any(part in MANAGED_PATH_PARTS for part in relative.parts):
            return "此目录由管理程序维护，工具无权访问"
        if any(part.startswith(".env") for part in relative.parts):
            return "工具无权读取凭据文件或目录"
        return None

    def code_path(self, path):
        requested = self.code_root / path
        target = requested.resolve()
        if not target.is_relative_to(self.code_root):
            raise PermissionError("文件路径超出当前实例源码目录")
        relatives = [target.relative_to(self.code_root)]
        if requested.is_relative_to(self.code_root):
            relatives.append(requested.relative_to(self.code_root))
        for relative in relatives:
            restriction = self._path_restriction(relative)
            if restriction:
                raise PermissionError(restriction)
        return target

    @staticmethod
    def _source_text(target):
        content = target.read_bytes()
        detected, _ = identify_sig_or_bom(content)
        encodings = {None: "utf-8", "utf_8": "utf-8-sig", "utf_16": "utf-16", "utf_32": "utf-32"}
        encoding = encodings.get(detected)
        if encoding is None:
            return content, None, detected, "unsupported_encoding"
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            return content, None, encoding, "invalid_encoding"
        if "\x00" in text:
            return content, None, encoding, "binary"
        return content, text, encoding, None

    async def read_file(self, path, start_line=1, end_line=200):
        return await asyncio.to_thread(self._read_file, path, start_line, end_line)

    def _read_file(self, path, start_line, end_line):
        if start_line < 1 or end_line < start_line or end_line - start_line > 2000:
            raise ValueError("读取范围无效或超过 2000 行")
        target = self.code_path(path)
        if not target.exists():
            return {"path": target.relative_to(self.code_root).as_posix(), "source": str(target), "exists": False, "file_version": "absent", "start_line": start_line, "end_line": 0, "total_lines": 0, "content": "", "truncated": False, "ends_with_newline": False}
        relative = target.relative_to(self.code_root).as_posix()
        if not stat.S_ISREG(target.stat().st_mode):
            raise ValueError(f"读取路径必须为普通文件：{relative}")
        content, text, encoding, reason = self._source_text(target)
        if reason:
            raise ValueError(f"无法作为文本读取 {relative}：{reason}；支持 UTF-8 及带 BOM 的 UTF-16/UTF-32")
        lines = text.splitlines()
        return {"path": relative, "source": str(target), "encoding": encoding, "start_line": start_line, "end_line": min(end_line, len(lines)), "total_lines": len(lines), "content": "\n".join(f"{index + 1}: {lines[index]}" for index in range(start_line - 1, min(end_line, len(lines)))), "file_version": hashlib.sha256(content).hexdigest(), "ends_with_newline": text.endswith(("\n", "\r")), "truncated": end_line < len(lines), "continuation": {"path": relative, "start_line": end_line + 1}}

    async def search_text(self, pattern, paths=None, limit=50, cursor=0):
        return await asyncio.to_thread(self._search_text, pattern, paths, limit, cursor)

    def _search_text(self, pattern, paths, limit, cursor):
        if limit < 1 or limit > SEARCH_MAX_LIMIT or cursor < 0:
            raise ValueError(f"limit 必须为 1..{SEARCH_MAX_LIMIT}，cursor 必须大于或等于 0")
        expression = re.compile(pattern)
        candidates, roots, skipped = set(), [], []

        def walk_error(error):
            raise error

        def allowed_entry(entry):
            relative = entry.relative_to(self.code_root).as_posix()
            if entry.is_symlink():
                skipped.append({"path": relative, "reason": "symbolic_link"})
                return False
            restriction = self._path_restriction(Path(relative))
            if restriction:
                skipped.append({"path": relative, "reason": "restricted_path", "detail": restriction})
                return False
            self.code_path(relative)
            return True

        for name in paths or ["examples", "core"]:
            target = self.code_path(name)
            roots.append(target.relative_to(self.code_root).as_posix())
            metadata = target.stat()
            requested = self.code_root / name
            if requested.is_symlink():
                skipped.append({"path": requested.relative_to(self.code_root).as_posix(), "reason": "symbolic_link"})
            elif stat.S_ISDIR(metadata.st_mode):
                for directory, directories, files in os.walk(target, onerror=walk_error, followlinks=False):
                    directories[:] = sorted(entry for entry in directories if allowed_entry(Path(directory) / entry))
                    for filename in sorted(files):
                        candidate = Path(directory) / filename
                        if allowed_entry(candidate):
                            candidates.add(candidate)
            elif stat.S_ISREG(metadata.st_mode):
                candidates.add(target)
            else:
                raise ValueError(f"搜索路径必须为常规文件或目录：{name}")
        hits = []
        checked = []
        for candidate in sorted(candidates):
            if candidate.suffix.lower() not in SEARCH_SUFFIXES:
                continue
            relative = candidate.relative_to(self.code_root).as_posix()
            candidate = self.code_path(relative)
            metadata = candidate.stat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"搜索文件类型发生变化：{relative}")
            if metadata.st_size > SEARCH_MAX_BYTES:
                skipped.append({"path": relative, "reason": "file_too_large", "size_bytes": metadata.st_size})
                continue
            _, text, encoding, reason = self._source_text(candidate)
            if reason:
                skipped.append({"path": relative, "reason": reason, "encoding": encoding})
                continue
            checked.append(relative)
            for line_number, line in enumerate(text.splitlines(), 1):
                if expression.search(line):
                    hits.append({"path": relative, "source": str(candidate), "line": line_number, "text": line, "encoding": encoding})
        return {
            "pattern": pattern,
            "cwd": str(self.code_root),
            "scope": {"paths": sorted(set(roots)), "suffixes": sorted(SEARCH_SUFFIXES), "max_file_bytes": SEARCH_MAX_BYTES, "encodings": ["UTF-8", "UTF-16 with BOM", "UTF-32 with BOM"], "binary_policy": "skip decoded NUL content; invalid or unsupported encodings are reported separately", "follow_symlinks": False},
            "limit": limit,
            "cursor": cursor,
            "total_matches": len(hits),
            "next_cursor": cursor + limit if len(hits) > cursor + limit else None,
            "skipped_count": len(skipped),
            "skipped_paths": sorted(skipped, key=lambda item: item["path"]),
            "matches": hits[cursor:cursor + limit],
            "checked_paths": checked,
        }

    def _registered_validation_report(self, relative):
        prefix = f"instances/{self.instance['instance_id']}/validations/"
        if not relative.startswith(prefix) or Path(relative).name != "report.json" or len(Path(relative).parts) != 5:
            return False
        instance = self.harness.instances.get(self.instance["instance_id"])
        references = {(instance.get("verification") or {}).get("static_report_ref")}
        for record in self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl"):
            if record.get("status") == "success" and record.get("name") in {"validate_instance", "verify_run"}:
                data = record.get("data") or {}
                references.update((data.get("report_ref"), data.get("static_report_ref")))
        root = self.harness.store.path(f"instances/{self.instance['instance_id']}/runs")
        for path in root.glob("*/verification.json"):
            verification = self.harness.store.read(path.relative_to(self.harness.store.root).as_posix())
            references.add(verification.get("static_report_ref"))
        return relative in references

    async def read_evidence(self, reference, start_line=1, end_line=200):
        return await asyncio.to_thread(self._read_evidence, reference, start_line, end_line)

    def _read_evidence(self, reference, start_line, end_line):
        target = self.harness.store.path(reference)
        relative = str(target.relative_to(self.harness.store.root)).replace("\\", "/")
        base = f"instances/{self.instance['instance_id']}/"
        permitted = (base + "runs/", self.harness.session_relative_path + "/tool-output/")
        environment_log = False
        if relative.startswith("environments/") and len(Path(relative).parts) >= 3:
            environment_id = Path(relative).parts[1]
            environment = self.harness.store.read(f"environments/{environment_id}/environment.json", {})
            environment_log = environment.get("instance_id") == self.instance["instance_id"] and (target.suffix == ".log" or relative == environment.get("pip_report_ref"))
        operation_evidence = False
        for kind, filename in (("environment-plans", None), ("environment-checks", "check.json"), ("environment-commands", "command.json")):
            if not relative.startswith(base + kind + "/"):
                continue
            parts = Path(relative).parts
            registered = base + kind + "/" + parts[3].removesuffix(".json") + ".json" if filename is None else base + kind + "/" + parts[3] + "/" + filename
            record = self.harness.store.read(registered, {})
            references = {record.get(key) for key in ("evidence_ref", "log_ref", "operation_ref", "parse_operation_ref", "resolve_operation_ref", "resolution_ref", "constraints_ref")}
            for name in ("parse_operation", "resolve_operation"):
                references.add((record.get(name) or {}).get("log_ref"))
            for name in ("parse_operation_ref", "resolve_operation_ref"):
                if record.get(name):
                    references.add(self.harness.store.read(record[name], {}).get("log_ref"))
            operation_evidence = record.get("instance_id") == self.instance["instance_id"] and (relative in references or str(target) in references)
        if not relative.startswith(permitted) and not environment_log and not operation_evidence and not self._registered_validation_report(relative):
            raise PermissionError("只能读取当前实例已登记的运行、工具、验证报告或环境日志证据")
        if not target.is_file():
            raise PermissionError("证据引用必须指向已经保存的普通文件")
        if start_line < 1 or end_line < start_line or end_line - start_line > 2000:
            raise ValueError("证据读取范围无效")
        lines = target.read_text(encoding="utf-8").splitlines()
        return {"reference": relative, "start_line": start_line, "end_line": min(end_line, len(lines)), "total_lines": len(lines), "content": "\n".join(f"{index + 1}: {lines[index]}" for index in range(start_line - 1, min(end_line, len(lines)))), "truncated": end_line < len(lines), "next_line": end_line + 1}

    def _instance_summary(self, instance):
        output_id = self.harness.store.allocate_id("OUTPUT")
        reference = self.harness.session_relative_path + f"/tool-output/{output_id}.json"
        self.harness.store.write(reference, instance)
        fields = ("instance_id", "name", "status", "benchmark_path", "code_root", "readme_path", "last_run_id", "needs_verification", "environment_changed", "source_updates_checked_at")
        summary = {key: instance[key] for key in fields if key in instance}
        environment = instance.get("environment") or {}
        summary["environment"] = environment_summary(environment)
        summary["runtime_credentials"] = credential_status(self.harness.settings, instance)
        summary["allowed_runtime_credentials"] = sorted(self.harness.settings.runtime_credentials)
        verification = instance.get("verification") or {}
        summary["verification"] = {key: verification[key] for key in ("run_id", "passed", "repair_completed", "stale", "status", "execution_completed", "results_complete", "checks_passed", "reasons", "unsupported_acceptance_fields", "verified_at") if key in verification}
        checks = verification.get("checks") or []
        summary["verification"]["check_count"] = len(checks)
        summary["verification"]["unsatisfied_checks"] = [check for check in checks if check.get("status") not in {"PASS", "WARN"}][:20]
        if verification.get("run_id"):
            summary["verification"]["report_ref"] = f"instances/{instance['instance_id']}/runs/{verification['run_id']}/verification.json"
        for field in ("changed_files", "source_updates"):
            values = instance.get(field) or []
            summary[field] = values[:30]
            summary[field + "_count"] = len(values)
            summary[field + "_truncated"] = len(values) > 30
        summary["source_file_count"] = len(instance.get("current_manifest") or {})
        summary["full_output_ref"] = reference
        summary["continuation"] = "使用 read_evidence 按需读取完整实例、依赖、文件版本和验证证据"
        return summary

    async def patch_approval(self, patch):
        return await self.harness._run_sync(self._patch_approval, patch)

    def _patch_approval(self, patch):
        task_id, instance_id = self.harness.task.get("task_id"), self.instance["instance_id"]
        if not task_id or self.harness.task.get("instance_id") != instance_id:
            raise PermissionError("补丁需要绑定当前实例的正式任务和读取记录")
        operations = {event["payload"].get("operation_id") for event in self.harness.store.read_events(self.harness.session_relative_path + "/events.jsonl") if event.get("task_id") == task_id and event.get("instance_id") == instance_id and event.get("event_type") == "tool.completed"}
        reads = {record["data"]["path"]: record for record in self.harness.store.read_events(self.harness.session_relative_path + "/tools.jsonl") if record.get("operation_id") in operations and record.get("name") == "read_file" and record.get("status") == "success"}
        versions = {path: record["data"]["file_version"] for path, record in reads.items()}
        targets, _ = self._patch_targets(patch, versions)
        return {"task_id": task_id, "instance_id": instance_id, "expected_versions": {path: versions[path] for path, _, _ in targets}, "read_evidence_refs": {path: "tool:" + reads[path]["operation_id"] for path, _, _ in targets}}

    def _patch_targets(self, patch, expected_versions):
        parsed = patch_ng.fromstring(patch.encode("utf-8"))
        if not parsed:
            raise ValueError("无法解析 unified diff，请提交完整补丁")
        targets = []
        strip_values = set()
        for item in parsed.items:
            source = item.source.decode("utf-8")
            destination = item.target.decode("utf-8")
            names = []
            for name in (source, destination):
                if name == "/dev/null":
                    continue
                prefixed = name.startswith(("a/", "b/"))
                strip_values.add(1 if prefixed else 0)
                normalized = name[2:] if prefixed else name
                if Path(normalized).is_absolute() or "\\" in normalized:
                    raise PermissionError("补丁文件需要使用实例内的相对路径")
                self.code_path(normalized)
                names.append(normalized)
            if not names or len(set(names)) != 1:
                raise ValueError("补丁每项需要修改同一个文件；重命名请分别提交删除与新增")
            name = names[0]
            target = self.code_path(name)
            relative = target.relative_to(self.code_root).as_posix()
            if relative.startswith(("core/", "ianvs_agent/", ".github/workflows/validator/")):
                raise PermissionError("当前工具无权修改运行控制、事件、评分或验证程序")
            frozen = self.instance.get("task_spec", {}).get("testenv_text")
            metrics = yaml.safe_load(frozen).get("testenv", {}).get("metrics", []) if frozen else []
            if any(str(metric.get("url", "")).removeprefix("./") == relative for metric in metrics):
                raise PermissionError("当前工具无权修改原始评测指标实现")
            version = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else "absent"
            if expected_versions.get(relative) != version:
                raise PermissionError(f"{relative} 的修改依据缺失或文件已变化，请重新读取和审批")
            targets.append((relative, target, target.read_text(encoding="utf-8") if target.exists() else None))
        if len(strip_values) != 1 or len({item[0] for item in targets}) != len(targets):
            raise ValueError("补丁路径前缀需要一致，单个文件仅能包含一项完整修改")
        return targets, strip_values.pop()

    async def apply_patch(self, patch, expected_versions=None, _approved=None):
        approval = await self.patch_approval(patch)
        if _approved is not None and _approved != approval:
            raise PermissionError("审批后的任务、读取依据或文件版本发生变化，需要重新审批")
        if expected_versions is not None and expected_versions != approval["expected_versions"]:
            raise PermissionError("expected_versions 与本任务正式读取记录不同，请重新读取")
        expected_versions = approval["expected_versions"]
        targets, strip = self._patch_targets(patch, expected_versions)
        patch_binary = shutil.which("patch")
        if patch_binary is None:
            raise FileNotFoundError("应用 unified diff 需要执行后端安装 GNU patch")
        modification_id = self.harness.store.allocate_id("PATCH")
        changes_root = f"instances/{self.instance['instance_id']}/changes/{modification_id}"
        staging = self.harness.store.path(changes_root + "/staging")
        staging.mkdir(parents=True, exist_ok=False)
        crlf_paths = set()
        for relative, target, original in targets:
            if original is not None:
                prepared = staging / relative
                prepared.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, prepared)
                with prepared.open(encoding="utf-8", newline=None) as source:
                    text = source.read()
                    if source.newlines == "\r\n":
                        crlf_paths.add(relative)
                if relative in crlf_paths:
                    prepared.write_text(text, encoding="utf-8", newline="\n")
        input_diff = self.harness.store.path(changes_root + "/input.diff")
        input_diff.write_bytes(patch.encode("utf-8"))
        execution_diff = self.harness.store.path(changes_root + "/execution.diff")
        with input_diff.open(encoding="utf-8", newline=None) as source:
            execution_diff.write_text(source.read(), encoding="utf-8", newline="\n")
        command = [patch_binary, "--unified", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", f"-p{strip}", "--directory", str(staging), "--input", str(execution_diff)]
        operation_refs = {}
        for phase in ("dry-run", "apply"):
            reference = changes_root + f"/{phase}-operation.json"
            operation_refs[phase] = reference
            log_path = self.harness.store.path(changes_root + f"/{phase}.log")
            operation = {"operation_id": modification_id + "-" + phase, "instance_id": self.instance["instance_id"], "task_id": self.harness.task.get("task_id"), "input_ref": changes_root + "/input.diff", "execution_input_ref": changes_root + "/execution.diff", "started_at": datetime.now(timezone.utc).isoformat()}
            try:
                await sandboxed_command(command + (["--dry-run"] if phase == "dry-run" else []), staging, None, log_path, timeout=120, writable_paths=[staging], readonly_paths=[execution_diff], operation=operation, on_update=lambda _: self.harness.store.write(reference, operation))
            except RuntimeError as error:
                details = log_path.read_text(encoding="utf-8", errors="replace")[-1000:] if log_path.is_file() else ""
                hint = "；请参考 read_file.ends_with_newline，核对上下文及标准 \\ No newline at end of file 标记" if operation.get("command_executed") and operation.get("exit_code") else ""
                raise RuntimeError(f"{error}；{details}{hint}") from error
            finally:
                operation["completed_at"] = datetime.now(timezone.utc).isoformat()
                self.harness.store.write(reference, operation)
        changes = []
        for relative, target, original in targets:
            prepared = staging / relative
            if prepared.exists() and relative in crlf_paths:
                prepared.write_text(prepared.read_text(encoding="utf-8"), encoding="utf-8", newline="\r\n")
            replacement = prepared.read_text(encoding="utf-8") if prepared.exists() else None
            current_version = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else "absent"
            if current_version != expected_versions[relative]:
                raise PermissionError("应用补丁期间文件已变化，需要重新读取")
            changes.append({"path": relative, "before": original, "after": replacement, "diff": "".join(difflib.unified_diff((original or "").splitlines(keepends=True), (replacement or "").splitlines(keepends=True), fromfile="a/" + relative, tofile="b/" + relative))})
        self.harness.store.write(changes_root + "/change.json", {"schema_version": 1, "modification_id": modification_id, "task_id": self.harness.task.get("task_id"), "input_ref": changes_root + "/input.diff", "execution_input_ref": changes_root + "/execution.diff", "dry_run_ref": operation_refs["dry-run"], "apply_ref": operation_refs["apply"], "changes": changes, "verified_run_id": None})
        for relative, target, original in targets:
            prepared = staging / relative
            if prepared.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(prepared, target)
            elif target.exists():
                target.unlink()
        updated = self.harness.instances.get(self.instance["instance_id"])
        updated["verification"] = {"status": "needs_verification", "reason": "实例源码已修改"}
        updated["needs_verification"] = True
        updated["status"] = "needs_verification"
        self.harness.store.write(f"instances/{updated['instance_id']}/instance.json", updated)
        self.harness.current_instance = updated
        self.harness.state.setdefault("changes", []).append({"modification_id": modification_id, "files": [item[0] for item in targets]})
        self.harness.state["verification"] = {"passed": False, "status": "needs_verification"}
        self.harness.task.setdefault("acceptance", {})["latest_patch_at"] = datetime.now(timezone.utc).isoformat()
        self.harness.events.emit("patch.applied", {"modification_id": modification_id, "files": [item[0] for item in targets], "diff": "".join(item["diff"] for item in changes)})
        await self.harness._save_session_async()
        return {"modification_id": modification_id, "changes": changes, "verification": "needs_verification"}

    async def bash(self, command, timeout=30):
        if timeout <= 0 or timeout > 120:
            raise ValueError("调查命令超时需要在 1 到 120 秒之间")
        if any(token in command for token in [";", "&&", "||", "`", "$", "\n", ">", "<", "|"]):
            raise PermissionError("调查命令不允许 Shell 拼接、重定向或变量展开")
        arguments = shlex.split(command)
        if not arguments:
            raise ValueError("命令为空")
        allowed = {"pwd", "ls", "stat"}
        if arguments[0] not in allowed:
            raise PermissionError("调查命令支持 pwd、ls、stat；文本搜索使用 search_text，环境操作使用领域工具")
        options = {"pwd": {"-L", "-P", "--"}, "ls": {"-l", "-a", "-la", "-al", "-h", "-lh", "-lah", "-1", "--color=never", "--"}, "stat": {"--"}}
        for argument in arguments[1:]:
            if argument.startswith("-"):
                if argument not in options[arguments[0]]:
                    raise PermissionError("调查命令选项未获得授权")
            else:
                self.code_path(argument)
        started = time.monotonic()
        environment = {name: value for name, value in os.environ.items() if not any(secret in name.upper() for secret in ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"])}
        process = await asyncio.create_subprocess_exec(*arguments, cwd=self.code_root, env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=os.name != "nt")
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            await process.wait()
            raise
        return {"command": arguments, "cwd": str(self.code_root), "return_code": process.returncode, "stdout": stdout.decode("utf-8", errors="replace"), "stderr": stderr.decode("utf-8", errors="replace"), "duration_seconds": time.monotonic() - started}


def register_tools(harness, registry):
    code = CodeTools(harness)
    string = {"type": "string"}
    integer = {"type": "integer"}
    registry.register("read_file", "读取当前实例源码和配置，支持 UTF-8 及带 BOM 的 UTF-16/UTF-32，正式记录行号、内容与版本供补丁审批使用；不存在的路径返回 exists=false、file_version=absent，作为新文件的创建依据", {"path": string, "start_line": integer, "end_line": integer}, ["path"], code.read_file)
    registry.register("search_text", "按 Python 正则逐行搜索当前实例文本；paths 默认为 examples、core，可指定源码相对路径。返回文件路径、行号、分页位置、搜索范围和跳过原因；支持 UTF-8 及带 BOM 的 UTF-16/UTF-32，不跟随符号链接；limit 1..1000，默认 50", {"pattern": string, "paths": {"type": "array", "items": string, "minItems": 1}, "limit": {"type": "integer", "minimum": 1, "maximum": SEARCH_MAX_LIMIT, "default": 50}, "cursor": {"type": "integer", "minimum": 0, "default": 0}}, ["pattern"], code.search_text)
    registry.register("read_evidence", "读取当前实例已登记的 Run、工具全文、独立验证报告或当前环境日志证据；使用工具返回的 reference/report_ref/static_report_ref", {"reference": string, "start_line": integer, "end_line": integer}, ["reference"], code.read_evidence)
    registry.register("apply_patch", "只提交完整 unified diff，程序自动绑定本任务 read_file 正式读取的版本并审批，执行前检查当前文件；未读取或文件变化时拒绝，创建新文件须先 read_file 确认不存在。使用 --- a/path、+++ b/path 和准确的 @@ 行范围，保留原文上下文并以换行结束", {"patch": string}, ["patch"], code.apply_patch, modifies=True, approval_context=code.patch_approval)
    registry.register("bash", "在实例源码目录运行只读调查命令，支持 pwd、ls、stat；文本搜索使用 search_text", {"command": string, "timeout": {"type": "number"}}, ["command"], code.bash)

    async def get_instance_state():
        instance = await asyncio.to_thread(harness.instances.get, code.instance["instance_id"])
        return await asyncio.to_thread(code._instance_summary, instance)

    async def run_example():
        return await harness._run()

    async def get_run_state(run_id=None):
        identity = run_id or harness.state.get("last_run_id")
        if not identity:
            raise ValueError("本任务尚无运行记录")
        return harness.run_summary(harness.runner.get_run(code.instance["instance_id"], identity))

    async def inspect_configuration(path=None):
        target = code.code_path(path or code.instance["benchmark_path"])
        result = {"path": str(target.relative_to(code.code_root)), "value": yaml.safe_load(target.read_text(encoding="utf-8")), "cwd": str(code.code_root), "instance_id": code.instance["instance_id"], "file_version": hashlib.sha256(target.read_bytes()).hexdigest(), "source_kind": "declared_configuration"}
        if result["path"] == code.instance["benchmark_path"]:
            result["configuration_chain"] = await asyncio.to_thread(configuration_evidence, code.code_root, code.instance["benchmark_path"])
        run_id = harness.state.get("last_run_id")
        if run_id:
            run = harness.runner.get_run(code.instance["instance_id"], run_id)
            result["last_execution"] = {"run_id": run_id, "command": run["command"], "cwd": run["cwd"], "environment_id": run["environment_id"], "workspace_override": run["output_dir"], "snapshot_ref": f"instances/{code.instance['instance_id']}/runs/{run_id}/run.json", "source_kind": "recorded_run"}
        return result

    async def compare_runs(before_run_id, after_run_id):
        runs = [harness.runner.get_run(code.instance["instance_id"], identity) for identity in (before_run_id, after_run_id)]
        return {"before": harness.run_summary(runs[0]), "after": harness.run_summary(runs[1]), "verification": {run["run_id"]: harness.store.read(f"instances/{code.instance['instance_id']}/runs/{run['run_id']}/verification.json", {}) for run in runs}}

    async def validate_instance():
        return await harness.verifier.validate(code.instance)

    async def verify_run(run_id=None):
        identity = run_id or harness.state.get("last_run_id")
        if not identity:
            raise ValueError("本任务尚无运行记录")
        run = harness.runner.get_run(code.instance["instance_id"], identity)
        result = await harness.verifier.verify(code.instance, run, harness.task.get("acceptance"))
        harness.current_instance = harness.instances.get(code.instance["instance_id"])
        harness.state["verification"] = harness.current_instance.get("verification") or {"passed": False, "status": "needs_verification"}
        await harness._save_session_async()
        return result

    async def get_change_impact(include_source_updates=False):
        instance = await harness._run_sync(harness.instances.check_changes, code.instance["instance_id"], include_source_updates)
        harness.current_instance = instance
        if instance.get("needs_verification"):
            harness.state["verification"] = instance.get("verification") or {"passed": False, "status": "needs_verification"}
        return await asyncio.to_thread(code._instance_summary, instance)

    registry.register("get_instance_state", "取得当前实例、环境与验证状态", {}, [], get_instance_state)
    registry.register("run_example", "在当前实例环境执行原 Ianvs benchmark，创建独立 Run 和产物", {}, [], run_example, modifies=True)
    registry.register("get_run_state", "取得实际运行阶段、异常、TestCase、指标和产物引用", {"run_id": string}, [], get_run_state)
    registry.register("inspect_configuration", "读取配置声明与来源；实际路径结合实例 cwd 和 Run 记录核查", {"path": string}, [], inspect_configuration)
    registry.register("compare_runs", "比较当前实例两个实际 Run 的阶段结果、指标、产物与独立验证", {"before_run_id": string, "after_run_id": string}, ["before_run_id", "after_run_id"], compare_runs)
    registry.register("validate_instance", "使用既有 validator 检查当前实例声明", {}, [], validate_instance)
    registry.register("verify_run", "由独立验证器检查指定本轮 Run 的原始任务与真实结果", {"run_id": string}, [], verify_run)
    registry.register("get_change_impact", "检查当前实例文件变化与验证状态；排查源项目更新时设 include_source_updates=true，会额外扫描来源目录", {"include_source_updates": {"type": "boolean"}}, [], get_change_impact)
    register_operations(harness, registry, code)
