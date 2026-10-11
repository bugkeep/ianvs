# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import hashlib
import ipaddress
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .dependencies import check_url, source_path
from .instances import IGNORED_DIRECTORIES, configuration_evidence, file_manifest, source_files, utc_now, validate_identity
from .packages import package_identity
from .processes import recorded_process, sandboxed_command, terminate_recorded_operation


def environment_summary(environment):
    fields = ("environment_id", "status", "created_at", "completed_at", "python_executable", "venv_root", "generation", "error", "summary", "warnings", "preparation_scope", "failed_step")
    summary = {key: environment[key] for key in fields if key in environment}
    summary["package_count"] = len(environment.get("packages") or [])
    summary["steps"] = [{key: step[key] for key in ("name", "label", "status", "exit_code", "log_ref", "started_at", "completed_at") if key in step} for step in environment.get("steps", [])]
    return summary


class EnvironmentManager:
    def __init__(self, settings, store, events):
        self.settings, self.store, self.events = settings, store, events
        self._preparing = set()
        self._active_task = None
        self._last_instance = None

    def _dependency_fingerprint(self, instance):
        root = Path(instance["code_root"])
        entry = instance.get("inventory_entry") or {}
        files = [root / "requirements.txt", root / "resources/third_party/sedna-0.6.0.1-py3-none-any.whl"]
        if entry.get("requirements_file"):
            files.append(root / entry["requirements_file"])
        for step in entry.get("prepare_env", {}).get("steps", []):
            files.append(root / entry["prepare_env"]["working_directory"] / step["script"])
        digest = hashlib.sha256(json.dumps(entry.get("prepare_env", {}), sort_keys=True).encode())
        digest.update(self.settings.python_executable.encode())
        for path in files:
            if path.exists():
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
        specification = (instance.get("environment") or {}).get("dependency_spec")
        if specification:
            digest.update(json.dumps(specification, sort_keys=True).encode())
            for name in specification.get("file_versions", {}):
                path = source_path(root, name)
                digest.update(name.encode())
                digest.update(path.read_bytes() if path.is_file() else b"missing")
        return digest.hexdigest()

    def _reference(self, instance, kind, identity, filename=None):
        validate_identity(instance["instance_id"], "实例")
        validate_identity(identity, kind)
        suffix = f"/{identity}/{filename}" if filename else f"/{identity}.json"
        return f"instances/{instance['instance_id']}/{kind}{suffix}"

    def _pip_cache(self, instance):
        validate_identity(instance["instance_id"], "实例")
        directory = self.store.root / "instances" / instance["instance_id"] / "pip-cache"
        if directory.resolve() != directory:
            raise PermissionError("pip 缓存必须属于当前实例的独立目录")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _cached_install(self, instance, command):
        return [*command[:4], "--cache-dir", str(self._pip_cache(instance)), *command[4:]]

    def _pip_writable_paths(self, instance, command):
        if list(command[1:4]) != ["-m", "pip", "install"]:
            return []
        cache = self._pip_cache(instance)
        if command.count("--cache-dir") != 1 or command[-1] == "--cache-dir" or command[command.index("--cache-dir") + 1] != str(cache):
            raise PermissionError("pip 安装命令需要使用当前实例批准的缓存目录")
        return [cache]

    @staticmethod
    def _loopback_cache_hosts(dependencies):
        hosts = set()
        for option in dependencies.get("options", []):
            if option["name"] != "find_links":
                continue
            for value in option["value"]:
                check_url(value)
                address = urlsplit(value)
                if address.scheme != "http" or not address.hostname:
                    continue
                try:
                    loopback = ipaddress.ip_address(address.hostname).is_loopback
                except ValueError:
                    loopback = address.hostname == "localhost"
                if loopback:
                    hosts.add(address.netloc)
        return sorted(hosts)

    @staticmethod
    def _strings(values, label):
        if values is None:
            return []
        if not isinstance(values, list) or not all(isinstance(value, str) and value and "\x00" not in value for value in values):
            raise ValueError(label + "需要非空字符串列表")
        return list(dict.fromkeys(values))

    def _request(self, instance, requirements_files=None, packages=None, constraints_files=None, modules=None, index_url=None):
        existing = (instance.get("environment") or {}).get("dependency_spec") or {}
        declared = (instance.get("inventory_entry") or {}).get("requirements_file")
        requirements = existing.get("requirements_files", [declared] if declared else []) if requirements_files is None else requirements_files
        additions = self._strings(packages, "packages")
        def distribution_name(value):
            try:
                requirement = Requirement(value)
            except InvalidRequirement:
                return None
            if requirement.url:
                check_url(requirement.url)
            return canonicalize_name(requirement.name)
        replacing = {name for value in additions if (name := distribution_name(value)) is not None}
        retained = [value for value in existing.get("packages", []) if distribution_name(value) not in replacing]
        request = {
            "requirements_files": self._strings(requirements, "requirements_files"),
            "constraints_files": self._strings(existing.get("constraints_files", []) if constraints_files is None else constraints_files, "constraints_files"),
            "packages": self._strings([*retained, *additions], "packages"),
            "modules": self._strings(modules, "modules"),
            "index_url": index_url or existing.get("index_url"),
        }
        root = Path(instance["code_root"]).resolve()
        for name in request["requirements_files"] + request["constraints_files"]:
            source_path(root, name)
        for value in request["packages"]:
            if value.startswith("-"):
                raise ValueError("packages 只能包含发行包要求，不能包含 pip 选项")
            check_url(value)
        if request["index_url"]:
            check_url(request["index_url"])
            if not request["index_url"].startswith("https://"):
                raise ValueError("index_url 需要 HTTPS 地址")
        return request

    @staticmethod
    def _fingerprint(data):
        return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _log_tail(reference):
        if not reference or not Path(reference).is_file():
            return ""
        path = Path(reference)
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 4096))
            return stream.read().decode("utf-8", errors="replace")[-1000:]

    @staticmethod
    def _dependency_command(python, root, specification):
        command = [str(python), "-m", "pip", "install", *[value for name in specification["requirements_files"] for value in ("-r", str(root / name))], *[value for name in specification["constraints_files"] for value in ("-c", str(root / name))], *specification["packages"]]
        if specification.get("index_url"):
            command.extend(["--index-url", specification["index_url"]])
        return command

    async def _worker(self, instance, python, request, directory, mode="inspect", mappings=()):
        directory.mkdir(parents=True, exist_ok=True)
        output_directory = directory / "worker-output"
        output_directory.mkdir(exist_ok=True)
        worker = Path(__file__).with_name("dependencies.py").resolve()
        output = output_directory / "observation.json"
        operation = {"operation_id": self.store.allocate_id("environment-operation"), "instance_id": instance["instance_id"], "task_id": self.events.identities.get("task_id"), "readonly_mappings": list(mappings), "started_at": utc_now()}
        reference = (directory / "operation.json").relative_to(self.store.root).as_posix()
        command = [str(python), "-I", str(worker), "--root", instance["code_root"], "--request", json.dumps(request), "--mode", mode, "--output", str(output)]
        try:
            await sandboxed_command(command, instance["code_root"], python, output_directory / "output.log", timeout=120, readonly_paths=[worker], writable_paths=[output_directory], operation=operation, on_update=lambda _: self.store.write(reference, operation))
        finally:
            operation["completed_at"] = utc_now()
            self.store.write(reference, operation)
        return json.loads(output.read_text(encoding="utf-8")), operation

    async def inspect(self, instance, requirements_files=None, modules=None, *, environment=None, request=None):
        environment = environment or instance.get("environment")
        request = request or self._request(instance, requirements_files=requirements_files, modules=modules)
        python = (environment or {}).get("python_executable") or self.settings.python_executable
        check_id = self.store.allocate_id("environment-check")
        reference = self._reference(instance, "environment-checks", check_id, "check.json")
        directory = self.store.path(reference).parent
        check = {"schema_version": 1, "check_id": check_id, "instance_id": instance["instance_id"], "task_id": self.events.identities.get("task_id"), "environment_id": (environment or {}).get("environment_id"), "generation": (environment or {}).get("generation", 0), "status": "running", "passed": False, "checked_at": utc_now(), "evidence_ref": reference, "source_kind": "observed_environment", "python_executable": str(python), "request": request}
        self.store.write(reference, check)
        try:
            result, operation = await self._worker(instance, python, request, directory)
            check.update(result)
            check.update(status="completed", checked_at=utc_now(), file_versions=result["dependencies"]["file_versions"], requirements_files=request["requirements_files"], dependency_fingerprint=self._fingerprint({"request": request, "file_versions": result["dependencies"]["file_versions"]}), **{key: operation.get(key) for key in ("command", "cwd", "log_ref", "isolation", "pid", "process_created_at")})
            expected = (environment or {}).get("packages")
            check["package_drift"] = bool(expected is not None and package_identity(expected) != package_identity(result["packages"]))
            if check["package_drift"]:
                check["passed"] = False
                check["drift_reason"] = "当前安装包版本与正式环境记录不同"
                if environment is instance.get("environment"):
                    environment.update(status="modified", observed_packages=result["packages"])
                    self._save(instance, environment)
            elif not check["passed"] and environment is instance.get("environment") and environment.get("status") == "ready":
                environment.update(status="modified", check_failure_ref=reference)
                self._save(instance, environment)
            if environment:
                environment["current_check"] = {key: check[key] for key in ("check_id", "passed", "checked_at", "evidence_ref", "environment_id", "generation", "dependency_fingerprint")}
                self._save(instance, environment, invalidate=False)
            instance["environment_check_ref"] = reference
            self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
            return check
        except (Exception, asyncio.CancelledError) as error:
            check.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=str(error), checked_at=utc_now())
            operation_reference = (directory / "operation.json").relative_to(self.store.root).as_posix()
            operation = self.store.read(operation_reference, {})
            if operation:
                check.update(operation_ref=operation_reference, **{key: operation.get(key) for key in ("command", "cwd", "log_ref", "isolation", "phase", "command_executed", "exit_code", "pid", "process_created_at")})
            if environment:
                if environment is instance.get("environment") and isinstance(error, FileNotFoundError) and not Path(python).is_file():
                    environment.update(status="modified", check_failure_ref=reference)
                environment["current_check"] = {key: check.get(key) for key in ("check_id", "passed", "checked_at", "evidence_ref", "environment_id", "generation", "dependency_fingerprint")}
                self._save(instance, environment)
            instance["environment_check_ref"] = reference
            self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
            raise
        finally:
            self.store.write(reference, check)
            self.events.emit("environment.inspected", check, instance_id=instance["instance_id"], environment_id=check["environment_id"])

    @staticmethod
    def _source_directories(root):
        root = Path(root)
        result = []
        for directory, folders, _ in os.walk(root):
            folders[:] = sorted(name for name in folders if name not in IGNORED_DIRECTORIES and not (Path(directory) / name).is_symlink())
            result.extend((Path(directory) / name).relative_to(root).as_posix() for name in folders)
        return sorted(result)

    def _new_plan(self, instance, action, request, timeout):
        existing = instance.get("environment") or {}
        root = Path(instance["code_root"]).resolve()
        plan_id = self.store.allocate_id("environment-plan")
        environment_id = self.store.allocate_id("environment")
        reference = self._reference(instance, "environment-plans", plan_id)
        directory = self.store.path(reference).with_suffix("")
        snapshot = directory / "input-snapshot"
        snapshot.mkdir(parents=True)
        directories = self._source_directories(root)
        for name in directories:
            (snapshot / name).mkdir(parents=True, exist_ok=True)
        versions = {}
        for path in source_files(root):
            relative = path.relative_to(root)
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            versions[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
        if file_manifest(root) != versions or self._source_directories(root) != directories:
            raise RuntimeError("复制依赖计划期间实例文件发生变化，请重新生成计划")
        venv = self.settings.state_root / f"environments/{environment_id}/venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command = self._cached_install(instance, [str(python), "-m", "pip", "install", "--disable-pip-version-check"])
        report = venv.parent / "pip-output" / "install-report.json"
        command.extend(["--report", str(report)])
        for name in request.get("requirements_files", []):
            command.extend(["-r", str(root / name)])
        for name in request.get("constraints_files", []):
            command.extend(["-c", str(root / name)])
        command.extend(request.get("packages", []))
        if request.get("index_url"):
            command.extend(["--index-url", request["index_url"]])
        plan = {"schema_version": 1, "plan_id": plan_id, "action": action, "instance_id": instance["instance_id"], "task_id": self.events.identities.get("task_id"), "status": "planned", "requires_install": True, "created_at": utc_now(), "evidence_ref": reference, "code_root": str(root), "base_environment_id": existing.get("environment_id"), "base_generation": existing.get("generation", 0), "candidate_environment_id": environment_id, "candidate_generation": existing.get("generation", 0) + 1, "base_python": self.settings.python_executable, "python_executable": str(python), "venv_root": str(venv), "snapshot_root": str(snapshot), "file_versions": versions, "snapshot_fingerprint": self._fingerprint(versions), "request": request, "timeout": timeout, "commands": [{"command": [self.settings.python_executable, "-m", "venv", str(venv)], "cwd": str(root), "network": False}, {"command": command, "cwd": str(root), "network": True}], "dependency_fingerprint": self._fingerprint({"request": request, "file_versions": versions})}
        plan.update(source_directories=directories, snapshot_fingerprint=self._fingerprint({"file_versions": versions, "source_directories": directories}), pip_cache_dir=str(self._pip_cache(instance)), pip_cache_ref=f"instances/{instance['instance_id']}/pip-cache")
        self.store.write(reference, plan)
        if action == "install":
            plan["status"] = "planning"
            self.store.write(reference, plan)
            instance["environment_operation_ref"] = reference
            self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        return plan

    def _prepare_request(self, instance):
        root = Path(instance["code_root"])
        entry = instance.get("inventory_entry") or {}
        preparation = entry.get("prepare_env") or {}
        specification = (instance.get("environment") or {}).get("dependency_spec") or {}
        requirements = ["requirements.txt", *specification.get("requirements_files", [])]
        declared = entry.get("requirements_file")
        if declared and not any(item.get("type") == "dependency" for item in preparation.get("steps", [])):
            requirements.append(declared)
        return self._request(instance, requirements_files=requirements, packages=[*specification.get("packages", []), str(root / "resources/third_party/sedna-0.6.0.1-py3-none-any.whl")], constraints_files=specification.get("constraints_files", []), modules=specification.get("modules", []))

    def _prepare_parse_error(self, plan):
        operation = plan.get("parse_operation") or self.store.read(plan.get("parse_operation_ref"), {})
        log = operation.get("log_ref")
        return f"{plan['error']}；{self._log_tail(log)}；plan_id={plan['plan_id']}；evidence_ref={plan['evidence_ref']}；log_ref={log}"

    def preview_prepare(self, instance):
        request = self._prepare_request(instance)
        preparation = (instance.get("inventory_entry") or {}).get("prepare_env") or {}
        cached = instance.get("prepare_plan_id")
        if cached:
            plan = self.store.read(self._reference(instance, "environment-plans", cached))
            if plan and plan["status"] == "planned" and ("pip_cache_dir" not in plan or "pip_cache_ref" not in plan):
                plan.update(status="invalid", invalidated_at=utc_now(), error="环境准备需要重新生成包含实例缓存目录的计划并审批")
                self.store.write(plan["evidence_ref"], plan)
            same_inputs = plan and plan.get("request") == request and plan.get("preparation") == preparation and plan["file_versions"] == file_manifest(instance["code_root"]) and plan.get("source_directories") == self._source_directories(instance["code_root"]) and plan["base_environment_id"] == (instance.get("environment") or {}).get("environment_id") and plan["base_generation"] == (instance.get("environment") or {}).get("generation", 0) and plan["base_python"] == self.settings.python_executable
            if same_inputs and plan["status"] == "planned":
                return self.revalidate(instance, cached)
        plan = self._new_plan(instance, "prepare", request, 1800)
        root, python = Path(instance["code_root"]), plan["python_executable"]
        plan["commands"][1]["command"] = self._cached_install(instance, [python, "-m", "pip", "install", "-r", str(root / "requirements.txt"), str(root / "resources/third_party/sedna-0.6.0.1-py3-none-any.whl")])
        plan["preparation"] = copy.deepcopy(preparation)
        declared = (instance.get("inventory_entry") or {}).get("requirements_file")
        if declared and not any(item.get("type") == "dependency" for item in preparation.get("steps", [])):
            plan["commands"].append({"command": self._cached_install(instance, [python, "-m", "pip", "install", "-r", str(root / declared)]), "cwd": str(root), "network": True})
        for index, item in enumerate(preparation.get("steps", [])):
            working = source_path(root, preparation["working_directory"])
            script = source_path(root, working / item["script"])
            execution_script = script
            if script.suffix == ".sh" and b"\r\n" in script.read_bytes():
                execution_script = Path(plan["venv_root"]).parent / "prepared-scripts" / f"prepare-{index}.sh"
            plan["commands"].append({"command": [python if script.suffix == ".py" else "sh", str(execution_script), *item.get("args", [])], "cwd": str(working), "network": item.get("type") == "dependency", "script_ref": str(script.relative_to(root)), "script_version": plan["file_versions"][script.relative_to(root).as_posix()]})
        specification = (instance.get("environment") or {}).get("dependency_spec")
        if specification:
            plan["commands"].append({"command": self._cached_install(instance, self._dependency_command(python, root, specification)), "cwd": str(root), "network": True})
        self.store.write(plan["evidence_ref"], plan)
        instance["prepare_plan_id"] = plan["plan_id"]
        self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        return plan

    async def _preflight_prepare(self, instance, plan):
        if plan is None:
            staging = asyncio.create_task(asyncio.to_thread(self.preview_prepare, instance))
            try:
                plan = await asyncio.shield(staging)
            except asyncio.CancelledError:
                plan = await staging
                plan.update(status="cancelled", completed_at=utc_now(), error="环境准备计划快照已取消")
                await asyncio.to_thread(self.store.write, plan["evidence_ref"], plan)
                raise
        directory = self.store.path(plan["evidence_ref"]).with_suffix("")
        mappings = [{"source": plan["snapshot_root"], "destination": plan["code_root"]}]
        plan.update(status="planning", parse_operation_ref=(directory / "parse/operation.json").relative_to(self.store.root).as_posix(), parse_result_ref=(directory / "parse/worker-output/observation.json").relative_to(self.store.root).as_posix())
        self.store.write(plan["evidence_ref"], plan)
        instance["environment_operation_ref"] = plan["evidence_ref"]
        self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        try:
            preparation = (instance.get("inventory_entry") or {}).get("prepare_env") or {}
            if plan["request"] != self._prepare_request(instance) or plan.get("preparation") != preparation:
                raise ValueError("准备计划的依赖声明已经变化，请重新生成计划")
            parsed, operation = await self._worker(instance, self.settings.python_executable, plan["request"], directory / "parse", "dependencies", mappings)
            plan.update(dependencies=parsed, parse_operation=operation, requirements_file_versions=parsed["file_versions"])
            if not parsed["passed"]:
                raise ValueError("依赖声明无法解析：" + json.dumps(parsed["errors"][:3], ensure_ascii=False)[:1000])
            if any(plan["file_versions"].get(name) != version for name, version in parsed["file_versions"].items()):
                raise RuntimeError("依赖解析结果与准备计划快照版本不同")
            await asyncio.to_thread(self._validate_plan_inputs, instance, plan)
            plan.update(status="planned", parsed_at=utc_now())
            return plan
        except (Exception, asyncio.CancelledError) as error:
            plan.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "invalid", error=str(error) or "环境准备依赖解析已取消", completed_at=utc_now())
            if isinstance(error, (PermissionError, ValueError, RuntimeError, TimeoutError)):
                raise type(error)(self._prepare_parse_error(plan)) from error
            raise
        finally:
            self.store.write(plan["evidence_ref"], plan)

    def _validate_plan_inputs(self, instance, plan):
        if plan["file_versions"] != file_manifest(instance["code_root"]):
            raise ValueError("依赖计划批准的文件版本已经变化，请重新生成计划")
        if plan["file_versions"] != file_manifest(plan["snapshot_root"]):
            raise RuntimeError("依赖计划输入快照发生变化")
        if "source_directories" not in plan:
            raise ValueError("依赖计划缺少目录版本记录，请重新生成计划")
        if plan["source_directories"] != self._source_directories(instance["code_root"]):
            raise ValueError("依赖计划批准的目录已经变化，请重新生成计划")
        if plan["source_directories"] != self._source_directories(plan["snapshot_root"]):
            raise RuntimeError("依赖计划输入快照目录发生变化")

    def revalidate(self, instance, plan_id):
        plan = self.store.read(self._reference(instance, "environment-plans", plan_id))
        if not plan or plan.get("plan_id") != plan_id or plan.get("instance_id") != instance["instance_id"]:
            raise ValueError("依赖计划未登记到当前实例")
        existing = instance.get("environment") or {}
        if plan["status"] != "planned":
            raise ValueError("依赖计划已经执行或失效")
        if plan["base_environment_id"] != existing.get("environment_id") or plan["base_generation"] != existing.get("generation", 0):
            raise ValueError("依赖计划引用的环境代次已经变化，请重新生成计划")
        if plan["base_python"] != self.settings.python_executable or plan["code_root"] != str(Path(instance["code_root"]).resolve()):
            raise ValueError("依赖计划的解释器或工作目录已经变化")
        if plan.get("pip_cache_dir") != str(self._pip_cache(instance)) or plan.get("pip_cache_ref") != f"instances/{instance['instance_id']}/pip-cache":
            raise PermissionError("依赖计划的 pip 缓存不属于当前实例")
        for operation in plan["commands"]:
            self._pip_writable_paths(instance, operation["command"])
        self._validate_plan_inputs(instance, plan)
        for name, version in plan.get("approval_file_versions", {}).items():
            if hashlib.sha256(self.store.path(name).read_bytes()).hexdigest() != version:
                raise RuntimeError("依赖计划正式解析材料发生变化")
        return plan

    async def _resolve_plan(self, instance, plan, directory, mappings):
        output = directory / "resolve-output"
        output.mkdir(exist_ok=True)
        report_path = output / "resolution.json"
        command = list(plan["commands"][1]["command"])
        command[0] = self.settings.python_executable
        command[command.index("--report") + 1] = str(report_path)
        command.extend(["--dry-run", "--ignore-installed"])
        operation = {"operation_id": self.store.allocate_id("environment-operation"), "readonly_mappings": mappings, "started_at": utc_now()}
        reference = (directory / "resolve-operation.json").relative_to(self.store.root).as_posix()
        plan["resolve_operation_ref"] = reference
        self.store.write(plan["evidence_ref"], plan)
        try:
            await sandboxed_command(command, plan["code_root"], self.settings.python_executable, output / "output.log", timeout=plan["timeout"], writable_paths=[output, *self._pip_writable_paths(instance, command)], network=True, operation=operation, on_update=lambda _: self.store.write(reference, operation))
        finally:
            operation["completed_at"] = utc_now()
            self.store.write(reference, operation)
            plan["resolve_operation"] = operation
        report = json.loads(report_path.read_text(encoding="utf-8"))
        frozen = {}
        for item in report["install"]:
            metadata = item["metadata"]
            requirement = Requirement(f"{metadata['name']}=={metadata['version']}")
            frozen[canonicalize_name(requirement.name)] = str(requirement.specifier)
        constraints = directory / "resolved-constraints.txt"
        constraints.write_text("".join(f"{name}{version}\n" for name, version in sorted(frozen.items())), encoding="utf-8")
        resolution_reference = (directory / "resolution.json").relative_to(self.store.root).as_posix()
        self.store.write(resolution_reference, report)
        constraints_reference = constraints.relative_to(self.store.root).as_posix()
        plan.update(resolved_packages=frozen, resolution_ref=resolution_reference, constraints_ref=constraints_reference, approval_file_versions={name: hashlib.sha256(self.store.path(name).read_bytes()).hexdigest() for name in (resolution_reference, constraints_reference)})
        plan["commands"][1]["command"].extend(["-c", str(constraints)])
        plan["commands"][1]["readonly_paths"] = [str(constraints)]

    def _verify_resolution(self, plan, installed_report, packages):
        installed = {canonicalize_name(item["name"]): item["version"] for item in packages}
        for name, specifier in plan["resolved_packages"].items():
            if name not in installed or not Requirement(name + specifier).specifier.contains(installed[name], prereleases=True):
                raise RuntimeError("实际安装版本与批准的依赖解析结果不同：" + name)
        expected = {canonicalize_name(item["metadata"]["name"]): item for item in self.store.read(plan["resolution_ref"])["install"]}
        for item in installed_report["install"]:
            name = canonicalize_name(item["metadata"]["name"])
            if name not in expected:
                raise RuntimeError("pip 安装了依赖计划之外的发行包：" + name)
            planned = expected[name].get("download_info", {}).get("archive_info", {}).get("hashes", {})
            observed = item.get("download_info", {}).get("archive_info", {}).get("hashes", {})
            if planned and planned != observed:
                raise RuntimeError("实际安装制品与批准的依赖解析结果不同：" + name)

    async def plan_install(self, instance, requirements_files=None, packages=None, constraints_files=None, index_url=None, timeout=1800, modules=None):
        identity = instance["instance_id"]
        if identity in self._preparing:
            raise RuntimeError("当前实例正在准备环境")
        self._preparing.add(identity)
        self._active_task = asyncio.current_task()
        try:
            await asyncio.to_thread(self.recover, instance)
            return await self._plan_install(instance, requirements_files, packages, constraints_files, index_url, timeout, modules)
        finally:
            self._preparing.remove(identity)
            self._active_task = None

    async def _plan_install(self, instance, requirements_files=None, packages=None, constraints_files=None, index_url=None, timeout=1800, modules=None):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 7200:
            raise ValueError("依赖安装时限必须在 0 到 7200 秒之间")
        request = self._request(instance, requirements_files, packages, constraints_files, modules, index_url)
        root = Path(instance["code_root"])
        if (root / "requirements.txt").is_file() and "requirements.txt" not in request["requirements_files"]:
            request["requirements_files"].insert(0, "requirements.txt")
        wheel = "resources/third_party/sedna-0.6.0.1-py3-none-any.whl"
        if (root / wheel).is_file() and str(root / wheel) not in request["packages"]:
            request["packages"].append(str(root / wheel))
        if not request["requirements_files"] and not request["packages"]:
            raise ValueError("依赖计划需要 requirements_files 或 packages")
        staging = asyncio.create_task(asyncio.to_thread(self._new_plan, instance, "install", request, timeout))
        try:
            plan = await asyncio.shield(staging)
        except asyncio.CancelledError:
            plan = await staging
            plan.update(status="cancelled", completed_at=utc_now(), error="依赖计划快照准备已取消")
            await asyncio.to_thread(self.store.write, plan["evidence_ref"], plan)
            raise
        directory = self.store.path(plan["evidence_ref"]).with_suffix("")
        mappings = [{"source": plan["snapshot_root"], "destination": plan["code_root"]}]
        plan["parse_operation_ref"] = (directory / "parse/operation.json").relative_to(self.store.root).as_posix()
        self.store.write(plan["evidence_ref"], plan)
        try:
            parsed, operation = await self._worker(instance, self.settings.python_executable, request, directory / "parse", "dependencies", mappings)
            plan.update(dependencies=parsed, parse_operation=operation)
            if not parsed["passed"]:
                raise ValueError("依赖声明无法解析：" + json.dumps(parsed["errors"][:3], ensure_ascii=False)[:1000])
            plan["requirements_file_versions"] = parsed["file_versions"]
            await asyncio.to_thread(self._validate_plan_inputs, instance, plan)
            # 只为正式依赖声明中的本机 HTTP 来源启用 pip 的标准缓存适配器。
            plan["cache_trusted_hosts"] = self._loopback_cache_hosts(parsed)
            plan["commands"][1]["command"].extend(value for host in plan["cache_trusted_hosts"] for value in ("--trusted-host", host))
            requested = {item["name"].lower().replace("_", "-") for item in parsed["requirements"] if item.get("name")}
            preserved = [f"{item['name']}=={item['version']}" for item in (instance.get("environment") or {}).get("packages") or [] if item["name"].lower().replace("_", "-") not in requested and item["name"].lower() != "pip"]
            plan["preserved_packages"] = preserved
            plan["commands"][1]["command"].extend(preserved)
            await self._resolve_plan(instance, plan, directory, mappings)
            await asyncio.to_thread(self._validate_plan_inputs, instance, plan)
            plan["status"] = "planned"
            return plan
        except (Exception, asyncio.CancelledError) as error:
            plan.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "invalid", error=str(error))
            if isinstance(error, (ValueError, RuntimeError, TimeoutError)):
                operation = plan.get("resolve_operation") or plan.get("parse_operation") or self.store.read(plan.get("resolve_operation_ref") or plan["parse_operation_ref"], {})
                log = operation.get("log_ref")
                raise type(error)(f"{error}；{self._log_tail(log)}；plan_id={plan['plan_id']}；evidence_ref={plan['evidence_ref']}；log_ref={log}") from error
            raise
        finally:
            self.store.write(plan["evidence_ref"], plan)

    async def install(self, instance, plan_id):
        identity = instance["instance_id"]
        if identity in self._preparing:
            raise RuntimeError("当前实例正在准备环境")
        self._preparing.add(identity)
        self._active_task = asyncio.current_task()
        self._last_instance = instance
        plan = None
        environment = None
        try:
            await asyncio.to_thread(self.recover, instance)
            plan = await asyncio.to_thread(self.revalidate, instance, plan_id)
            if plan["action"] != "install":
                raise ValueError("依赖安装需要 install 计划")
            plan.update(status="installing", started_at=utc_now())
            self.store.write(plan["evidence_ref"], plan)
            venv = Path(plan["venv_root"])
            venv.mkdir(parents=True, exist_ok=False)
            directory = venv.parent
            output_directory = directory / "pip-output"
            output_directory.mkdir(exist_ok=True)
            environment = {"environment_id": plan["candidate_environment_id"], "instance_id": identity, "generation": plan["candidate_generation"], "status": "preparing", "candidate": True, "created_at": utc_now(), "base_python": plan["base_python"], "python_executable": plan["python_executable"], "venv_root": str(venv), "plan_id": plan_id, "steps": [], "packages": None, "dependency_spec": {**plan["request"], "file_versions": plan["requirements_file_versions"]}, "dependency_fingerprint": plan["dependency_fingerprint"]}
            self._save(instance, environment)
            mappings = [{"source": plan["snapshot_root"], "destination": plan["code_root"]}]
            for index, command in enumerate(plan["commands"]):
                await self._run_step(instance, environment, "create_venv" if index == 0 else "dependency_install", command["command"], command["cwd"], plan["base_python"] if index == 0 else plan["python_executable"], directory / f"install-{index}.log", timeout=plan["timeout"], writable_paths=[venv, output_directory], readonly_paths=command.get("readonly_paths", []), network=command["network"], readonly_mappings=mappings)
            report_path = output_directory / "install-report.json"
            environment["pip_report_ref"] = report_path.relative_to(self.store.root).as_posix()
            environment["pip_report"] = json.loads(report_path.read_text(encoding="utf-8"))
            # 检查当前源码，保证安装期间发生的文件变化不能成为可用环境。
            await asyncio.to_thread(self._validate_plan_inputs, instance, plan)
            check = await self.inspect(instance, environment=environment, request=plan["request"])
            environment["postcheck"] = check
            if not check["passed"]:
                raise RuntimeError("候选环境依赖或模块检查失败，保留当前环境；evidence_ref=" + check["evidence_ref"])
            self._verify_resolution(plan, environment["pip_report"], check["packages"])
            await asyncio.to_thread(self._validate_plan_inputs, instance, plan)
            environment.update(status="ready", packages=check["packages"], completed_at=utc_now(), summary="依赖安装与环境检查通过", warnings=[], preparation_scope={"model_data_status": "not_verified"})
            environment["dependency_fingerprint"] = self._dependency_fingerprint({**instance, "environment": environment})
            environment["candidate"] = False
            self._save(instance, environment)
            plan.update(status="completed", completed_at=utc_now(), postcheck_ref=check["evidence_ref"])
            self.events.emit("environment.completed", environment_summary(environment), instance_id=identity, environment_id=environment["environment_id"])
            return environment
        except (Exception, asyncio.CancelledError) as error:
            if environment:
                environment.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=str(error) or "操作已取消", completed_at=utc_now())
                self._save(instance, environment)
                self.events.emit("environment.failed", environment_summary(environment), instance_id=identity, environment_id=environment["environment_id"])
            if plan:
                plan.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=str(error) or "操作已取消", completed_at=utc_now())
            if plan and isinstance(error, (ValueError, RuntimeError, TimeoutError)):
                failed_step = (environment or {}).get("failed_step") or {}
                check = (environment or {}).get("postcheck") or {}
                details = failed_step.get("error_tail") or json.dumps({"imports": [item for item in check.get("imports", []) if not item["passed"]], "pip_check": check.get("pip_check"), "dependencies": (check.get("dependencies") or {}).get("errors", [])}, ensure_ascii=False)[:1000]
                raise type(error)(f"{error}；{details}；plan_id={plan['plan_id']}；evidence_ref={plan['evidence_ref']}；log_ref={failed_step.get('log_ref', check.get('log_ref'))}") from error
            raise
        finally:
            if plan:
                self.store.write(plan["evidence_ref"], plan)
            self._preparing.remove(identity)
            self._active_task = None

    def recover(self, instance):
        self._last_instance = instance
        plan_reference = instance.get("environment_operation_ref")
        if plan_reference:
            plan = self.store.read(plan_reference)
            if not plan or plan.get("instance_id") != instance["instance_id"]:
                raise ValueError("环境操作计划不属于当前实例")
            if plan["status"] == "planning":
                for field in ("parse_operation_ref", "resolve_operation_ref"):
                    operation = self.store.read(plan[field]) if plan.get(field) else None
                    if operation and operation.get("status") in {"starting", "running"} and recorded_process(operation) is not None:
                        raise RuntimeError("依赖解析进程仍然存在，请取消所属操作后恢复")
                plan.update(status="interrupted", completed_at=utc_now())
                self.store.write(plan_reference, plan)
        pending = instance.get("pending_environment_id") or ((instance.get("environment") or {}).get("environment_id") if (instance.get("environment") or {}).get("status") == "preparing" else None)
        if not pending:
            return None
        validate_identity(pending, "环境")
        environment = self.store.read(f"environments/{pending}/environment.json")
        if not environment or environment.get("instance_id") != instance["instance_id"]:
            raise ValueError("待恢复环境记录不属于当前实例")
        if environment["status"] != "preparing":
            return environment
        for step in environment.get("steps", []):
            if step.get("status") not in {"running", "starting"} or not step.get("pid"):
                continue
            if recorded_process(step) is not None:
                raise RuntimeError("环境操作进程仍然存在，请取消所属操作后恢复")
            step.update(status="interrupted", completed_at=utc_now())
        environment.update(status="interrupted", error="环境准备被中断，候选环境未激活", completed_at=utc_now())
        self._save(instance, environment)
        plan_id = environment.get("plan_id")
        if plan_id:
            reference = self._reference(instance, "environment-plans", plan_id)
            plan = self.store.read(reference)
            plan.update(status="interrupted", completed_at=utc_now())
            self.store.write(reference, plan)
        self.events.emit("environment.interrupted", environment_summary(environment), instance_id=instance["instance_id"], environment_id=pending)
        return environment

    async def cancel(self, instance=None):
        if self._active_task and self._active_task is not asyncio.current_task():
            task = self._active_task
            task.cancel()
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            if isinstance(result, Exception):
                raise result
            return True
        instance = instance or self._last_instance
        if instance is None:
            return False
        cancelled = False
        plan = self.store.read(instance["environment_operation_ref"]) if instance.get("environment_operation_ref") else None
        if plan and plan.get("status") == "planning":
            if plan.get("instance_id") != instance["instance_id"]:
                raise ValueError("环境操作计划不属于当前实例")
            for field in ("parse_operation_ref", "resolve_operation_ref"):
                operation = self.store.read(plan[field]) if plan.get(field) else None
                if operation and operation.get("status") in {"starting", "running"}:
                    cancelled = await terminate_recorded_operation(operation) or cancelled
        pending = instance.get("pending_environment_id") or ((instance.get("environment") or {}).get("environment_id") if (instance.get("environment") or {}).get("status") == "preparing" else None)
        if pending is None:
            if plan and plan.get("status") == "planning":
                await asyncio.to_thread(self.recover, instance)
            return cancelled
        validate_identity(pending, "环境")
        environment = self.store.read(f"environments/{pending}/environment.json")
        if not environment or environment.get("instance_id") != instance["instance_id"]:
            raise ValueError("环境操作不属于当前实例")
        if environment.get("status") == "preparing":
            for step in environment.get("steps", []):
                if step.get("status") in {"starting", "running"}:
                    cancelled = await terminate_recorded_operation(step) or cancelled
        await asyncio.to_thread(self.recover, instance)
        return cancelled

    async def prepare(self, instance, install=True, plan_id=None):
        instance_id = instance["instance_id"]
        if instance_id in self._preparing:
            raise RuntimeError("当前实例正在准备环境")
        self._preparing.add(instance_id)
        self._active_task = asyncio.current_task()
        try:
            await asyncio.to_thread(self.recover, instance)
            plan = await asyncio.to_thread(self.revalidate, instance, plan_id) if plan_id else None
            if plan and plan["action"] != "prepare":
                raise ValueError("准备环境需要 prepare 计划")
            return await self._prepare(instance, install, plan)
        finally:
            self._preparing.remove(instance_id)
            self._active_task = None

    def _describe(self, instance, environment):
        root = Path(instance["code_root"])
        entry = instance.get("inventory_entry") or {}
        preparation = entry.get("prepare_env") or {}
        requirements = entry.get("requirements_file")
        scripts = preparation.get("steps", [])
        candidates = [path.relative_to(root).as_posix() for name in ("requirements.txt", "requirement.txt") if (path := root / Path(instance["benchmark_path"]).parent / name).is_file()]
        selected = (environment.get("dependency_spec") or {}).get("requirements_files", [])
        undeclared = [] if requirements or any(step.get("type") == "dependency" for step in scripts) else [path for path in candidates if path not in selected]
        configuration = configuration_evidence(root, instance["benchmark_path"])
        warnings = [f"发现未声明的 Example 依赖文件，尚未安装：{path}" for path in undeclared]
        if not requirements and not selected and not any(step.get("type") == "dependency" for step in scripts):
            warnings.append("Example 未声明依赖安装步骤，请根据 README 核实所需依赖")
        warnings.extend(f"配置检查：{issue}" for issue in configuration["issues"])
        warnings.append("本次环境准备未验证模型和数据，请查看实际运行与独立验证结果")
        environment["preparation_scope"] = {
            "core_dependencies": "installed" if any((step.get("name") == "core_install" or Path(step.get("log_ref", "")).name == "core-install.log") and step.get("exit_code") == 0 for step in environment["steps"]) else "not_installed",
            "requirements_file": requirements,
            "discovered_requirements": candidates,
            "undeclared_requirements": undeclared,
            "selected_requirements": selected,
            "prepare_scripts": [{key: step.get(key) for key in ("name", "type", "script")} for step in scripts],
            "configuration_status": configuration["status"],
            "model_data_status": "not_verified",
        }
        environment["warnings"] = warnings
        completed = sum(step.get("exit_code") == 0 for step in environment["steps"])
        environment["summary"] = f"已完成声明的环境步骤（{completed} 项）；本次准备未验证模型和数据" if environment["status"] == "ready" else "独立 Python 环境已创建，尚未安装依赖" if environment["status"] == "created" else f"环境准备{'已取消' if environment['status'] == 'cancelled' else '失败'}：{environment.get('error', '')}"

    async def _run_step(self, instance, environment, name, command, cwd, python, log_path, timeout=900, *, writable_paths=(), readonly_paths=(), network=False, readonly_mappings=()):
        labels = {"create_venv": "创建独立 Python 环境", "core_install": "安装 Ianvs 基础依赖与 Sedna", "example_install": "安装 Example 声明的依赖", "record_packages": "记录已安装依赖版本"}
        step = {"operation_id": self.store.allocate_id("environment-operation"), "name": name, "label": labels.get(name, name), "status": "running", "command": list(command), "log_ref": str(log_path), "exit_code": None, "started_at": utc_now(), "readonly_mappings": list(readonly_mappings or environment.get("readonly_mappings", []))}
        environment["steps"].append(step)
        self._save(instance, environment)
        identities = {"instance_id": instance["instance_id"], "environment_id": environment["environment_id"]}
        self.events.emit("environment.step_started", step, **identities)
        try:
            await sandboxed_command(command, cwd, python, log_path, timeout=timeout, writable_paths=[*writable_paths, *self._pip_writable_paths(instance, command)], readonly_paths=readonly_paths, network=network, operation=step, on_update=lambda _: self._save(instance, environment))
        except (Exception, asyncio.CancelledError) as error:
            step.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=str(error) or "操作已取消", completed_at=utc_now())
            if Path(log_path).is_file():
                step["error_tail"] = self._log_tail(log_path)
            environment["failed_step"] = dict(step)
            self._save(instance, environment)
            self.events.emit("environment.step_failed", step, **identities)
            raise
        step.update(status="completed", completed_at=utc_now())
        self._save(instance, environment)
        self.events.emit("environment.step_completed", step, **identities)
        return step

    async def _prepare(self, instance, install, plan=None):
        fingerprint = self._dependency_fingerprint(instance)
        existing = instance.get("environment")
        if existing and existing.get("status") == "ready" and existing.get("dependency_fingerprint") == fingerprint:
            if not Path(existing["python_executable"]).is_file():
                raise FileNotFoundError("实例解释器缺失，需要重新创建环境")
            package_log = self.settings.state_root / f"environments/{existing['environment_id']}/packages-check.json"
            checked = await self.inspect(instance, modules=(existing.get("dependency_spec") or {}).get("modules", []))
            package_log.parent.mkdir(parents=True, exist_ok=True)
            package_log.write_text(json.dumps(checked["packages"], ensure_ascii=False), encoding="utf-8")
            observed = checked["packages"]
            if checked["passed"]:
                if plan:
                    await asyncio.to_thread(self._validate_plan_inputs, instance, plan)
                existing["postcheck"] = checked
                self._describe(instance, existing)
                self._save(instance, existing, invalidate=False)
                self.events.emit("environment.completed", {"environment_id": existing["environment_id"], "status": "ready", "summary": existing["summary"], "warnings": existing["warnings"], "reused": True}, instance_id=instance["instance_id"])
                if plan:
                    plan.update(status="completed", reused_environment_id=existing["environment_id"], completed_at=utc_now())
                    self.store.write(plan["evidence_ref"], plan)
                return existing
            if not checked["package_drift"]:
                raise RuntimeError("已有环境检查失败，请查看正式检查报告并准备新的依赖计划")
            existing["status"] = "modified"
            existing["observed_packages"] = observed
            self._save(instance, existing)
            self.events.emit("environment.changed", {"environment_id": existing["environment_id"], "reason": "已安装依赖发生变化，按照当前声明创建新环境代次"}, instance_id=instance["instance_id"])
        if install:
            plan = await self._preflight_prepare(instance, plan)
        environment_id = plan["candidate_environment_id"] if plan else self.store.allocate_id("environment")
        relative = f"environments/{environment_id}"
        environment_root = self.settings.state_root / relative
        venv = environment_root / "venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        environment = {"environment_id": environment_id, "instance_id": instance["instance_id"], "generation": (existing or {}).get("generation", 0) + 1, "status": "preparing", "created_at": utc_now(), "python_executable": str(python), "base_python": self.settings.python_executable, "venv_root": str(venv), "dependency_fingerprint": fingerprint, "steps": [], "packages": None, "candidate": True, "isolation": {"workspace": True, "python_packages": True, "process": True}}
        if existing and existing.get("dependency_spec"):
            environment["dependency_spec"] = copy.deepcopy(existing["dependency_spec"])
        if plan:
            plan.update(status="installing", started_at=utc_now())
            self.store.write(plan["evidence_ref"], plan)
            environment.update(plan_id=plan["plan_id"], readonly_mappings=[{"source": plan["snapshot_root"], "destination": plan["code_root"]}])
        environment_root.mkdir(parents=True, exist_ok=True)
        venv.mkdir(exist_ok=True)
        self._save(instance, environment)
        self.events.emit("environment.started", {"environment_id": environment_id}, instance_id=instance["instance_id"])
        root = Path(instance["code_root"])
        try:
            await self._run_step(instance, environment, "create_venv", [self.settings.python_executable, "-m", "venv", str(venv)], root, self.settings.python_executable, environment_root / "create.log", writable_paths=[venv])
            if install:
                dependencies = [str(root / "requirements.txt"), str(root / "resources/third_party/sedna-0.6.0.1-py3-none-any.whl")]
                command = self._cached_install(instance, [str(python), "-m", "pip", "install", "-r", dependencies[0], dependencies[1]])
                await self._run_step(instance, environment, "core_install", command, root, python, environment_root / "core-install.log", timeout=1800, writable_paths=[venv], network=True)
                entry = instance.get("inventory_entry") or {}
                preparation = entry.get("prepare_env") or {}
                requirements = entry.get("requirements_file")
                has_dependency_step = any(step.get("type") == "dependency" for step in preparation.get("steps", []))
                if requirements and not has_dependency_step:
                    await self._run_step(instance, environment, "example_install", self._cached_install(instance, [str(python), "-m", "pip", "install", "-r", str(root / requirements)]), root, python, environment_root / "example-install.log", timeout=1800, writable_paths=[venv], network=True)
                for index, step in enumerate(preparation.get("steps", [])):
                    working = (root / preparation["working_directory"]).resolve()
                    script = (working / step["script"]).resolve()
                    if not working.is_relative_to(root.resolve()) or not script.is_relative_to(root.resolve()):
                        raise ValueError("准备脚本超出当前实例源码目录")
                    arguments = step.get("args", [])
                    if not isinstance(arguments, list) or not all(isinstance(argument, str) for argument in arguments):
                        raise ValueError("准备脚本参数需要字符串列表")
                    executable = str(python) if script.suffix == ".py" else "sh"
                    execution_script = script
                    if script.suffix == ".sh" and b"\r\n" in script.read_bytes():
                        execution_script = environment_root / "prepared-scripts" / f"prepare-{index}.sh"
                        execution_script.parent.mkdir(parents=True, exist_ok=True)
                        approved_script = Path(plan["snapshot_root"]) / script.relative_to(root) if plan else script
                        execution_script.write_bytes(approved_script.read_bytes().replace(b"\r\n", b"\n"))
                        environment.setdefault("script_preparation", []).append({"source": str(script), "execution_copy": str(execution_script), "line_endings": "LF"})
                    await self._run_step(instance, environment, step.get("name") or f"prepare_{index}", [executable, str(execution_script), *arguments], working, python, environment_root / f"prepare-{index}.log", timeout=step.get("timeout", 900), writable_paths=[venv], readonly_paths=[execution_script], network=step.get("type") == "dependency")
                if environment.get("dependency_spec"):
                    specification = environment["dependency_spec"]
                    command = self._cached_install(instance, self._dependency_command(python, root, specification))
                    await self._run_step(instance, environment, "selected_dependencies", command, root, python, environment_root / "selected-dependencies.log", timeout=1800, writable_paths=[venv], network=True)
                package_log = environment_root / "packages.json"
                await self._run_step(instance, environment, "record_packages", [str(python), "-m", "pip", "--disable-pip-version-check", "list", "--format=json"], root, python, package_log)
                environment["packages"] = json.loads(package_log.read_text(encoding="utf-8"))
                check = await self.inspect(instance, environment=environment)
                environment["postcheck"] = check
                if not check["passed"]:
                    raise RuntimeError("准备后的依赖与模块检查失败，请查看正式环境检查报告")
                environment["status"] = "ready"
            else:
                environment["status"] = "created"
            environment["completed_at"] = utc_now()
            if plan:
                await asyncio.to_thread(self._validate_plan_inputs, instance, plan)
            if environment.get("dependency_spec"):
                environment["dependency_spec"]["file_versions"] = environment["postcheck"]["file_versions"]
                environment["dependency_fingerprint"] = self._dependency_fingerprint({**instance, "environment": environment})
            self._describe(instance, environment)
            environment["candidate"] = bool(not install and existing and existing.get("status") in {"ready", "modified"})
            self._save(instance, environment)
            if plan:
                plan.update(status="completed", completed_at=utc_now())
                self.store.write(plan["evidence_ref"], plan)
            self.events.emit("environment.completed", {"environment_id": environment_id, "status": environment["status"], "summary": environment["summary"], "warnings": environment["warnings"]}, instance_id=instance["instance_id"])
            return environment
        except (Exception, asyncio.CancelledError) as error:
            environment["status"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            environment["error"] = str(error) or "环境准备已取消"
            environment["completed_at"] = utc_now()
            self._describe(instance, environment)
            self._save(instance, environment)
            if plan:
                plan.update(status=environment["status"], error=environment["error"], completed_at=utc_now())
                self.store.write(plan["evidence_ref"], plan)
            self.events.emit("environment.failed", {"environment_id": environment_id, "error": str(error), "summary": environment["summary"], "failed_step": environment.get("failed_step"), "warnings": environment["warnings"]}, instance_id=instance["instance_id"])
            raise

    def _save(self, instance, environment, invalidate=True):
        active = instance.get("environment")
        retaining = environment.get("candidate") and active and active.get("status") in {"ready", "modified"} and active.get("environment_id") != environment["environment_id"]
        if retaining:
            if environment["status"] == "preparing":
                instance["pending_environment_id"] = environment["environment_id"]
            elif instance.get("pending_environment_id") == environment["environment_id"]:
                instance.pop("pending_environment_id", None)
            instance["last_environment_attempt"] = environment_summary(environment)
        else:
            instance["environment"] = environment
            if not environment.get("candidate") and (not active or active.get("environment_id") != environment["environment_id"] or instance.get("pending_environment_id") == environment["environment_id"]):
                instance.pop("pending_environment_id", None)
        if invalidate and not retaining:
            instance["status"] = "environment_ready" if environment["status"] == "ready" else "environment_" + environment["status"]
            instance["needs_verification"] = True
            if instance.get("verification"):
                instance["verification"]["stale"] = True
        self.store.write(f"environments/{environment['environment_id']}/environment.json", environment)
        self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
