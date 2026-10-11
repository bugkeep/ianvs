# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import codecs
import copy
import json
import os
import shutil
import traceback
from pathlib import Path

import psutil
from .instances import configuration_evidence, file_manifest, utc_now, validate_identity
from .original_goal import OriginalGoalStore
from .processes import _terminate_owned, runtime_environment, spawn, terminate_process
from .credentials import SecretRedactor, credential_status, resolve_credentials


class Runner:
    def __init__(self, settings, store, events):
        self.settings, self.store, self.events = settings, store, events
        self._process = None
        self._current = None
        self._cancelled = False
        self._orphaned = []
        self.recover()

    def _relative(self, instance_id, run_id):
        validate_identity(instance_id, "实例")
        validate_identity(run_id, "运行")
        return f"instances/{instance_id}/runs/{run_id}"

    def _save(self, run):
        self.store.write(f"{self._relative(run['instance_id'], run['run_id'])}/run.json", run)

    def _invalidate_verification(self, instance, run_id, status):
        previous = instance.get("verification")
        if previous:
            instance["verification"] = copy.deepcopy(previous)
            instance["verification"].update(stale=True, superseded_by_run_id=run_id, stale_reason="新的 Run 尚未完成本轮独立验证")
        instance.update(last_run_id=run_id, needs_verification=True, status=status)
        self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)

    def get_run(self, instance_id, run_id):
        run = self.store.read(f"{self._relative(instance_id, run_id)}/run.json")
        if run is None:
            raise KeyError("运行记录不存在")
        if run.get("instance_id") != instance_id or run.get("run_id") != run_id:
            raise ValueError("运行记录身份与请求不一致")
        return run

    @staticmethod
    def _owned_process(run):
        pid = run.get("pid")
        if not pid or not psutil.pid_exists(pid):
            return None
        try:
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE or process.create_time() != run.get("process_created_at"):
                return None
            if process.environ().get("IANVS_RUN_ID") != run["run_id"]:
                return None
            return process
        except psutil.NoSuchProcess:
            return None

    def recover(self):
        self._orphaned = []
        for path in self.settings.state_root.glob("instances/*/runs/*/run.json"):
            run = self.store.read(path.relative_to(self.settings.state_root).as_posix())
            if run["status"] not in {"running", "orphaned_running"}:
                continue
            process = self._owned_process(run)
            if process:
                run["status"] = "orphaned_running"
                self._orphaned.append(run)
            else:
                run.update(status="interrupted", interrupted=True, completed_at=utc_now(), recovery_reason="重启核查确认原运行进程已结束；退出状态未知")
            self._save(run)
            instance = self.store.read(f"instances/{run['instance_id']}/instance.json")
            if instance:
                self._invalidate_verification(instance, run["run_id"], "needs_verification")
        return list(self._orphaned)

    async def run(self, instance, overrides=None, *, original_goal=None):
        if self._current is None and self._orphaned:
            self.recover()
        if self._current is not None or self._orphaned:
            raise RuntimeError("已有运行或重启前仍活动的所属进程，请先取消或等待结束")
        overrides = overrides or {}
        if set(overrides) - {"timeout_seconds"}:
            raise ValueError("运行覆盖参数仅允许调整超时")
        run_id = self.store.allocate_id("run")
        relative = self._relative(instance["instance_id"], run_id)
        run_root = self.settings.state_root / relative
        run_root.mkdir(parents=True, exist_ok=False)
        environment = instance.get("environment") or {}
        python = environment.get("python_executable")
        cwd = Path(instance["code_root"])
        event_file = run_root / "core-events.jsonl"
        command = [python, "-m", "core.cmd.benchmarking", "-f", instance["benchmark_path"]] if python else []
        run = {"run_id": run_id, "instance_id": instance["instance_id"], "environment_id": environment.get("environment_id"), "status": "running", "exit_code": None, "timed_out": False, "interrupted": False, "started_at": utc_now(), "completed_at": None, "command": command, "cwd": str(cwd), "output_dir": str(run_root / "output"), "stdout_ref": f"{relative}/stdout.log", "stderr_ref": f"{relative}/stderr.log", "events": [], "testcases": [], "metrics": {}, "artifacts": [], "config_snapshot": {}, "source_manifest": None, "pid": None, "process_created_at": None}
        self._current = run
        run["environment_snapshot"] = copy.deepcopy(environment)
        self._cancelled = False
        self._save(run)
        self._invalidate_verification(instance, run_id, "running")
        self.events.emit("run.started", {"command": command, "cwd": str(cwd)}, instance_id=instance["instance_id"], run_id=run_id)
        readers = []
        credentials = {}
        event_pipe = None
        event_pipe_path = None
        temporary = run_root / "temporary"
        try:
            goals = OriginalGoalStore(self.settings, self.store)
            if original_goal is None:
                await asyncio.to_thread(goals.resolve, instance)
            else:
                await asyncio.to_thread(goals.load, instance)
                if original_goal.get("evidence_ref") != instance.get("original_goal_ref") or any(original_goal.get(key) != instance.get(key) for key in ("goal_id", "goal_digest")):
                    raise ValueError("任务绑定的原目标版本与实例正式记录不一致")
            run.update({key: instance[key] for key in ("original_goal_ref", "goal_id", "goal_digest")})
            self._save(run)
            if environment.get("status") not in {"ready", "modified"} or not python or not Path(python).is_file():
                raise RuntimeError("实例环境尚未准备完成")
            run["credential_status"] = credential_status(self.settings, instance)
            process_env, credentials = self._execution_environment(instance, python)
            if credentials and os.name != "posix":
                raise RuntimeError("使用运行凭据需要 Linux/WSL 执行后端")
            run["source_manifest"] = await asyncio.to_thread(file_manifest, cwd)
            if self._cancelled:
                raise asyncio.CancelledError
            self.events.emit("run.stage_changed", {"stage": "configuration_capture"}, instance_id=instance["instance_id"], run_id=run_id)
            run["config_snapshot"] = await asyncio.to_thread(configuration_evidence, cwd, instance["benchmark_path"])
            if self._cancelled:
                raise asyncio.CancelledError
            self._save(run)
            run["execution_configuration"] = {"entrypoint": "core.cmd.benchmarking", "mock_runtime_enabled": "IANVS_LLM_MOCK" in process_env, "pythonpath_override": "PYTHONPATH" in process_env, "evidence_type": "runner_launch_environment", "environment_status": environment.get("status"), "diagnostic_environment": environment.get("status") == "modified", "credential_names": sorted(credentials)}
            temporary.mkdir(parents=True, exist_ok=True)
            event_target = event_file
            if credentials:
                # 旧实例也经由内存管道传递事件，正式文件只接收脱敏内容。
                event_pipe_path = temporary / "core-events.pipe"
                os.mkfifo(event_pipe_path, 0o600)
                event_pipe = os.open(event_pipe_path, os.O_RDWR | os.O_NONBLOCK)
                event_target = event_pipe_path
            process_env.update(IANVS_RUN_ID=run_id, IANVS_EVENT_FILE=str(event_target), IANVS_REDACT_ENV_NAMES=json.dumps(sorted(credentials)), IANVS_RUN_WORKSPACE=run["output_dir"], TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary))
            self._process = await spawn(command, cwd, process_env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            run["pid"] = self._process.pid
            try:
                run["process_created_at"] = psutil.Process(self._process.pid).create_time()
            except psutil.NoSuchProcess:
                # 很快退出的子进程仍需读取实际退出码与管道日志。
                run["process_created_at"] = None
            self._save(run)
            if self._cancelled:
                raise asyncio.CancelledError
            readers = [asyncio.create_task(self._stream(self._process.stdout, run_root / "stdout.log", "stdout", run, credentials.values())), asyncio.create_task(self._stream(self._process.stderr, run_root / "stderr.log", "stderr", run, credentials.values())), asyncio.create_task(self._read_core_events(event_file, run, credentials.values(), event_pipe))]
            timeout = float(overrides.get("timeout_seconds", self.settings.run_timeout_seconds))
            try:
                await asyncio.wait_for(self._process.wait(), timeout)
            except asyncio.TimeoutError:
                run["timed_out"] = True
                await terminate_process(self._process)
            run["exit_code"] = self._process.returncode
            run["interrupted"] = self._cancelled
            run["status"] = "cancelled" if self._cancelled else "timed_out" if run["timed_out"] else "completed" if run["exit_code"] == 0 else "failed"
            await asyncio.gather(*readers)
            readers = []
        except asyncio.CancelledError:
            if self._process:
                await terminate_process(self._process)
                run["exit_code"] = self._process.returncode
            run.update(status="cancelled", interrupted=True)
            if readers:
                cancelled_readers = readers
                readers = []
                results = await asyncio.gather(*cancelled_readers, return_exceptions=True)
                reader_error = next((result for result in results if isinstance(result, Exception)), None)
                if reader_error is not None:
                    run.update(status="failed", error=self._redact_text(str(reader_error), credentials.values()), error_type=type(reader_error).__name__)
                    with (run_root / "stderr.log").open("a", encoding="utf-8") as output:
                        output.write(self._redact_text("".join(traceback.format_exception(reader_error)), credentials.values()))
                    raise reader_error
        except Exception as error:
            if self._process and self._process.returncode is None:
                await terminate_process(self._process)
                run["exit_code"] = self._process.returncode
            run.update(status="failed", error=self._redact_text(str(error), credentials.values()), error_type=type(error).__name__)
            with (run_root / "stderr.log").open("a", encoding="utf-8") as output:
                output.write(self._redact_text(traceback.format_exc(), credentials.values()))
            if readers:
                for reader in readers:
                    reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
                readers = []
            if run["error"] != str(error):
                raise RuntimeError(run["error"]) from None
            raise
        finally:
            remaining_readers = readers
            for reader in remaining_readers:
                reader.cancel()
            if remaining_readers:
                await asyncio.gather(*remaining_readers, return_exceptions=True)
            run["completed_at"] = utc_now()
            if event_pipe is not None:
                os.close(event_pipe)
                event_pipe_path.unlink(missing_ok=True)
            try:
                if credentials:
                    restricted = await asyncio.to_thread(self._quarantine_artifacts, run["output_dir"], credentials.values())
                    if temporary.is_dir():
                        restricted.extend(await asyncio.to_thread(self._quarantine_artifacts, temporary, credentials.values()))
                    if restricted:
                        run.update(status="failed", error="运行产物包含凭据内容，已隔离并阻止作为评测结果发布", restricted_artifacts=restricted)
                run["artifacts"] = [str(path) for path in Path(run["output_dir"]).rglob("*") if path.is_file()] if Path(run["output_dir"]).exists() else []
            except Exception as error:
                run.update(status="failed", artifacts=[], error="运行产物检查失败：" + self._redact_text(str(error), credentials.values()), error_type=type(error).__name__)
                raise RuntimeError(run["error"]) from None
            finally:
                try:
                    self._save(run)
                    self._invalidate_verification(instance, run_id, "needs_verification")
                    self.events.emit("run.completed" if run["status"] == "completed" else "run.failed", {"status": run["status"], "exit_code": run["exit_code"], "error": run.get("error"), "stdout_ref": run["stdout_ref"], "stderr_ref": run["stderr_ref"]}, instance_id=instance["instance_id"], run_id=run_id)
                finally:
                    self._process = None
                    self._current = None
        return run

    @staticmethod
    def _redact_text(text, secrets):
        return SecretRedactor(secrets).feed(text.encode("utf-8"), final=True).decode("utf-8", errors="replace")

    def _execution_environment(self, instance, python):
        credentials = resolve_credentials(self.settings, instance)
        environment = runtime_environment(python)
        protected = {*self.settings.runtime_credentials, *self.settings.runtime_credentials.values(), *(profile.api_key_env for profile in self.settings.profiles.values())}
        for name in protected:
            environment.pop(name, None)
        environment.update(credentials)
        return environment, credentials

    def _redact_value(self, value, secrets):
        if isinstance(value, str):
            return self._redact_text(value, secrets)
        if isinstance(value, list):
            return [self._redact_value(item, secrets) for item in value]
        if isinstance(value, dict):
            return {self._redact_value(key, secrets): self._redact_value(item, secrets) for key, item in value.items()}
        return value

    def _quarantine_artifacts(self, output_dir, secrets):
        root = Path(output_dir)
        secrets = tuple(secrets)
        scanner = SecretRedactor(secrets)
        if not root.exists() or scanner.pattern is None:
            return []
        restricted = []
        for path in root.rglob("*"):
            if not path.is_file() and not path.is_symlink():
                continue
            sensitive = path.is_symlink() or bool(scanner.pattern.search(str(path).encode("utf-8")))
            if not sensitive:
                tail = b""
                with path.open("rb") as stream:
                    while chunk := stream.read(65536):
                        combined = tail + chunk
                        if scanner.pattern.search(combined):
                            sensitive = True
                            break
                        tail = combined[-scanner.keep:] if scanner.keep else b""
            if sensitive:
                identity = self.store.allocate_id("restricted-artifact")
                reference = "quarantine/" + identity
                destination = self.store.path(reference)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), destination)
                restricted.append({"original_name": self._redact_text(str(path.relative_to(root)), secrets), "private_ref": reference, "reason": "credential_content_or_unsafe_link"})
        return restricted

    async def _stream(self, stream, path, channel, run, secrets=()):
        redactor = SecretRedactor(secrets)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with path.open("wb") as output:
            try:
                while chunk := await stream.read(8192):
                    safe = redactor.feed(chunk)
                    output.write(safe)
                    output.flush()
                    text = decoder.decode(safe)
                    if text:
                        self.events.emit("run.output", {"channel": channel, "text": text}, instance_id=run["instance_id"], run_id=run["run_id"])
            finally:
                safe = redactor.feed(b"", final=True)
                output.write(safe)
                output.flush()
                text = decoder.decode(safe, final=True)
                if text:
                    self.events.emit("run.output", {"channel": channel, "text": text}, instance_id=run["instance_id"], run_id=run["run_id"])

    def _core_event(self, line, run):
        event = json.loads(line)
        if event.get("run_id") != run["run_id"]:
            raise ValueError("阶段事件的 Run 标识不一致")
        run["events"].append(event)
        payload = event["payload"]
        if event["event_type"] == "testcases.built":
            run["testcases"] = payload["testcases"]
        elif event["event_type"] == "testcase.completed":
            run["metrics"][payload["testcase_id"]] = payload["metrics"]
        self.events.emit(event["event_type"], payload, instance_id=run["instance_id"], run_id=run["run_id"])
        self._save(run)

    async def _read_core_events(self, path, run, secrets=(), pipe=None):
        position = 0
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")()
        secrets = tuple(secrets)
        while True:
            finished = self._process is None or self._process.returncode is not None
            if pipe is not None:
                try:
                    chunk = os.read(pipe, 65536)
                except BlockingIOError:
                    chunk = b""
                pending += decoder.decode(chunk)
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    event = self._redact_value(json.loads(line), secrets)
                    safe = json.dumps(event, ensure_ascii=False) + "\n"
                    with path.open("a", encoding="utf-8") as output:
                        output.write(safe)
                    self._core_event(safe, run)
                if chunk:
                    continue
            elif path.exists():
                with path.open("r", encoding="utf-8") as source:
                    source.seek(position)
                    while True:
                        line = source.readline()
                        if not line or not line.endswith("\n"):
                            break
                        position = source.tell()
                        self._core_event(line, run)
            if finished:
                if pipe is not None:
                    pending += decoder.decode(b"", final=True)
                    if pending:
                        run["core_events_partial"] = True
                return
            await asyncio.sleep(0.1)

    async def cancel(self):
        self._cancelled = True
        if self._process:
            await terminate_process(self._process)
            return
        for run in self._orphaned:
            process = self._owned_process(run)
            if process:
                try:
                    owned = [*process.children(recursive=True), process]
                except psutil.NoSuchProcess:
                    owned = []
                await _terminate_owned(owned)
            run.update(status="interrupted", interrupted=True, completed_at=utc_now(), recovery_reason="用户取消重启前仍活动的所属进程")
            self._save(run)
            instance = self.store.read(f"instances/{run['instance_id']}/instance.json")
            if instance:
                self._invalidate_verification(instance, run["run_id"], "needs_verification")
        self._orphaned = []
