# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import math
import os
import signal
import subprocess
from pathlib import Path

import psutil

from .sandbox import sandbox_command, sandbox_network


def runtime_environment(python=None):
    environment = dict(os.environ)
    for name in list(environment):
        if name.endswith(("_API_KEY", "_TOKEN", "_PASSWORD")) or name in {"PYTHONPATH", "PYTHONHOME", "IANVS_LLM_MOCK", "IANVS_EVENT_FILE", "IANVS_RUN_WORKSPACE", "IANVS_RUN_ID"}:
            environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if python:
        environment["PYTHON"] = str(python)
        environment["PATH"] = str(Path(python).parent) + os.pathsep + environment.get("PATH", "")
    return environment


async def spawn(command, cwd, environment, **kwargs):
    options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    return await asyncio.create_subprocess_exec(*command, cwd=str(cwd), env=environment, **options, **kwargs)


async def terminate_process(process):
    try:
        parent = psutil.Process(process.pid)
        owned = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        owned = []
    await _terminate_owned(owned)
    await asyncio.wait_for(process.wait(), timeout=5)


async def _terminate_owned(owned):
    groups = set()
    if os.name == "posix":
        for item in tuple(owned):
            try:
                if item.is_running() and os.getpgid(item.pid) == item.pid:
                    groups.add(item.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
        identities = {item.pid for item in owned}
        for item in psutil.process_iter():
            try:
                if item.pid not in identities and os.getpgid(item.pid) in groups:
                    item.create_time()
                    owned.append(item)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
        for group in groups:
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                continue
    for item in owned:
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = await asyncio.to_thread(psutil.wait_procs, owned, timeout=5)
    if os.name == "posix":
        for item in alive:
            try:
                group = os.getpgid(item.pid)
                if item.is_running() and group in groups:
                    os.killpg(group, signal.SIGKILL)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
    for item in alive:
        try:
            if item.status() != psutil.STATUS_ZOMBIE:
                item.kill()
        except psutil.NoSuchProcess:
            continue
    _, alive = await asyncio.to_thread(psutil.wait_procs, alive, timeout=5)
    survivors = []
    for item in alive:
        try:
            if item.is_running() and item.status() != psutil.STATUS_ZOMBIE:
                survivors.append(item.pid)
        except psutil.NoSuchProcess:
            continue
    if survivors:
        raise RuntimeError(f"取消后仍存在所属环境进程：{survivors}")


def recorded_process(operation):
    if not operation.get("pid") or not operation.get("operation_id") or not psutil.pid_exists(operation["pid"]):
        return None
    process = psutil.Process(operation["pid"])
    try:
        if process.status() == psutil.STATUS_ZOMBIE or process.create_time() != operation.get("process_created_at"):
            return None
        if process.environ().get("IANVS_OPERATION_ID") != operation["operation_id"]:
            return None
    except psutil.NoSuchProcess:
        return None
    return process


async def terminate_recorded_operation(operation):
    process = recorded_process(operation)
    if process is None:
        return False
    await _terminate_owned([*process.children(recursive=True), process])
    return True


async def recorded_command(command, cwd, python, log_path, timeout=900, *, check=True):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = log_path.parent / "temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment(python)
    environment.update(TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary))
    with log_path.open("wb") as output:
        process = await spawn(command, cwd, environment, stdout=output, stderr=asyncio.subprocess.STDOUT)
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            await terminate_process(process)
            raise
    if check and process.returncode != 0:
        raise RuntimeError(f"命令执行失败，退出状态 {process.returncode}，完整日志：{log_path}")
    return {"command": list(command), "exit_code": process.returncode, "log_ref": str(log_path)}


async def sandboxed_command(command, cwd, python, log_path, timeout=900, *, writable_paths=(), readonly_paths=(), network=False, operation=None, on_update=None, pip_timeout_seconds=60, pip_retries=5):
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("环境命令时限必须为正数")
    if isinstance(pip_timeout_seconds, bool) or not isinstance(pip_timeout_seconds, (int, float)) or not math.isfinite(pip_timeout_seconds) or pip_timeout_seconds <= 0:
        raise ValueError("pip 网络请求时限必须为有限正数")
    if isinstance(pip_retries, bool) or not isinstance(pip_retries, int) or pip_retries < 0:
        raise ValueError("pip 网络重试次数必须为非负整数")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = log_path.parent / "temporary"
    temporary.mkdir(exist_ok=True)
    result = operation if operation is not None else {}
    result.update(command=list(command), cwd=str(cwd), log_ref=str(log_path), phase="preflight", command_executed=False, status="starting", exit_code=None)

    async def update():
        if on_update:
            response = on_update(result)
            if inspect.isawaitable(response):
                await response

    await update()
    with sandbox_network(network) as (proxy_fd, proxy_names, redactor):
        try:
            execution, isolation = await asyncio.to_thread(sandbox_command, list(command), cwd, python, temporary, writable_paths=writable_paths, readonly_paths=readonly_paths, readonly_mappings=result.get("readonly_mappings", ()), network=network, proxy_fd=proxy_fd, proxy_names=proxy_names, pip_timeout_seconds=pip_timeout_seconds, pip_retries=pip_retries)
        except (Exception, asyncio.CancelledError) as error:
            safe_error = redactor.feed(str(error).encode("utf-8"), final=True).decode("utf-8", errors="replace")
            log_path.write_text(f"环境执行预检失败：{type(error).__name__}: {safe_error}\n", encoding="utf-8")
            result.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=safe_error or "预检已取消")
            await update()
            raise
        result.update(isolation=isolation, phase="spawn")
        environment = runtime_environment(python)
        if result.get("operation_id"):
            environment["IANVS_OPERATION_ID"] = result["operation_id"]
        try:
            process = await spawn(execution, cwd, environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, pass_fds=(proxy_fd,) if proxy_fd is not None else ())
        except (Exception, asyncio.CancelledError) as error:
            safe_error = redactor.feed(str(error).encode("utf-8"), final=True).decode("utf-8", errors="replace")
            log_path.write_text(f"环境进程启动失败：{type(error).__name__}: {safe_error}\n", encoding="utf-8")
            result.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed", error=safe_error or "启动已取消")
            await update()
            raise
    try:
        created = psutil.Process(process.pid).create_time()
    except psutil.NoSuchProcess:
        created = None
    result.update(pid=process.pid, process_created_at=created, phase="execution", command_executed=True, status="running")

    async def stream():
        with log_path.open("wb") as output:
            while chunk := await process.stdout.read(8192):
                output.write(redactor.feed(chunk))
                output.flush()
            output.write(redactor.feed(b"", final=True))
            output.flush()

    reader = asyncio.create_task(stream())
    try:
        await update()
        waiter = asyncio.create_task(process.wait())
        completed, _ = await asyncio.wait([waiter, reader], timeout=timeout, return_when=asyncio.FIRST_EXCEPTION)
        if not completed:
            raise asyncio.TimeoutError
        if reader in completed and reader.exception() is not None:
            raise reader.exception()
        if not waiter.done():
            raise asyncio.TimeoutError
        await waiter
        await reader
        result.update(exit_code=process.returncode, status="completed" if process.returncode == 0 else "failed")
        await update()
        if process.returncode != 0:
            raise RuntimeError(f"环境命令执行失败，退出状态 {process.returncode}，完整日志：{log_path}")
        return result
    except (asyncio.TimeoutError, asyncio.CancelledError) as error:
        await terminate_process(process)
        await reader
        result.update(exit_code=process.returncode, status="cancelled" if isinstance(error, asyncio.CancelledError) else "timed_out")
        await update()
        raise
    except Exception:
        if process.returncode is None:
            await terminate_process(process)
        if not reader.done():
            await reader
        result.update(exit_code=process.returncode, status="failed")
        await update()
        raise
    finally:
        if "waiter" in locals() and not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
