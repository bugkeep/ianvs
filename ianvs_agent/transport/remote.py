# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
from collections import deque

from ianvs_agent.config import ModelProfile
from ianvs_agent.transport.protocol import MAX_FRAME_BYTES, MessageAssembler, RemoteCallError, TransportError, encode_message


class RemoteEvents:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)

    def publish(self, event):
        for callback in tuple(self.subscribers):
            callback(copy.deepcopy(event))


class CachedCatalog:
    def __init__(self, remote):
        self.remote = remote

    def list_examples(self):
        return copy.deepcopy(self.remote.cache.get("examples", []))


class CachedInstances:
    def __init__(self, remote):
        self.remote = remote

    def list_instances(self):
        return copy.deepcopy(self.remote.cache.get("instances", []))


class CachedStore:
    def __init__(self, remote):
        self.remote = remote

    def read_events(self, relative_path):
        expected = (self.remote.session_relative_path or "") + "/events.jsonl"
        if str(relative_path).replace("\\", "/") != expected:
            raise ValueError("前端只能读取当前会话的事件缓存")
        return copy.deepcopy(self.remote.cache.get("session_events", []))


class CachedRunner:
    def __init__(self, remote):
        self.remote = remote

    def get_run(self, instance_id, run_id):
        run = self.remote.cache.get("last_run")
        if not run or run.get("instance_id") != instance_id or run.get("run_id") != run_id:
            raise ValueError("当前前端缓存没有这条运行记录")
        return copy.deepcopy(run)


class RemoteHarness:
    ACTIONS = {"create_instance", "restore_instance", "delete_instance", "prepare_instance", "run_current", "send_message", "resume_current", "execute_tool", "switch_model", "compact"}

    def __init__(self, settings, command, backend_label="WSL"):
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise ValueError("后台启动命令必须为非空字符串参数列表")
        self.settings = settings
        self.command = list(command)
        self.backend_label = backend_label
        self.connected = False
        self.connection_error = None
        self.backend_info = {}
        self.events = RemoteEvents()
        self.catalog = CachedCatalog(self)
        self.instances = CachedInstances(self)
        self.store = CachedStore(self)
        self.runner = CachedRunner(self)
        self.cache = {"session": None, "current_instance": None, "task": {}, "state": {}, "busy": False, "history": [], "session_events": [], "diff": "", "usage_summary": {}, "verification_report": {}}
        self.process = None
        self.pending = {}
        self._ignored_ids = set()
        self._request_counter = 0
        self._revision = 0
        self._event_ids = set()
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._reader_task = None
        self._stderr_task = None
        self._drain_task = None
        self._ready = None
        self._closing = False
        self._stderr = deque(maxlen=200)

    @property
    def current_instance(self):
        return copy.deepcopy(self.cache.get("current_instance"))

    @property
    def session(self):
        return copy.deepcopy(self.cache.get("session"))

    @property
    def session_relative_path(self):
        return self.cache.get("session_relative_path")

    @property
    def task(self):
        return copy.deepcopy(self.cache.get("task", {}))

    @property
    def state(self):
        return copy.deepcopy(self.cache.get("state", {}))

    @property
    def busy(self):
        return bool(self.cache.get("busy")) or any(method in self.ACTIONS for _, method in self.pending.values())

    @property
    def backend_stderr(self):
        return "".join(self._stderr)

    async def start(self):
        async with self._start_lock:
            if self.connected:
                return self.backend_info
            if self.process is not None:
                await self._dispose_process()
            self.connection_error = None
            self._closing = False
            self._revision = 0
            self._stderr.clear()
            self._ready = asyncio.get_running_loop().create_future()
            try:
                self.process = await asyncio.create_subprocess_exec(*self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=MAX_FRAME_BYTES)
                self._reader_task = asyncio.create_task(self._read_frames())
                self._stderr_task = asyncio.create_task(self._read_stderr())
                async with asyncio.timeout(180):
                    await asyncio.shield(self._ready)
            except asyncio.TimeoutError as error:
                self._fail(TransportError("后台启动超过 180 秒，完成状态尚未确认"))
                self._ready.exception()
                await self._dispose_process()
                raise TransportError(self.connection_error) from error
            except asyncio.CancelledError:
                self._fail(TransportError("后台连接启动已取消"))
                self._ready.exception()
                await asyncio.shield(self._dispose_process())
                raise
            except Exception as error:
                if not self.connection_error:
                    detail = TransportError(f"无法启动后台 {self.command[0]}：{error}") if self.process is None else error
                    self._fail(detail)
                self._ready.exception()
                await self._dispose_process()
                raise
            return copy.deepcopy(self.backend_info)

    def _fail(self, error):
        self.connected = False
        self.connection_error = str(error)
        for future, _ in tuple(self.pending.values()):
            if not future.done():
                future.set_exception(TransportError(self.connection_error))
        self._ignored_ids.update(self.pending)
        self.pending.clear()
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(TransportError(self.connection_error))
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()

    def _update_settings(self, public):
        synchronized = {"selected_profile", "context_threshold", "context_input_budget", "keep_recent_turns", "max_model_calls", "max_tool_calls", "max_input_tokens", "max_output_tokens", "max_task_seconds", "max_repair_attempts", "tool_output_chars", "run_timeout_seconds", "pip_timeout_seconds", "pip_retries", "approval_policy", "dependency_mode"}
        for name, value in public.items():
            if name == "profiles":
                self.settings.profiles = {key: ModelProfile(**profile) for key, profile in value.items()}
            elif name in synchronized and hasattr(self.settings, name):
                setattr(self.settings, name, value)

    def _apply_snapshot(self, frame):
        revision, data = frame.get("revision"), frame.get("data")
        if type(revision) is not int or revision <= self._revision or not isinstance(data, dict) or type(frame.get("replace")) is not bool:
            raise TransportError("后台快照版本或内容无效")
        replacement = frame["replace"]
        delta = frame.get("history_delta")
        if replacement:
            if delta is not None or not isinstance(data.get("history"), list) or not isinstance(data.get("session_events"), list):
                raise TransportError("后台完整快照缺少对话或事件记录")
            if any(not isinstance(event, dict) or not isinstance(event.get("event_id"), str) for event in data["session_events"]):
                raise TransportError("后台完整快照的事件记录无效")
            updated = data
        else:
            updated = {**self.cache, **data}
            if delta is not None:
                previous = self.cache.get("history", [])
                if not isinstance(delta, dict) or set(delta) != {"start", "items"} or type(delta["start"]) is not int or not 0 <= delta["start"] <= len(previous) or not isinstance(delta["items"], list):
                    raise TransportError("后台对话增量的起始位置或内容无效")
                updated["history"] = previous[:delta["start"]] + delta["items"]
        self._update_settings(data.get("settings", {}))
        self.cache = updated
        if replacement:
            self._event_ids = {event["event_id"] for event in data["session_events"]}
        self._revision = revision
        self.backend_info = copy.deepcopy(updated.get("backend_info", self.backend_info))

    async def _read_frames(self):
        assembler = MessageAssembler()
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    if assembler.message_id is not None:
                        raise TransportError("后台连接在完整消息接收完成之前关闭")
                    if not self._closing:
                        await self.process.wait()
                        await self._stderr_task
                        raise TransportError(f"后台连接已经中断（退出码 {self.process.returncode}）\n{self.backend_stderr[-4000:]}")
                    if any(not future.done() for future, _ in self.pending.values()):
                        self._fail(TransportError("后台连接在操作完成确认之前关闭"))
                    return
                frame = await asyncio.to_thread(assembler.feed, line) if len(line) > 65536 else assembler.feed(line)
                if frame is None:
                    continue
                kind = frame["type"]
                if kind == "snapshot":
                    self._apply_snapshot(frame)
                elif kind == "ready":
                    if self._revision < 1 or self.connected:
                        raise TransportError("后台启动确认缺少有效快照或重复发送")
                    self.backend_info = copy.deepcopy(frame["backend_info"])
                    self.connected = True
                    self._ready.set_result(True)
                elif kind == "event":
                    event = frame["event"]
                    if not isinstance(event, dict) or "event_type" not in event:
                        raise TransportError("后台事件结构无效")
                    session = self.cache.get("session")
                    if session and event.get("session_id") == session["session_id"]:
                        cached = self.cache.setdefault("session_events", [])
                        if event.get("event_id") not in self._event_ids:
                            cached.append(event)
                            self._event_ids.add(event.get("event_id"))
                    self.events.publish(event)
                elif kind == "response":
                    request_id = frame.get("id")
                    entry = self.pending.get(request_id)
                    if entry is None:
                        if request_id in self._ignored_ids:
                            self._ignored_ids.discard(request_id)
                            continue
                        raise TransportError("后台响应没有对应的请求")
                    future = entry[0]
                    if future.done():
                        raise TransportError("后台重复完成同一请求")
                    error = frame.get("error")
                    if error:
                        if error.get("type") == "CancelledError":
                            future.set_exception(asyncio.CancelledError(error.get("message")))
                        else:
                            future.set_exception(RemoteCallError(error))
                    else:
                        future.set_result(frame.get("result"))
                else:
                    raise TransportError("前端收到无效方向的后台消息")
        except Exception as error:
            self._fail(error)
            self._drain_task = asyncio.create_task(self.process.stdout.read())

    async def _read_stderr(self):
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    return
                self._stderr.append(line.decode("utf-8"))
        except Exception as error:
            self._fail(error)
            raise

    async def _write_request(self, frame):
        async with self._write_lock:
            payloads = await asyncio.to_thread(lambda: list(encode_message(frame)))
            for payload in payloads:
                self.process.stdin.write(payload)
                await self.process.stdin.drain()

    async def _send_request(self, frame):
        sending = asyncio.create_task(self._write_request(frame))
        cancelled = False
        while True:
            try:
                await asyncio.shield(sending)
                break
            except asyncio.CancelledError:
                if sending.cancelled():
                    raise
                cancelled = True
        # 发送完当前消息后，才允许取消请求进入同一 stdio 通道。
        if cancelled:
            raise asyncio.CancelledError

    async def _rpc(self, method, params):
        if not self.connected:
            raise TransportError(self.connection_error or "后台尚未连接")
        self._request_counter += 1
        request_id = f"RPC-{self._request_counter:06d}"
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = (future, method)
        try:
            await self._send_request({"type": "request", "id": request_id, "method": method, "params": params})
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if not future.done() and method in self.ACTIONS and self.connected and not self._closing:
                await asyncio.shield(self.cancel())
            raise
        except (BrokenPipeError, ConnectionResetError) as error:
            self._fail(error)
            if future.done() and not future.cancelled():
                future.exception()
            raise TransportError(self.connection_error) from error
        finally:
            self.pending.pop(request_id, None)
            if not future.done():
                self._ignored_ids.add(request_id)
                future.cancel()
            elif not future.cancelled():
                future.exception()

    async def create_instance(self, benchmark_path):
        return await self._rpc("create_instance", {"benchmark_path": str(benchmark_path)})

    async def restore_instance(self, instance_id):
        return await self._rpc("restore_instance", {"instance_id": instance_id})

    async def delete_instance(self, instance_id):
        return await self._rpc("delete_instance", {"instance_id": instance_id})

    async def prepare_instance(self):
        return await self._rpc("prepare_instance", {})

    async def run_current(self):
        return await self._rpc("run_current", {})

    async def send_message(self, text):
        return await self._rpc("send_message", {"text": text})

    async def resume_current(self):
        return await self._rpc("resume_current", {})

    async def compact(self):
        return await self._rpc("compact", {})

    async def approve(self, approval_id, approved):
        return await self._rpc("approve", {"approval_id": approval_id, "approved": approved})

    async def switch_model(self, profile_name):
        return await self._rpc("switch_model", {"profile_name": profile_name})

    async def execute_tool(self, call):
        return await self._rpc("execute_tool", {"call": call})

    async def refresh(self):
        return await self._rpc("refresh", {})

    async def cancel(self):
        active = [future for future, method in self.pending.values() if method in self.ACTIONS and not future.done()]
        result = await self._rpc("cancel", {})
        results = await asyncio.gather(*(asyncio.shield(future) for future in active), return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                raise item
        return result

    def history(self):
        return copy.deepcopy(self.cache.get("history", []))

    def diff(self):
        return self.cache.get("diff", "")

    def usage_summary(self):
        return copy.deepcopy(self.cache.get("usage_summary", {}))

    def verification_report(self):
        return copy.deepcopy(self.cache.get("verification_report", {}))

    async def _dispose_process(self):
        if self.process is None:
            return
        process = self.process
        self._closing = True
        if process.stdin:
            process.stdin.close()
        drain = self._drain_task
        if drain is None and (self._reader_task is None or self._reader_task.done()):
            drain = asyncio.create_task(process.stdout.read())
        shutdown_error = None
        try:
            async with asyncio.timeout(30):
                await process.wait()
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.terminate()
            await process.wait()
            shutdown_error = TransportError("后台关闭尚未确认：启动进程已经终止，WSL 内的执行状态需要重新连接后核查")
            self.connection_error = str(shutdown_error)
        for task in (self._reader_task, self._stderr_task, drain):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (self._reader_task, self._stderr_task, drain) if task is not None), return_exceptions=True)
        self.connected = False
        self.process = None
        self._reader_task = self._stderr_task = None
        self._drain_task = None
        if shutdown_error is not None:
            raise shutdown_error

    async def close(self):
        if self.process is None:
            return
        if self.process.returncode is not None or self.process.stdin.is_closing():
            await self._dispose_process()
            return
        self._closing = True
        try:
            if self.connected:
                await self.cancel()
                await self._rpc("close", {})
        finally:
            await self._dispose_process()
