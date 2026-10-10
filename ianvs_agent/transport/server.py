# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import copy
import os
import platform
import sys
import threading
import traceback
from pathlib import Path

from filelock import FileLock

from ianvs_agent.transport.protocol import MAX_FRAME_BYTES, MessageAssembler, TransportError, encode_message


RPC_OUTPUT = sys.stdout.buffer
sys.stdout = sys.stderr

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import load_settings


class BackendService:
    def __init__(self, settings):
        self.settings = settings
        self.harness = AgentHarness(settings)
        self.loop = asyncio.get_running_loop()
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        self.stopping = asyncio.Event()
        self.snapshot_lock = asyncio.Lock()
        self.requests = {}
        self.revision = 0
        self.snapshot_session = None
        self.published_data = {}
        self.examples = None
        self.publisher_task = None
        self.fatal_error = None
        self.backend_info = {
            "os": os.name,
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
            "pid": os.getpid(),
            "project_root": str(settings.project_root),
            "state_root": str(settings.state_root),
        }
        self.methods = {
            "create_instance": ({"benchmark_path"}, self.harness.create_instance),
            "restore_instance": ({"instance_id"}, self.harness.restore_instance),
            "prepare_instance": (set(), self.harness.prepare_instance),
            "run_current": (set(), self.harness.run_current),
            "send_message": ({"text"}, self.harness.send_message),
            "resume_current": (set(), self.harness.resume_current),
            "compact": (set(), self.harness.compact),
            "approve": ({"approval_id", "approved"}, self.harness.approve),
            "cancel": (set(), self.harness.cancel),
            "switch_model": ({"profile_name"}, self.switch_model),
            "execute_tool": ({"call"}, self.execute_tool),
            "refresh": (set(), self.refresh),
            "close": (set(), self.harness.close),
        }
        self.unsubscribe = self.harness.events.subscribe(self.receive_event)

    def receive_event(self, event):
        self.loop.call_soon_threadsafe(self.event_queue.put_nowait, copy.deepcopy(event))

    def read_input(self):
        try:
            while True:
                line = sys.stdin.buffer.readline(MAX_FRAME_BYTES + 1)
                self.loop.call_soon_threadsafe(self.incoming.put_nowait, line)
                if not line:
                    return
        except Exception as error:
            self.loop.call_soon_threadsafe(self.incoming.put_nowait, error)

    async def send(self, frame):
        if self.fatal_error is not None:
            raise self.fatal_error
        payloads = await asyncio.to_thread(lambda: list(encode_message(frame)))
        if self.fatal_error is not None:
            raise self.fatal_error
        # 同一完整消息的全部分片连续入队，保持 stdio 中的消息顺序。
        for payload in payloads:
            self.outgoing.put_nowait(payload)

    async def write_output(self):
        while True:
            payload = await self.outgoing.get()
            try:
                await asyncio.to_thread(self.write_bytes, payload)
            except Exception as error:
                self.fatal_error = error
                self.stopping.set()
                raise
            finally:
                self.outgoing.task_done()

    @staticmethod
    def write_bytes(payload):
        RPC_OUTPUT.write(payload)
        RPC_OUTPUT.flush()

    def read_snapshot(self, full, session_path, instance_id, usage, busy, public_settings, history, diff, instances):
        data = {"busy": busy, "settings": public_settings, "backend_info": self.backend_info}
        instance = self.harness.instances.get(instance_id) if instance_id else None
        data["current_instance"] = self.public_instance(instance)
        data["session_relative_path"] = session_path
        saved = self.harness.store.read(session_path + "/session_snapshot.json", {}) if session_path else {}
        data.update({"session": saved.get("session"), "task": saved.get("task", {}), "state": saved.get("state", {})})
        data["usage_summary"] = usage.summary() if usage else self.harness.usage_summary()
        data["verification_report"] = (instance.get("verification") or {}) if instance else {}
        run_id = instance.get("last_run_id") if instance else None
        data["last_run"] = self.harness.run_summary(self.harness.runner.get_run(instance_id, run_id)) if run_id else None
        if full:
            if self.examples is None:
                self.examples = self.harness.catalog.list_examples()
            data["examples"] = self.examples
            data["session_events"] = self.harness.store.read_events(session_path + "/events.jsonl") if session_path else []
        if full or instances:
            data["instances"] = [self.public_instance(item) for item in self.harness.instances.list_instances()]
        if full or history:
            data["history"] = saved.get("history", [])
        if full or diff:
            data["diff"] = self.harness.diff()
        return data

    @staticmethod
    def public_instance(instance):
        if instance is None:
            return None
        # 文件完整性清单由后端正式记录管理，界面只接收实例概要。
        return {key: value for key, value in instance.items() if key not in {"source_manifest", "current_manifest"}}

    async def publish_snapshot(self, full=False, history=False, diff=False, instances=False):
        async with self.snapshot_lock:
            session_path = self.harness.session_relative_path
            full = full or self.revision == 0 or session_path != self.snapshot_session
            instance_id = self.harness.current_instance["instance_id"] if self.harness.current_instance else None
            data = await asyncio.to_thread(self.read_snapshot, full, session_path, instance_id, self.harness.usage, self.harness.busy, self.settings.public_dict(), history, diff, instances)
            frame = {"type": "snapshot", "replace": full}
            if full:
                changes = data
            else:
                changes = {key: value for key, value in data.items() if key != "history" and value != self.published_data.get(key)}
                if "history" in data and data["history"] != self.published_data.get("history", []):
                    previous = self.published_data.get("history", [])
                    updated = data["history"]
                    start = 0
                    while start < min(len(previous), len(updated)) and previous[start] == updated[start]:
                        start += 1
                    frame["history_delta"] = {"start": start, "items": updated[start:]}
                if not changes and "history_delta" not in frame:
                    return
            self.revision += 1
            self.snapshot_session = session_path
            frame.update(revision=self.revision, data=changes)
            await self.send(frame)
            if full:
                self.published_data = data
            else:
                self.published_data.update(data)

    async def publish_events(self):
        while True:
            event = await self.event_queue.get()
            try:
                kind = event["event_type"]
                if kind not in {"llm.started", "llm.delta", "llm.progress", "llm.retry", "context.preparing", "run.output", "environment.command_progress"}:
                    await self.publish_snapshot(history=kind in {"agent.message", "context.compacted", "agent.completed", "agent.failed", "agent.blocked", "agent.cancelled"}, diff=kind == "patch.applied", instances=kind in {"instance.created", "environment.prepared", "verification.completed"})
                await self.send({"type": "event", "event": event})
            except Exception as error:
                self.fatal_error = error
                self.stopping.set()
                raise
            finally:
                self.event_queue.task_done()

    async def switch_model(self, profile_name):
        self.harness.switch_model(profile_name)

    async def execute_tool(self, call):
        self.harness._require_instance()
        if self.harness.busy:
            raise RuntimeError("当前任务正在执行")
        if not isinstance(call, dict) or set(call) != {"id", "name", "arguments"}:
            raise ValueError("工具请求必须提供调用 ID、名称和结构化参数")
        if not isinstance(call["id"], str) or not call["id"] or not isinstance(call["arguments"], dict) or not isinstance(call["name"], str) or call["name"] not in self.harness.tools.tools:
            raise ValueError("工具请求名称、调用 ID 或参数无效")
        self.harness._ensure_task("执行用户指定工具：" + call["name"])
        return await self.harness._action(self.harness.tools.execute(copy.deepcopy(call)))

    async def refresh(self):
        return {"status": "refreshed"}

    def error_detail(self, error):
        message = str(error)
        for profile in self.settings.profiles.values():
            secret = os.environ.get(profile.api_key_env)
            if secret:
                message = message.replace(secret, "[凭据已隐藏]")
        return {"type": type(error).__name__, "message": message or "后台操作已取消"}

    async def flush_events(self):
        # EventBus 的线程回调进入队列后，才确认对应 RPC 已经完成。
        await asyncio.sleep(0)
        await self.flush_queue(self.event_queue, self.publisher_task, "后台事件发布已经停止")

    async def flush_queue(self, queue, consumer, message):
        if self.fatal_error is not None:
            raise self.fatal_error
        joined = asyncio.create_task(queue.join())
        try:
            done, _ = await asyncio.wait((joined, consumer), return_when=asyncio.FIRST_COMPLETED)
            if consumer in done:
                consumer.result()
                raise TransportError(message)
            await joined
        finally:
            if not joined.done():
                joined.cancel()
            await asyncio.gather(joined, return_exceptions=True)

    async def dispatch(self, frame):
        request_id = frame["id"]
        method = frame["method"]
        try:
            if method not in self.methods:
                raise ValueError("后台未登记此操作")
            expected, handler = self.methods[method]
            params = frame.get("params", {})
            if not isinstance(params, dict) or set(params) != expected:
                raise ValueError("后台操作参数与已登记接口不一致")
            if method == "approve" and not isinstance(params["approved"], bool):
                raise ValueError("审批结果必须为布尔值")
            result = await handler(**params)
        except asyncio.CancelledError as error:
            await self.flush_events()
            await self.publish_snapshot(history=True, diff=True, instances=True)
            await self.send({"type": "response", "id": request_id, "error": self.error_detail(error)})
        except Exception as error:
            await self.flush_events()
            await self.publish_snapshot(history=True, diff=True, instances=True)
            await self.send({"type": "response", "id": request_id, "error": self.error_detail(error)})
        else:
            if method in {"create_instance", "restore_instance"} and isinstance(result, dict):
                result = self.public_instance(result)
            await self.flush_events()
            await self.publish_snapshot(history=True, diff=method in {"execute_tool", "send_message", "resume_current", "refresh"}, instances=True)
            await self.send({"type": "response", "id": request_id, "result": result})
        finally:
            self.requests.pop(request_id, None)
            if method == "close":
                self.stopping.set()

    async def accept_requests(self):
        assembler = MessageAssembler()
        while True:
            line = await self.incoming.get()
            if isinstance(line, Exception):
                raise line
            if not line:
                self.stopping.set()
                return
            frame = await asyncio.to_thread(assembler.feed, line)
            if frame is None:
                continue
            if frame["type"] != "request" or not isinstance(frame.get("id"), str) or not isinstance(frame.get("method"), str):
                raise TransportError("后台收到无效请求")
            request_id = frame["id"]
            if not request_id or request_id in self.requests:
                raise TransportError("后台请求 ID 为空或重复")
            task = asyncio.create_task(self.dispatch(frame))
            self.requests[request_id] = task
            task.add_done_callback(self.request_finished)

    def request_finished(self, task):
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                self.fatal_error = error
                self.stopping.set()

    async def serve(self):
        writer = asyncio.create_task(self.write_output())
        publisher = asyncio.create_task(self.publish_events())
        self.publisher_task = publisher
        reader = asyncio.create_task(self.accept_requests())
        stopped = None
        input_thread = threading.Thread(target=self.read_input, name="ianvs-stdio-input", daemon=True)
        input_thread.start()
        try:
            await self.publish_snapshot(full=True)
            await self.send({"type": "ready", "backend_info": self.backend_info, "methods": list(self.methods)})
            stopped = asyncio.create_task(self.stopping.wait())
            done, _ = await asyncio.wait((reader, writer, publisher, stopped), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not stopped:
                    task.result()
            await self.harness.close()
            await asyncio.gather(*tuple(self.requests.values()), return_exceptions=True)
            await self.flush_events()
            await self.publish_snapshot(history=True, diff=True, instances=True)
            await self.flush_queue(self.outgoing, writer, "后台通信发送已经停止")
            if self.fatal_error:
                raise self.fatal_error
        finally:
            self.unsubscribe()
            await self.harness.close()
            tasks = [task for task in (reader, writer, publisher, stopped) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def run_server(settings):
    settings.state_root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(settings.state_root / "transport-backend.lock"), timeout=0)
    with lock:
        await BackendService(settings).serve()


def main():
    parser = argparse.ArgumentParser(description="Ianvs Agent stdio 后台")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        asyncio.run(run_server(load_settings(args.project_root, args.config)))
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
