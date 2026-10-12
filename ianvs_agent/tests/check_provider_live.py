# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import load_settings
from ianvs_agent.context.builder import ContextManager
from ianvs_agent.llm.provider import Provider
from ianvs_agent.llm.usage import UsageTracker
from ianvs_agent.storage.local_store import LocalStore


READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取当前验收任务明确指定的本地文档，返回真实原文与行号",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}},
            "required": ["path", "start_line", "end_line"],
            "additionalProperties": False,
        },
    },
}


def execute_read(call, allowed):
    if call["name"] != "read_file":
        raise ValueError("模型选择了验收任务之外的工具")
    arguments = call["arguments"]
    path = Path(arguments["path"]).resolve()
    if path not in allowed:
        raise ValueError("模型读取路径不在本次文档验收范围")
    start = arguments["start_line"]
    end = arguments["end_line"]
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start or end - start > 100:
        raise ValueError("模型读取行范围超出本次验收限制")
    lines = path.read_text(encoding="utf-8").splitlines()
    return {"status": "success", "source": str(path), "start_line": start, "end_line": min(end, len(lines)), "content": "\n".join(f"{index + 1}: {line}" for index, line in enumerate(lines[start - 1:end], start - 1))}


async def read_round(provider, history, source):
    history.append({"role": "user", "content": f"请必须调用 read_file 读取 {source} 第 1 至 25 行，收到真实工具结果后使用中文说明这份文件的用途。当前阶段只读取文档。"})
    response = await provider.complete(history, [READ_TOOL], purpose="provider_acceptance")
    if not response["tool_calls"]:
        raise ValueError("模型没有按验收任务调用 read_file")
    history.append(response["raw_message"])
    for call in response["tool_calls"]:
        result = execute_read(call, {source})
        provider.usage.record_tool("success")
        history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
    answer = await provider.complete(history, [], purpose="provider_acceptance")
    if not answer["content"].strip() or answer["tool_calls"]:
        raise ValueError("模型读取工具结果后没有给出说明")
    history.append(answer["raw_message"])
    return answer


async def check(settings, profile_name, store, compact):
    session_id = store.allocate_id("PROVIDER-CHECK")
    events = EventBus(store, session_id=session_id, instance_id="DOCUMENT-CHECK")
    session_path = f"instances/DOCUMENT-CHECK/sessions/{session_id}"
    usage = UsageTracker(store, session_path, events)
    provider = Provider(settings.profiles[profile_name], events, usage)
    context = ContextManager(settings, store, events, provider)
    events.emit("session.started", {"purpose": "真实模型文档读取验收", "profile": profile_name})
    source = (settings.project_root / "README.md").resolve()
    history = [{"role": "system", "content": "使用中文完成本地文档读取验收。必须通过 read_file 工具获取原文，所有工具结果来自实际文件；工具结果尚未返回时不要自行描述文件内容。"}]
    report = {"profile": profile_name, "model": provider.profile.model, "session_id": session_id, "evidence_type": "real_provider_and_files", "passed": False}
    try:
        await read_round(provider, history, source)
        if compact:
            await read_round(provider, history, (settings.project_root / "docs/guides/quick-start.md").resolve())
            await read_round(provider, history, (settings.project_root / "docs/guides/how-to-test-algorithms.md").resolve())
            task = {"task_id": "DOCUMENT-CHECK", "goal": "解释通过工具读取的 Ianvs 文档，保持来源和未知内容", "readme_path": str(source), "acceptance": {"require_real_source": True}}
            state = {"facts": [], "hypotheses": [], "changes": [], "verification": {"passed": False, "scope": "document_reading_only"}}
            checkpoint = await context.compact(task, state, history, [READ_TOOL], manual=True)
            recovered = ContextManager(settings, LocalStore(store.root), events, provider)
            prepared = await recovered.prepare(task, state, history, [READ_TOOL])
            if prepared["checkpoint_id"] != checkpoint["checkpoint_id"]:
                raise ValueError("重新创建 ContextManager 后没有恢复压缩检查点")
            report["checkpoint_id"] = checkpoint["checkpoint_id"]
            report["source_event_range"] = checkpoint["source_event_range"]
            history.append({"role": "user", "content": "当前状态经过压缩后已恢复。请简短说明已读材料的用途，并明确当前任务只核查文档读取，没有运行 Benchmark。"})
            continuation = await recovered.complete(task, state, history, [], purpose="context_acceptance")
            if not continuation["content"].strip():
                raise ValueError("压缩后模型没有继续回应")
            history.append(continuation["raw_message"])
            settings.context_threshold = 0.001
            automatic = await recovered.prepare(task, state, history, [])
            if automatic["checkpoint_id"] == checkpoint["checkpoint_id"]:
                raise ValueError("增加真实交互后没有触发自动压缩")
            report["automatic_checkpoint_id"] = automatic["checkpoint_id"]
        report.update({"passed": True, "usage": usage.summary()})
        store.write(session_path + "/acceptance.json", report)
        store.write(session_path + "/history.json", {"messages": history})
        print(json.dumps({"profile": profile_name, "model": provider.profile.model, "passed": True, "model_calls": usage.summary()["model_calls"], "compactions": usage.summary()["compactions"], "unknown_usage_calls": usage.summary()["unknown_usage_calls"], "report": str(store.path(session_path + "/acceptance.json"))}, ensure_ascii=False))
    except Exception as error:
        report.update({"error_type": type(error).__name__, "error": provider._error_message(error), "usage": usage.summary()})
        store.write(session_path + "/acceptance.json", report)
        store.write(session_path + "/history.json", {"messages": history})
        raise
    finally:
        await provider.close()
    return report


async def main():
    parser = argparse.ArgumentParser(description="使用真实方舟 Provider 和本地文档检查 Tool Calling、Usage 与上下文压缩")
    parser.add_argument("--profile", action="append", choices=["ark", "glm"])
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args()
    settings = load_settings()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    store = LocalStore(settings.state_root / "acceptance/provider" / stamp)
    for name in arguments.profile or ["ark", "glm"]:
        await check(settings, name, store, arguments.compact and name == "ark")


if __name__ == "__main__":
    asyncio.run(main())
