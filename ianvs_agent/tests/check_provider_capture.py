# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk

from ianvs_agent.llm.provider import Provider


def replay(chunks):
    state = ChatCompletionStreamState()
    roles = {}
    for value in chunks:
        Provider._accumulate(state, ChatCompletionChunk.model_validate(value), roles)
    return state.get_final_completion()


def check(path):
    saved = json.loads(path.read_text(encoding="utf-8"))
    chunks = saved.get("raw_chunks")
    if not chunks or saved.get("status") != "completed":
        raise ValueError("验收材料必须包含真实请求完成后的 raw_chunks")
    original_chunks = json.dumps(chunks, ensure_ascii=False)
    response = replay(chunks)
    expected = saved["response"]
    content, calls = Provider._validate_response(response, bool(saved["request"].get("tools")))
    if content != expected["content"] or calls != expected["tool_calls"]:
        raise ValueError("SDK 重放内容或完整工具参数与原始记录不一致")
    message = response.choices[0].message.model_dump(mode="json", exclude_none=True)
    if message["role"] != "assistant":
        raise ValueError("SDK 重放后的模型消息身份必须为 assistant")
    if {key: value for key, value in message.items() if key != "role"} != {key: value for key, value in expected["raw_message"].items() if key != "role"}:
        raise ValueError("SDK 重放消息及模型扩展字段与原始记录不一致")
    if original_chunks != json.dumps(chunks, ensure_ascii=False):
        raise ValueError("SDK 规范化处理修改了原始捕获记录")
    usage = response.usage.model_dump(mode="json", exclude_none=True) if response.usage is not None else None
    if usage != expected["usage"]["raw_usage"]:
        raise ValueError("SDK 重放用量与 Provider 正式用量记录不一致")
    interrupted_tool_rejected = None
    if calls:
        # 截取真实响应的工具参数片段，验证连接中断时不会交付工具调用。
        prefix = []
        for chunk in chunks:
            prefix.append(chunk)
            if any(choice.get("delta", {}).get("tool_calls") and choice.get("finish_reason") is None for choice in chunk["choices"]):
                break
        if len(prefix) == len(chunks):
            raise ValueError("真实工具捕获缺少可用于中断检查的参数片段")
        partial = replay(prefix)
        try:
            Provider._validate_response(partial, True)
        except ValueError:
            interrupted_tool_rejected = True
        else:
            raise ValueError("未完成的真实工具响应被错误交付")
    return {
        "capture": str(path), "passed": True, "chunk_count": len(chunks),
        "tool_calls": len(calls), "interrupted_tool_rejected": interrupted_tool_rejected,
        "usage_preserved": True, "message_fields_preserved": True, "role": message["role"], "raw_chunks_preserved": True,
        "extended_fields": sorted(set(message) - {"content", "role", "tool_calls"}),
    }


def main():
    parser = argparse.ArgumentParser(description="通过真实 Provider 捕获重放 SDK 内容、工具参数、用量和扩展字段")
    parser.add_argument("capture", type=Path, nargs="+")
    arguments = parser.parse_args()
    for path in arguments.capture:
        print(json.dumps(check(path.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
