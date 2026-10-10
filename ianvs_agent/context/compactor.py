# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

from ianvs_agent.message_types import is_observation


def message_fingerprint(messages):
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def request_history(history):
    complete_interactions(history)
    return [
        {key: value for key, value in message.items() if not key.startswith("_ianvs_")}
        for message in history if not is_observation(message)
    ]


def latest_tool_interaction(history):
    blocks = complete_interactions(history)
    if not blocks:
        return []
    latest = blocks[-1]
    if latest[0].get("role") != "assistant" or not latest[0].get("tool_calls"):
        raise ValueError("Observe 需要当前完整工具交互，不能重复观察较早历史")
    return [{key: value for key, value in message.items() if not key.startswith("_ianvs_")} for message in latest]


def complete_interactions(history):
    blocks = []
    pending = set()
    block = []
    for message in history:
        role = message.get("role")
        calls = message.get("tool_calls") or []
        if pending and role != "tool":
            raise ValueError("工具调用缺少配对结果，无法组织上下文")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("工具结果没有对应的调用 ID")
            pending.remove(call_id)
        elif calls:
            if role != "assistant":
                raise ValueError("tool_calls 必须属于 assistant 消息")
            for call in calls:
                call_id = call.get("id")
                if not call_id or call_id in pending:
                    raise ValueError("工具调用 ID 缺失或重复")
                pending.add(call_id)
        block.append(message)
        if not pending:
            blocks.append(block)
            block = []
    if pending:
        raise ValueError("历史末尾存在缺少结果的工具调用")
    return blocks


def history_split(history, keep_recent_turns):
    blocks = complete_interactions(history)
    assistant_blocks = [index for index, block in enumerate(blocks) if block[0].get("role") == "assistant"]
    if len(assistant_blocks) <= keep_recent_turns:
        return [], list(history)
    boundary = assistant_blocks[-keep_recent_turns]
    # 与下一条 assistant 对应的用户输入一同保留。
    while boundary > 0 and blocks[boundary - 1][0].get("role") == "user":
        boundary -= 1
    older = [message for block in blocks[:boundary] for message in block]
    recent = [message for block in blocks[boundary:] for message in block]
    return older, recent
