# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json


def tool_evidence_references(data, instance_id):
    if not isinstance(data, dict):
        return set()
    references = {data[field] for field in ("path", "source", "reference", "full_output_ref") if isinstance(data.get(field), str)}
    prefix = f"instances/{instance_id}/"
    for record in (data, data.get("postcheck"), data.get("current_check"), data.get("original_goal")):
        reference = record.get("evidence_ref") if isinstance(record, dict) else None
        if isinstance(reference, str) and reference.startswith(prefix):
            references.add(reference)
    return references


def observation_statement(item):
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        # 兼容已保存会话的中文字段，正式证据与原始记录保持原样。
        for field in ("value", "text", "fact", "statement", "事实", "假设"):
            if field in item:
                return observation_statement(item[field])
    return ""


def format_task_result(task, state, status, message, run=None, record_ref=None):
    statement = observation_statement
    facts = [{"text": statement(item), "evidence_ref": item.get("evidence_ref")} for item in state.get("facts", []) if isinstance(item, dict) and statement(item)]
    hypotheses = list(dict.fromkeys(statement(item) for item in state.get("hypotheses", []) if statement(item)))
    verification = state.get("verification") or {}
    next_action = state.get("next_goal") or ""
    maintenance = state.get("maintenance_progress") or {}
    result = {"status": status, "message": message, "facts": facts, "hypotheses": hypotheses, "next_action": next_action, "verification": verification, "maintenance": maintenance, "record_ref": record_ref}
    if status == "completed" and task.get("kind") == "conversation":
        result["content"] = message
        return result
    titles = {"completed": "任务处理完成", "blocked": "当前任务阻塞", "failed": "当前操作失败", "cancelled": "任务已取消", "interrupted": "上次任务中断"}
    lines = [titles.get(status, "任务状态") + "。"]
    if task.get("kind") == "maintenance":
        receipt = maintenance.get("environment_check") or {}
        lines.append("环境检查：" + ("已通过，本任务的安装与运维证据已保存。" if maintenance.get("phase") == "completed" else "尚未取得修改后的完整检查结果。"))
        if receipt.get("evidence_ref"):
            lines.append("环境检查记录：" + receipt["evidence_ref"])
    run = run or state.get("last_run") or {}
    if run.get("error"):
        lines.append(f"历史运行 {run.get('run_id', '')} 的失败原因：{run['error']}。此记录反映当时的环境，当前环境需要重新运行确认。")
    if facts:
        unique = list({item["evidence_ref"] or item["text"]: item for item in facts}.values())
        visible = unique if len(unique) <= 6 else unique[:3] + unique[-3:]
        lines.append("已记录的诊断发现与证据：")
        lines.extend(f"- {item['text']}\n  证据：{item['evidence_ref']}" for item in visible)
        if len(unique) > len(visible):
            lines.append(f"另有 {len(unique) - len(visible)} 项诊断证据已保存。")
    if hypotheses:
        lines.append("尚待核实：")
        lines.extend("- " + item for item in hypotheses[-3:])
        if len(hypotheses) > 3:
            lines.append(f"其余 {len(hypotheses) - 3} 项待核实内容已保存。")
    if message and message != next_action:
        lines.append("处理结果：" + message)
    if next_action:
        lines.append("下一步：" + next_action)
    passed = verification.get("passed") is True and not verification.get("stale")
    lines.append("原始评测独立验证：" + ("通过。" if passed else "尚未通过；当前不能宣称修复完成。"))
    if record_ref:
        lines.append("完整诊断记录：" + record_ref)
    result["content"] = "\n\n".join(lines)
    return result


def is_observation(message):
    if message.get("role") != "assistant" or message.get("tool_calls"):
        return False
    stage = message.get("_ianvs_stage")
    if stage is not None:
        return stage == "observe"
    content = message.get("content") or ""
    if not isinstance(content, str):
        return False
    value = content.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(parsed, dict)
        and parsed.get("decision") in {"continue", "completed", "blocked"}
        and isinstance(parsed.get("facts"), list)
        and isinstance(parsed.get("hypotheses"), list)
        and isinstance(parsed.get("next_goal"), str)
        and set(parsed) <= {"facts", "hypotheses", "next_goal", "decision"}
    )
