# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime

from ianvs_agent.message_types import is_observation


def short_text(value, limit=90):
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def assistant_text(payload):
    content = payload.get("content") or payload.get("text")
    if payload.get("role") == "compaction" or not isinstance(content, str) or not content.strip():
        return ""
    message = {"role": "assistant", "content": content, "_ianvs_stage": payload.get("role")}
    return "" if is_observation(message) else content


def environment_text(event, run=None):
    payload = event.get("payload") or {}
    if event["event_type"] == "environment.inspected":
        imports = payload.get("imports") or []
        pip_check = payload.get("pip_check") or {}
        lines = ["环境检查通过。" if payload.get("passed") else "环境检查尚未通过。"]
        lines.append(f"依赖一致性：{'通过' if pip_check.get('passed') else '未通过'}；模块导入：{sum(item.get('passed') is True for item in imports)}/{len(imports)}。")
        if payload.get("evidence_ref"):
            lines.append("检查证据：" + payload["evidence_ref"])
        if payload.get("error"):
            lines.append("检查错误：" + short_text(payload["error"], 300))
        lines.append("原始评测的成功条件仍由独立验证器检查。")
        return "\n\n".join(lines)
    if event["event_type"] in {"environment.command_completed", "environment.command_failed", "environment.credentials_configured"}:
        label = "运行凭据引用已绑定。" if event["event_type"] == "environment.credentials_configured" else "环境命令已完成。" if event["event_type"] == "environment.command_completed" else "环境命令未完成。"
        reference = payload.get("evidence_ref") or payload.get("log_ref")
        return label + ("\n\n证据：" + reference if reference else "") + ("\n\n错误：" + short_text(payload["error"], 300) if payload.get("error") else "")
    if event["event_type"] not in {"environment.completed", "environment.prepared", "environment.failed"}:
        return ""
    lines = [payload.get("summary") or ("环境准备失败：" + str(payload.get("error", "")) if event["event_type"] == "environment.failed" else "声明的运行环境已经准备完成。模型和数据仍需要通过运行确认。")]
    lines.extend("提示：" + str(warning) for warning in payload.get("warnings", []))
    failed = payload.get("failed_step") or {}
    if failed.get("log_ref"):
        lines.append("失败日志：" + failed["log_ref"])
    if failed.get("error_tail"):
        lines.append("失败输出：\n" + failed["error_tail"])
    if run and run.get("started_at") and event.get("timestamp") and datetime.fromisoformat(run["started_at"]) < datetime.fromisoformat(event["timestamp"]):
        lines.append(f"最近运行 {run['run_id']} 发生在本次环境准备之前；准备后尚未重新运行。")
    elif not run:
        lines.append("尚未运行示例。")
    if event["event_type"] == "environment.failed":
        lines.append("请根据失败步骤与日志修复环境，然后重新准备。")
    elif payload.get("status") == "created":
        lines.append("需要完成依赖安装后才能运行示例。")
    else:
        lines.append("本次环境步骤已完成；若任务要求验证评测结果，可运行原始任务并取得独立验证结果。")
    return "\n\n".join(lines)


def conversation_messages(history, events):
    assistant_events = [event for event in events if event["event_type"] in {"agent.message", "assistant.message"}]
    used = set()
    next_event = 0
    messages = []
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        if role == "assistant":
            source = None
            for index in range(next_event, len(assistant_events)):
                if assistant_events[index].get("payload", {}).get("content") == content:
                    source = assistant_events[index]
                    used.add(index)
                    next_event = index + 1
                    break
            stage = message.get("_ianvs_stage") or (source.get("payload", {}).get("role") if source else None)
            if stage == "compaction" or is_observation({**message, "_ianvs_stage": stage}):
                continue
        messages.append((role, content))
    for index, event in enumerate(assistant_events):
        content = assistant_text(event.get("payload") or {})
        if index not in used and content:
            messages.append(("assistant", content))
    return messages


def tool_parameters(arguments):
    for key in ("path", "command", "pattern", "reference", "benchmark_path", "skill"):
        if arguments.get(key):
            return short_text(arguments[key])
    if arguments.get("patch"):
        return "应用补丁 · " + short_text(arguments["patch"].splitlines()[0])
    return short_text(json.dumps(arguments, ensure_ascii=False)) if arguments else "无参数"


def tool_outcome(record):
    outcome = record.get("outcome")
    if isinstance(outcome, dict):
        return outcome
    data = record.get("data")
    if record.get("name") not in {"run_example", "verify_run"} or not isinstance(data, dict):
        return {}
    run = data.get("run") or {}
    verification = data.get("verification") if record.get("name") == "run_example" else data
    verification = verification if isinstance(verification, dict) else {}
    return {
        "run_id": run.get("run_id") or verification.get("run_id"),
        "run_status": run.get("status"),
        "exit_code": run.get("exit_code"),
        "verification_passed": verification.get("passed"),
        "verification_stale": bool(verification.get("stale") or verification.get("status") == "needs_verification"),
        "repair_completed": verification.get("repair_completed"),
        "full_run_ref": run.get("full_run_ref"),
    }


def tool_display_status(record):
    status = record.get("status")
    if status != "success":
        return status
    outcome = tool_outcome(record)
    if (outcome.get("run_status") in {"failed", "cancelled", "interrupted", "timed_out"}
            or outcome.get("exit_code") not in {None, 0}
            or outcome.get("verification_passed") is False
            or outcome.get("verification_stale")):
        return "error"
    if record.get("name") in {"run_example", "verify_run"} and outcome.get("verification_passed") is not True:
        return "unconfirmed"
    data = record.get("data")
    if isinstance(data, dict) and (data.get("passed") is False or data.get("valid") is False
                                  or isinstance(data.get("postcheck"), dict) and data["postcheck"].get("passed") is False):
        return "error"
    return status


def tool_summary(record):
    if record.get("error"):
        return short_text(record["error"])
    status = record.get("status")
    if status == "awaiting_approval":
        return "等待批准或拒绝"
    if status == "running":
        return "等待执行结果"
    outcome = tool_outcome(record)
    if outcome:
        labels = []
        if outcome.get("run_id"):
            labels.append(outcome["run_id"])
        run_status = outcome.get("run_status")
        if run_status:
            labels.append({"completed": "原任务执行完成", "failed": "原任务执行失败", "cancelled": "原任务已取消", "interrupted": "原任务已中断", "timed_out": "原任务执行超时"}.get(run_status, run_status))
        if outcome.get("exit_code") is not None:
            labels.append(f"退出码 {outcome['exit_code']}")
        passed = outcome.get("verification_passed")
        labels.append("需要重新验证" if outcome.get("verification_stale") else "独立验证通过" if passed is True else "独立验证未通过" if passed is False else "独立验证尚未确认")
        if outcome.get("repair_completed") is True and passed is True and not outcome.get("verification_stale"):
            labels.append("修复完成")
        return short_text(" · ".join(labels))
    if record.get("name") in {"run_example", "verify_run"}:
        return "运行与验证结果待核查，完整结果见正式记录"
    data = record.get("data")
    if isinstance(data, dict):
        if "plan_id" in data and data.get("requires_install"):
            return short_text(f"安装计划 {data['plan_id']} · 等待执行 · {data.get('evidence_ref', '')}")
        if data.get("source_kind") == "observed_environment":
            return short_text(f"环境检查{'通过' if data.get('passed') else '未通过'} · {data.get('check_id', '')} · {data.get('evidence_ref', '')}")
        if "postcheck" in data:
            check = data.get("postcheck") or {}
            return short_text(f"{data.get('environment_id', '')} · 安装后检查{'通过' if check.get('passed') else '未通过'} · {check.get('evidence_ref', '')}")
        if "operation_id" in data and "exit_code" in data:
            return short_text(f"环境命令 {data['operation_id']} · 退出码 {data.get('exit_code')} · {data.get('evidence_ref', '')}")
        if "return_code" in data:
            text = data.get("stderr") if data["return_code"] else data.get("stdout")
            return short_text(f"退出码 {data['return_code']} · {text or '没有输出'}")
        if data.get("exists") is False:
            return "文件尚不存在，已记录 absent 版本"
        if "content" in data:
            if "start_line" in data and "end_line" in data:
                return f"读取第 {data['start_line']}–{data['end_line']} 行"
            return short_text(data["content"])
        if "matches" in data:
            return f"找到 {data.get('total_matches', len(data['matches']))} 处匹配"
        if "run_id" in data:
            return short_text(f"{data['run_id']} · {data.get('status', '已取得运行记录')}")
        if "modification_id" in data:
            return short_text(f"补丁 {data['modification_id']} · {len(data.get('changes', []))} 个文件")
        if "valid" in data:
            return "配置检查通过" if data["valid"] else "配置需要修复"
        return short_text(" · ".join(f"{key}={value}" for key, value in list(data.items())[:2]))
    if isinstance(data, list):
        return f"返回 {len(data)} 项结果"
    return short_text(data) or "执行完成，详情已保存"
