# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import load_settings
from ianvs_agent.context.compactor import complete_interactions
from ianvs_agent.runtime.instances import file_manifest


BENCHMARK = "examples/llm-edge-benchmark-suite/single_task_bench/benchmarkingjob.yaml"
PRIVATE_MODEL = "company-private-model-v1"


async def check(state_root=None, model_call_limit=10, project_root=None):
    settings_root = Path(__file__).resolve().parents[2]
    project_root = Path(project_root).resolve() if project_root else settings_root
    settings = load_settings(settings_root)
    settings.project_root = project_root
    identity = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    settings.state_root = Path(state_root).resolve() if state_root else project_root / "workspace/ianvs-agent/acceptance/harness-dialogue" / identity
    settings.selected_profile = "ark"
    if model_call_limit < 2 or model_call_limit > 10:
        raise ValueError("真实会话验收模型预算必须在 2 至 10 次之间")
    settings.max_model_calls = model_call_limit
    settings.max_tool_calls = 20
    settings.max_task_seconds = 300
    settings.approval_policy = "deny"
    harness = AgentHarness(settings)
    report = {"identity": identity, "evidence_type": "real_harness_provider_and_example_files", "model": settings.profile.model, "benchmark_path": BENCHMARK, "state_root": str(settings.state_root), "passed": False, "cases": []}
    try:
        if harness.current_instance:
            raise ValueError("本次验收目录已有实例；请指定新的独立目录")
        instance = await harness.create_instance(BENCHMARK)
        code_root = Path(instance["code_root"])
        baseline = file_manifest(code_root)
        report["instance_id"] = instance["instance_id"]
        report["session_path"] = harness.session_relative_path
        readme = instance["readme_path"]
        consultation = await harness.send_message(f"请解释当前 Example 的用途、模型与数据准备方法、运行命令和结果含义。先调用 read_file 读取 {readme}，根据实际原文用中文回答；当前只说明文档，不执行环境准备、评测或文件修改。")
        tools_path = harness.session_relative_path + "/tools.jsonl"
        consultation_tools = harness.store.read_events(tools_path)
        consultation_case = {"name": "README 自然语言咨询", "result": consultation, "task": harness.task, "state": harness.state, "tool_records": consultation_tools, "usage": harness.usage.summary()}
        report["cases"].append(consultation_case)
        if consultation["status"] != "completed" or harness.task.get("kind") != "conversation":
            raise ValueError("自然语言咨询未作为 conversation 任务完成")
        if not any(record["name"] == "read_file" and record["status"] == "success" for record in consultation_tools):
            raise ValueError("自然语言咨询没有通过真实工具读取文档")
        if harness.state.get("verification", {}).get("passed") is True or harness.state.get("verification", {}).get("repair_completed") is True:
            raise ValueError("自然语言咨询错误地声称评测或修复通过")
        if "repair_task_id" in harness.task["acceptance"] or harness.state.get("changes"):
            raise ValueError("自然语言咨询错误地创建修复验收或文件修改")
        if consultation_case["usage"].get("current_task_id") != harness.task["task_id"] or consultation_case["usage"].get("task_model_calls") != consultation_case["usage"]["model_calls"]:
            raise ValueError("自然语言咨询的正式任务用量没有独立登记")
        print(json.dumps({"case": "consultation", "status": consultation["status"], "calls": harness.usage.summary()["model_calls"]}, ensure_ascii=False), flush=True)
        remaining_calls = model_call_limit - consultation_case["usage"]["model_calls"]
        if remaining_calls <= 0:
            raise ValueError("本次会话剩余模型预算不足以开始阻塞检查")
        settings.max_model_calls = remaining_calls

        model_root = code_root / "models" / PRIVATE_MODEL
        if model_root.exists():
            raise ValueError("验收请求的私有模型已存在，不能用于缺失模型检查")
        missing_configuration = "models/" + PRIVATE_MODEL + "/config.json"
        goal = f"修复当前 Example 的运行准备。本次任务必须使用公司私有模型 {PRIVATE_MODEL}。权重、Tokenizer 和分发授权均未提供，当前没有合法来源。第一项动作必须调用 read_file 读取 {missing_configuration}，使用实际文件结果确认模型配置的缺失条件；README 原文已经通过工具读取，可用于解释原 Example。取得文件结果之后，将缺失模型写为带已登记 evidence_ref 的事实，并明确判定 blocked；模型身份和原始任务要求保持有效。禁止生成占位模型文件、改用其他模型、降低验收要求、执行环境安装或开始评测。"
        harness._ensure_task(goal, kind="repair")
        harness.task["acceptance"]["required_model_identity"] = PRIVATE_MODEL
        harness.task["acceptance"]["model_authorization"] = "not_provided"
        blocked = await harness.send_message(goal)
        blocked_tools = harness.store.read_events(tools_path)[len(consultation_tools):]
        blocked_case = {"name": "缺失私有模型与授权", "result": blocked, "task": harness.task, "state": harness.state, "tool_records": blocked_tools, "usage": harness.usage.summary()}
        report["cases"].append(blocked_case)
        if blocked["status"] != "blocked" or harness.task.get("kind") != "repair":
            raise ValueError("缺失私有模型没有形成正式修复任务阻塞记录")
        if harness.task["goal"] != goal or harness.task["acceptance"].get("required_model_identity") != PRIVATE_MODEL:
            raise ValueError("阻塞检查改变了原始模型目标")
        if not any(record["name"] == "read_file" and record["arguments"].get("path") == missing_configuration and record["status"] == "error" for record in blocked_tools):
            raise ValueError("没有真实读取缺失模型配置的文件错误证据")
        facts = harness.state.get("facts", [])
        if not any(PRIVATE_MODEL in json.dumps(fact, ensure_ascii=False) and fact.get("evidence_status") == "registered_reference" for fact in facts):
            raise ValueError("正式状态没有保存带登记证据的缺失模型事实")
        if harness.state.get("verification", {}).get("passed") is True or harness.state.get("verification", {}).get("repair_completed") is True:
            raise ValueError("缺失模型任务错误地声称验证或修复通过")
        if model_root.exists() or harness.state.get("changes") or file_manifest(code_root) != baseline:
            raise ValueError("阻塞检查创建了模型文件或改变了实例源码")
        complete_interactions(harness._messages)
        report["usage"] = harness.usage.summary()
        if report["usage"]["current_task_id"] != harness.task["task_id"] or report["usage"]["task_model_calls"] != report["usage"]["model_calls"] - consultation_case["usage"]["model_calls"]:
            raise ValueError("阻塞任务用量与当前会话累计用量没有正确区分")
        if report["usage"]["model_calls"] > model_call_limit:
            raise ValueError("本次会话检查超过模型调用预算")
        report["passed"] = True
        print(json.dumps({"case": "private_model", "status": blocked["status"], "calls": report["usage"]["model_calls"], "registered_facts": len(facts)}, ensure_ascii=False), flush=True)
    except Exception as error:
        report["error"] = harness.provider._error_message(error) if harness.provider else type(error).__name__
        raise RuntimeError("真实会话验收未通过；请检查已保存的脱敏记录") from None
    finally:
        harness.store.write("acceptance.json", report)
        exported = project_root / "workspace/ianvs-agent/acceptance/harness-dialogue" / identity
        exported.mkdir(parents=True, exist_ok=True)
        (exported / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        await harness.close()
        print(json.dumps({"passed": report["passed"], "report": str(exported / "acceptance.json")}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="使用真实方舟调用检查咨询完成与缺失私有模型阻塞")
    parser.add_argument("--state-root")
    parser.add_argument("--max-model-calls", type=int, default=10)
    parser.add_argument("--project-root")
    args = parser.parse_args()
    asyncio.run(check(args.state_root, args.max_model_calls, args.project_root))


if __name__ == "__main__":
    main()
