# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import difflib
import json
from dataclasses import replace
from pathlib import Path

from ianvs_agent.agent.harness import AgentHarness
from ianvs_agent.config import load_settings
from ianvs_agent.context.compactor import complete_interactions


async def check(args):
    settings = replace(load_settings(), state_root=args.state_root.resolve(), python_executable=args.python, approval_policy="auto_scoped", keep_recent_turns=1)
    if args.request_timeout:
        settings.profile.timeout_seconds = args.request_timeout
    harness = AgentHarness(settings)
    harness.select_instance(args.instance)
    instance = harness.current_instance
    baseline = harness.runner.get_run(args.instance, args.baseline_run)
    verification = harness.store.read(f"instances/{args.instance}/runs/{args.baseline_run}/verification.json")
    if baseline["status"] != "completed" or not verification or not verification["passed"]:
        raise RuntimeError("历史真实健康基线没有通过独立检查")
    benchmark = instance["benchmark_path"]
    harness._ensure_task("修复所选 Example 的配置故障，保持原始算法、数据、模型、指标和 acc=1.0，通过实际 Ianvs 运行与独立验证")
    originals = baseline["source_manifest"]
    protected = {path: value for path, value in originals.items() if path.startswith("examples/llm_simple_qa/") or path.startswith("dataset/llm_simple_qa/")}
    harness.task["acceptance"].update(metric_thresholds={"acc": {"min": 1.0}}, file_identities=protected, repair_task_id=harness.task["task_id"])
    harness._save_session()
    original = (Path(instance["code_root"]) / benchmark).read_bytes().decode("utf-8")
    target = "./examples/llm_simple_qa/testenv/testenv.yaml"
    if target in original:
        current_verification = await harness.verifier.verify(instance, baseline)
        if not current_verification["passed"]:
            raise RuntimeError("当前实例没有通过真实健康检查")
        read = await harness.tools.execute({"id": "acceptance-read-benchmark", "name": "read_file", "arguments": {"path": benchmark}})
        if read["status"] != "success":
            raise RuntimeError("故障注入所需的配置读取没有完成：" + read["error"])
        damaged = original.replace(target, "./examples/llm_simple_qa/testenv/missing-testenv.yaml")
        patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), damaged.splitlines(keepends=True), fromfile="a/" + benchmark, tofile="b/" + benchmark))
        result = await harness.tools.execute({"id": "acceptance-fault-injection", "name": "apply_patch", "arguments": {"patch": patch}})
        if result["status"] != "success":
            raise RuntimeError("真实故障注入没有完成：" + result["error"])
    elif "./examples/llm_simple_qa/testenv/missing-testenv.yaml" not in original:
        raise ValueError("当前配置不属于声明的健康状态或故障状态")
    previous_failure = harness.state.get("last_run") or {}
    if previous_failure.get("status") == "failed" and harness.state.get("failure"):
        recorded_failure = harness.runner.get_run(args.instance, previous_failure["run_id"])
        failed = {"run": harness.run_summary(recorded_failure), "verification": harness.verification_report()}
    else:
        failed = await harness.run_current()
    if failed["run"]["status"] != "failed" or failed["verification"]["passed"]:
        raise RuntimeError("缺失 testenv 故障没有产生预期真实失败")
    print(json.dumps({"stage": "故障已确认", "run_id": failed["run"]["run_id"]}, ensure_ascii=False), flush=True)
    queued = False
    compact_requests = []

    def observe(event):
        nonlocal queued
        kind, payload = event["event_type"], event["payload"]
        if kind in {"llm.started", "tool.started", "run.started", "verification.completed", "context.compacted", "agent.completed", "agent.blocked"}:
            print(json.dumps({"event": kind, "tool": payload.get("name"), "run_id": event.get("run_id"), "passed": payload.get("passed")}, ensure_ascii=False), flush=True)
        if not queued and kind == "llm.completed" and payload.get("purpose") == "observe":
            queued = True
            compact_requests.append(asyncio.get_running_loop().create_task(harness.compact()))

    harness.events.subscribe(observe)
    report = {"baseline_run_id": baseline["run_id"], "failed_run_id": failed["run"]["run_id"], "approval_policy": "auto_scoped", "evidence_type": "real_llm_ianvs"}
    report_path = settings.project_root / "workspace/ianvs-agent/acceptance/agent-repair.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        outcome = await harness.send_message("当前 Example 的原任务运行失败了。读取真实异常、README 和相关配置，诊断并修复配置问题，保持原始算法与数据，运行原命令并接受独立验证。所有修改只能在当前实例中进行。")
        if compact_requests:
            await asyncio.gather(*compact_requests)
        if outcome["status"] != "completed" or not harness.verification_report().get("repair_completed"):
            raise RuntimeError("Agent 没有完成通过独立验证的修复")
        if not harness.usage_summary()["compactions"]:
            await harness.compact()
        complete_interactions(harness.history())
        before = {"task": harness.task, "state": harness.state, "history": harness.history(), "usage": harness.usage_summary(), "diff": harness.diff()}
        await harness.close()
        restored = AgentHarness(settings)
        assert restored.task == before["task"]
        assert restored.state == before["state"]
        assert restored.history() == before["history"]
        assert restored.usage_summary() == before["usage"]
        assert restored.diff() == before["diff"]
        assert restored.verification_report()["repair_completed"]
        report.update(passed=True, repaired_run_id=restored.state["last_run_id"], session_path=restored.session_relative_path, usage=restored.usage_summary(), restored=True, verification=restored.verification_report())
        await restored.close()
    except BaseException as error:
        report.update(passed=False, error=str(error), session_path=harness.session_relative_path, state=harness.state, usage=harness.usage_summary())
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        await harness.close()
        if compact_requests:
            await asyncio.gather(*compact_requests, return_exceptions=True)
        raise
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "report": str(report_path)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="真实模型与 Ianvs 故障修复验收")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--baseline-run", default="run-0002")
    parser.add_argument("--request-timeout", type=float, help="本次真实模型请求的超时秒数")
    asyncio.run(check(parser.parse_args()))


if __name__ == "__main__":
    main()
