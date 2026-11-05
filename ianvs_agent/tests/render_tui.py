# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from ianvs_agent.app.tui import IanvsAgentApp
from ianvs_agent.config import load_settings


async def render_session(settings, output, size, require_repaired=False):
    app = IanvsAgentApp(settings)
    output.mkdir(parents=True, exist_ok=True)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if app.harness.current_instance is None or app.harness.session is None:
            raise ValueError("导出界面需要已有的正式实例与会话")
        verification = app.harness.verification_report()
        if require_repaired and (verification.get("repair_completed") is not True or verification.get("passed") is not True or verification.get("stale") or app.harness.current_instance.get("needs_verification") or verification.get("run_id") != app.harness.current_instance.get("last_run_id")):
            raise ValueError("当前会话尚无独立验证确认的修复完成记录")
        session_id = app.harness.session["session_id"]
        model_calls_before = app.harness.usage_summary()["model_calls"]
        screenshots = []
        for name, key in [("conversation", None), ("diff", "f3"), ("usage", "f4"), ("verification", "f5")]:
            if key:
                await pilot.press(key)
                await pilot.pause()
            filename = f"{session_id}-{name}.svg"
            saved = app.save_screenshot(filename=filename, path=str(output))
            screenshots.append({"view": name, "path": saved})
        layout = {selector: {"x": app.query_one(selector).region.x, "y": app.query_one(selector).region.y, "width": app.query_one(selector).region.width, "height": app.query_one(selector).region.height} for selector in ("#topbar", "#sidebar", "#main-panel", "#summary-panel", "#message-input")}
        usage = app.harness.usage_summary()
        if usage["model_calls"] != model_calls_before:
            raise RuntimeError("界面导出期间发生了模型调用")
        evidence = {"source": "Textual App.run_test 与正式持久会话", "session_id": session_id, "instance_id": app.harness.current_instance["instance_id"], "run_id": app.harness.current_instance.get("last_run_id"), "model": settings.profile.model, "viewport": list(size), "layout": layout, "usage": usage, "model_calls_during_export": usage["model_calls"] - model_calls_before, "verification": verification, "screenshots": screenshots}
        (output / f"{session_id}-ui-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"session_id": session_id, "screenshots": screenshots, "repair_completed": verification.get("repair_completed")}, ensure_ascii=False))
    await app.harness.close()


def main():
    parser = argparse.ArgumentParser(description="从正式持久会话导出 Textual 界面与布局记录")
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, nargs=2, default=(160, 50))
    parser.add_argument("--require-repaired", action="store_true")
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    settings = replace(load_settings(project_root), state_root=Path(arguments.state_root).resolve())
    asyncio.run(render_session(settings, Path(arguments.output).resolve(), tuple(arguments.size), arguments.require_repaired))


if __name__ == "__main__":
    main()
