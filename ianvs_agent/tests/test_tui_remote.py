# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import platform
import sys
import unittest
from pathlib import Path

from textual.widgets import Button, DataTable, Select, Static, TabbedContent

from ianvs_agent.__main__ import build_wsl_command
from ianvs_agent.app.tui import IanvsAgentApp
from ianvs_agent.config import load_settings
from ianvs_agent.transport.remote import RemoteHarness


@unittest.skipUnless(os.name == "nt" and os.environ.get("IANVS_TUI_BACKEND_CONFIG"), "需要 Windows 前端及真实 WSL 会话配置")
class RemoteTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_windows_frontend_restores_wsl_session_and_closes_backend(self):
        project_root = Path(__file__).resolve().parents[2]
        config = Path(os.environ["IANVS_TUI_BACKEND_CONFIG"]).resolve()
        settings = load_settings(project_root, config, load_credentials=False)
        command = build_wsl_command(settings, config, python_executable=os.environ.get("IANVS_TUI_BACKEND_PYTHON"))
        harness = RemoteHarness(settings, command)
        self.assertIsNone(harness.process)
        self.assertFalse(harness.connected)
        app = IanvsAgentApp(settings, harness)
        output = project_root / "workspace/ianvs-agent/acceptance/windows-tui"
        output.mkdir(parents=True, exist_ok=True)
        async with app.run_test(size=(160, 50)) as pilot:
            self.assertTrue(app.query_one("#send-message", Button).disabled)
            async with asyncio.timeout(180):
                while not harness.connected or app._connection_worker is not None:
                    if harness.connection_error:
                        self.fail(harness.connection_error)
                    await pilot.pause(0.1)
            self.assertEqual(platform.system(), "Windows")
            self.assertEqual(harness.backend_info["platform"], "Linux")
            identity_details = str(app.query_one("#backend-status", Static).tooltip)
            self.assertIn(harness.backend_info["python_executable"], identity_details)
            self.assertIn(harness.backend_info["wsl_distribution"], identity_details)
            self.assertIsNotNone(harness.current_instance)
            self.assertIsNotNone(harness.session)
            self.assertGreater(app.query_one("#example-table", DataTable).row_count, 0)
            self.assertEqual(app._displayed_session_id, harness.session["session_id"])
            before = harness.usage_summary()["model_calls"]
            report = harness.verification_report()
            self.assertIs(report.get("passed"), True)
            self.assertIs(report.get("repair_completed"), True)
            self.assertFalse(report.get("stale"))
            self.assertEqual(report["run_id"], harness.current_instance["last_run_id"])
            layouts = []
            for width, height in ((160, 50), (100, 35), (80, 24)):
                await pilot.resize_terminal(width, height)
                await pilot.pause()
                layout = {}
                for selector in ("#topbar", "#backend-status", "#model-select", "#workspace", "#summary-panel", "#message-input", "#send-message"):
                    region = app.query_one(selector).region
                    self.assertGreater(region.width, 0, selector)
                    self.assertGreater(region.height, 0, selector)
                    self.assertGreaterEqual(region.x, 0, selector)
                    self.assertLessEqual(region.right, width, selector)
                    self.assertLessEqual(region.bottom, height, selector)
                    layout[selector] = dict(x=region.x, y=region.y, width=region.width, height=region.height)
                layouts.append(dict(viewport=[width, height], controls=layout))
            await pilot.resize_terminal(160, 50)
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            screenshots = []
            for view, key in (("conversation", None), ("diff", "f3"), ("usage", "f4"), ("verification", "f5")):
                if key:
                    await pilot.press(key)
                    await pilot.pause()
                self.assertEqual(tabs.active, view)
                saved = app.save_screenshot(filename=f"{harness.session['session_id']}-{view}.svg", path=str(output))
                screenshots.append(dict(view=view, path=saved))
            self.assertIn("修复完成：通过", str(app.query_one("#verification-detail", Static).render()))
            original_model = settings.selected_profile
            alternate_model = next(name for name in settings.profiles if name != original_model)
            app.query_one("#model-select", Select).value = alternate_model
            async with asyncio.timeout(20):
                while settings.selected_profile != alternate_model or app._active_action is not None:
                    await pilot.pause(0.1)
            app.query_one("#model-select", Select).value = original_model
            async with asyncio.timeout(20):
                while settings.selected_profile != original_model or app._active_action is not None:
                    await pilot.pause(0.1)
            after = harness.usage_summary()["model_calls"]
            self.assertEqual(before, after)
            evidence = dict(source="原生 Windows Python Textual App.run_test，真实 WSL 后端正式会话", frontend_platform=platform.system(), backend_info=harness.backend_info, instance_id=harness.current_instance["instance_id"], session_id=harness.session["session_id"], run_id=harness.current_instance["last_run_id"], verification=report, model_calls_before=before, model_calls_after=after, model_calls_during_export=after - before, layouts=layouts, screenshots=screenshots)
            await pilot.press("ctrl+q")
            try:
                async with asyncio.timeout(30):
                    while not app._harness_closed:
                        await asyncio.sleep(0.1)
            except TimeoutError:
                print(json.dumps(dict(exit_screen=type(app.screen).__name__, close_started=app._close_task is not None, close_done=app._close_task.done() if app._close_task else None, backend_busy=harness.busy, frontend_action=app._active_action.name if app._active_action else None, pending=[dict(id=key, method=entry[1], done=entry[0].done()) for key, entry in harness.pending.items()]), ensure_ascii=False))
                raise
            self.assertIsNone(harness.process)
            self.assertFalse(harness.connected)
            evidence["backend_closed_after_exit"] = True
            (output / "session-ui-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertTrue(app._harness_closed)
        self.assertIsNone(harness.process)


@unittest.skipUnless(os.name == "nt", "需要原生 Windows Python")
class FailedRemoteTuiTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_failed_process_leaves_frontend_responsive_and_exits(self):
        await self.check_failure_command([sys.executable, "-X", "utf8", "-m", "ianvs_agent.nonexistent_backend"])

    async def test_real_missing_executable_can_retry_and_exit(self):
        project_root = Path(__file__).resolve().parents[2]
        await self.check_failure_command([str(project_root / "workspace/ianvs-agent/nonexistent-backend.exe")])

    async def check_failure_command(self, command):
        project_root = Path(__file__).resolve().parents[2]
        settings = load_settings(project_root, load_credentials=False)
        harness = RemoteHarness(settings, command)
        app = IanvsAgentApp(settings, harness)
        async with app.run_test(size=(80, 24), notifications=True) as pilot:
            async with asyncio.timeout(20):
                while app._connection_worker is not None or not harness.connection_error:
                    await pilot.pause(0.1)
            await pilot.pause()
            self.assertFalse(harness.connected)
            self.assertTrue(harness.connection_error)
            self.assertIn("连接失败", str(app.query_one("#backend-status", Static).render()))
            self.assertTrue(app.query_one("#send-message", Button).disabled)
            self.assertTrue(app.query_one("#run-example", Button).disabled)
            self.assertFalse(app.query_one("#connect-backend", Button).disabled)
            await pilot.press("f4")
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "usage")
            await pilot.press("ctrl+r")
            async with asyncio.timeout(20):
                while app._connection_worker is not None:
                    await pilot.pause(0.1)
            self.assertFalse(harness.connected)
            await pilot.pause()
            await pilot.press("ctrl+q")
            async with asyncio.timeout(20):
                while not app._harness_closed:
                    await asyncio.sleep(0.1)
        self.assertIsNone(harness.process)
        self.assertFalse(harness.connected)


if __name__ == "__main__":
    unittest.main()
