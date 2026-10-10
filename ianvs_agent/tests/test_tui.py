# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import codecs
import cProfile
import ensurepip
import difflib
import json
import sys
import tempfile
import unittest
import os
import shutil
from dataclasses import replace
from pathlib import Path

from textual.widgets import Button, DataTable, Input, LoadingIndicator, RichLog, Select, Static, TabbedContent
from textual.worker import WorkerCancelled
import yaml

from ianvs_agent.app.screens import ApprovalScreen, ToolDetailsScreen
from ianvs_agent.app.records import tool_display_status, tool_summary
from ianvs_agent.app.tui import IanvsAgentApp
from ianvs_agent.config import load_settings
from ianvs_agent.transport.remote import RemoteHarness, TransportError


class TuiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[2]
        artifacts = Path(os.environ.get("IANVS_TUI_TEST_ROOT", self.project_root / "workspace/ianvs-agent/test-artifacts"))
        artifacts.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="tui-", dir=artifacts)
        directory = Path(self.directory.name)
        source = directory / "source"
        shutil.copytree(self.project_root / "examples/llm_simple_qa", source / "examples/llm_simple_qa")
        self.settings = replace(load_settings(self.project_root), project_root=source, state_root=directory / "state")

    def tearDown(self):
        self.directory.cleanup()

    async def patch_call(self, harness, call_id):
        if not isinstance(harness, RemoteHarness):
            harness._ensure_task("检查独立 Example 配置补丁的审批与文件范围", kind="repair")
        execute = harness.execute_tool if isinstance(harness, RemoteHarness) else harness.tools.execute
        path = harness.current_instance["benchmark_path"]
        read = await execute({"id": call_id + "-read", "name": "read_file", "arguments": {"path": path}})
        self.assertEqual(read["status"], "success")
        target = Path(harness.current_instance["code_root"]) / path
        before = target.read_text(encoding="utf-8")
        name = yaml.safe_load(before)["benchmarkingjob"]["name"]
        after = before.replace(f'name: "{name}"', f'name: "{call_id}"', 1)
        self.assertNotEqual(before, after)
        source = target.read_bytes().decode("utf-8")
        patched = source.replace(f'name: "{name}"', f'name: "{call_id}"', 1)
        patch = "".join(difflib.unified_diff(source.splitlines(keepends=True), patched.splitlines(keepends=True), fromfile="a/" + path, tofile="b/" + path))
        return {"id": call_id, "name": "apply_patch", "arguments": {"patch": patch}}, target, before, after

    async def test_browsing_model_switch_and_views_use_no_model_calls(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            table = app.query_one("#example-table", DataTable)
            self.assertGreater(table.row_count, 0)
            self.assertIsNone(app.harness.current_instance)
            app.query_one("#example-filter", Input).value = "不存在的示例名称"
            await pilot.pause()
            self.assertEqual(table.row_count, 0)
            app.query_one("#example-filter", Input).value = ""
            await pilot.pause()
            self.assertGreater(table.row_count, 0)
            app.query_one("#model-select", Select).value = "glm"
            await pilot.pause()
            self.assertEqual(app.settings.profile.model, "glm-5.3")
            tabs = app.query_one("#main-tabs", TabbedContent)
            await pilot.click(tabs.get_tab("verification"))
            await pilot.pause()
            self.assertIn("当前实例尚无独立验证记录", str(app.query_one("#verification-detail", Static).render()))
            await pilot.click(tabs.get_tab("usage"))
            await pilot.pause()
            self.assertIn("当前会话累计", str(app.query_one("#usage-detail", Static).render()))
            await pilot.press("f3")
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "diff")
            await pilot.press("f4")
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "usage")
            await pilot.press("f5")
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "verification")
            await pilot.press("ctrl+k")
            await pilot.pause()
            self.assertFalse(any(event["event_type"] == "llm.started" for event in app.harness.store.read_events("events.jsonl")))
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
        await app.harness.close()

    async def test_unneeded_manual_compaction_preserves_task_result_and_status(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(100, 35)) as pilot:
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            app.harness._ensure_task("等待合法取得数据后继续原任务", kind="repair")
            await app.harness._finish("blocked", "当前任务因缺少数据停止，诊断记录已保存")
            await pilot.pause()
            state = copy.deepcopy(app.harness.state)
            task = copy.deepcopy(app.harness.task)
            status = app._last_status
            result = await app.harness.compact()
            await pilot.pause()
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(app.harness.state, state)
            self.assertEqual(app.harness.task, task)
            self.assertEqual(app._last_status, status)
            self.assertEqual(app.query_one("#activity-indicator").styles.visibility, "hidden")
            events = app.harness.store.read_events("events.jsonl")
            skipped = [event for event in events if event["event_type"] == "context.compaction_skipped"]
            self.assertEqual(len(skipped), 1)
            self.assertIn("无需压缩", app.format_event(skipped[0])[0])
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
        await app.harness.close()

    async def test_controls_are_accessible_at_supported_terminal_sizes(self):
        for size in ((160, 50), (100, 35), (80, 24)):
            with self.subTest(size=size):
                app = IanvsAgentApp(self.settings)
                async with app.run_test(size=size) as pilot:
                    await pilot.pause()
                    width, height = size
                    for selector in ("#topbar", "#workspace", "#summary-panel", "#message-input", "#send-message", "#activity-row", "#task-status", "#activity-time"):
                        region = app.query_one(selector).region
                        self.assertGreater(region.width, 0, selector)
                        self.assertGreater(region.height, 0, selector)
                        self.assertGreaterEqual(region.x, 0, selector)
                        self.assertLessEqual(region.right, width, selector)
                        self.assertLessEqual(region.bottom, height, selector)
                    self.assertLess(app.query_one("#sidebar").region.width, app.query_one("#main-panel").region.width)
                    self.assertLessEqual(app.query_one("#message-input").region.bottom, app.query_one("#summary-panel").region.y)
                    self.assertIn("上下文占用", str(app.query_one("#context-stat", Static).render()))
                    self.assertIn("输入任务", app.query_one("#message-input", Input).placeholder)
                    self.assertIsInstance(app.query_one("#activity-indicator"), LoadingIndicator)
                    self.assertEqual(app.query_one("#activity-indicator").styles.visibility, "hidden")
                    self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
                await app.harness.close()

    async def test_hidden_logs_reflow_after_tab_changes_and_terminal_resize(self):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "print('\\x1b[31mTestCase failed: missing evaluation dataset\\x1b[0m'); print('prepare 10%\\rprepare 100%')",
            stdout=asyncio.subprocess.PIPE,
        )
        output, _ = await process.communicate()
        self.assertEqual(process.returncode, 0)
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            tabs.active = "logs"
            await pilot.pause()
            tabs.active = "conversation"
            await pilot.pause()
            app.harness.events.emit("run.output", {"output": output.decode("utf-8")})
            await pilot.pause()
            tabs.active = "logs"
            await pilot.pause()
            log = app.query_one("#run-log", RichLog)
            rendered = "\n".join(line.text for line in log.lines)
            self.assertIn("TestCase failed: missing evaluation dataset", rendered)
            self.assertIn("prepare 100%", rendered)
            self.assertNotIn("\x1b", rendered)
            self.assertNotIn("prepare 10%", rendered)
            self.assertLess(len(log.lines), 12)
            for size in ((80, 24), (180, 55), (100, 35), (140, 45)):
                await pilot.resize_terminal(*size)
                self.assertGreater(log.scrollable_content_region.width, 0)
                self.assertTrue(all(line.cell_length <= log.scrollable_content_region.width for line in log.lines))
                self.assertIn("TestCase", "\n".join(line.text for line in log.lines))
            tabs.active = "logs"
            app.write_conversation("assistant", "诊断结果：缺少评估数据，请检查数据路径和准备步骤。")
            await pilot.pause()
            tabs.active = "conversation"
            await pilot.pause()
            self.assertIn("诊断结果：缺少评估数据", "\n".join(line.text for line in app.query_one("#conversation-log", RichLog).lines))
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)

    async def test_subprocess_progress_preserves_lines_and_coalesces_status_refreshes(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)
            tabs.active = "logs"
            await pilot.pause()
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-c",
                "import sys,time; sys.stdout.write('\\x1b[31mTestCase diagnostic'); sys.stdout.flush(); time.sleep(.05); "
                "sys.stdout.write(' evidence\\x1b[0m\\r\\n'); sys.stdout.flush(); "
                "[(sys.stdout.write('prepare '+str(i)+'%\\r'),sys.stdout.flush(),time.sleep(.006)) for i in range(101)]; "
                "sys.stdin.readline(); sys.stdout.write('prepare complete\\r\\nfinal unterminated evidence'); sys.stdout.flush()",
                stdout=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
            )
            raw = bytearray()
            chunks = 0
            profile = cProfile.Profile()

            async def read_output():
                nonlocal chunks
                decoder = codecs.getincrementaldecoder("utf-8")()
                while data := await process.stdout.read(7):
                    raw.extend(data)
                    chunks += 1
                    app.harness.events.emit("run.output", {"channel": "stdout", "text": decoder.decode(data)}, run_id="subprocess-output-check")
                self.assertEqual(decoder.decode(b"", final=True), "")

            profile.enable()
            reader = asyncio.create_task(read_output())
            await asyncio.sleep(.12)
            await pilot.press("f6")
            self.assertEqual(tabs.active, "tools")
            self.assertIsNone(process.returncode)
            await pilot.press("f4")
            self.assertEqual(tabs.active, "usage")
            process.stdin.write(b"continue\n")
            await process.stdin.drain()
            await reader
            await process.wait()
            app.harness.events.emit("run.completed", {"exit_code": process.returncode}, run_id="subprocess-output-check")
            await pilot.pause(.3)
            profile.disable()
            self.assertEqual(process.returncode, 0)
            refreshes = sum(entry.callcount for entry in profile.getstats() if getattr(entry.code, "co_name", None) == "refresh_status")
            self.assertGreater(chunks, 100)
            self.assertLess(refreshes, chunks // 4)
            tabs.active = "logs"
            await pilot.pause()
            log = app.query_one("#run-log", RichLog)
            rendered = "\n".join(line.text for line in log.lines)
            self.assertIn("TestCase diagnostic evidence", rendered)
            self.assertIn("prepare complete", rendered)
            self.assertIn("final unterminated evidence", rendered)
            self.assertNotIn("prepare 20%", rendered)
            self.assertNotIn("\x1b", rendered)
            self.assertLess(len(log._entries), 10)
            self.assertFalse(app.query_one("#run-progress", Static).display)
            self.assertIn(b"prepare 20%\r", raw)
            events = app.harness.store.read_events("events.jsonl")
            self.assertEqual("".join(event["payload"]["text"] for event in events if event["event_type"] == "run.output"), raw.decode("utf-8"))
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)

    async def test_continue_entry_uses_original_failed_task_and_survives_provider_configuration_error(self):
        profiles = {name: replace(profile, api_key_env="IANVS_TUI_TEST_UNCONFIGURED_KEY") for name, profile in self.settings.profiles.items()}
        self.assertNotIn("IANVS_TUI_TEST_UNCONFIGURED_KEY", os.environ)
        app = IanvsAgentApp(replace(self.settings, profiles=profiles))
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            button = app.query_one("#resume-task", Button)
            self.assertTrue(button.disabled)
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            with self.assertRaisesRegex(RuntimeError, "环境尚未准备"):
                await app.harness.run_current()
            await pilot.pause()
            app.harness.task["acceptance"]["required_metrics"] = ["acc"]
            original = copy.deepcopy(app.harness.task)
            app.harness._save_session()
            app.refresh_status()
            self.assertFalse(button.disabled)
            app.query_one("#sidebar").scroll_end(animate=False)
            await pilot.pause()
            self.assertTrue(await pilot.click(button))

            async def wait_for_resume(count):
                while app.harness.state.get("resume_count") != count or app._active_action is not None:
                    await asyncio.sleep(.02)

            await asyncio.wait_for(wait_for_resume(1), timeout=10)
            await pilot.pause()
            self.assertEqual(app.harness.task, original)
            self.assertEqual(app.harness.state["resume_count"], 1)
            self.assertEqual(app.harness.state["status"], "failed")
            self.assertIn("IANVS_TUI_TEST_UNCONFIGURED_KEY", app.harness.state["error"])
            self.assertFalse(button.disabled)
            await pilot.press("f7")
            await asyncio.wait_for(wait_for_resume(2), timeout=10)
            await pilot.pause()
            self.assertEqual(app.harness.state["resume_count"], 2)
            self.assertEqual(app.harness.task, original)
            self.assertIsNone(app.harness.provider.client)
        restored = IanvsAgentApp(replace(self.settings, profiles=profiles))
        async with restored.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            self.assertEqual(restored.harness.task, original)
            self.assertFalse(restored.query_one("#resume-task", Button).disabled)

    @unittest.skipIf(os.name == "nt", "独立环境和验证通过 Linux/WSL 后端执行")
    async def test_failed_example_and_verification_keep_formal_status_and_full_evidence(self):
        validator = self.settings.project_root / ".github/workflows/validator"
        shutil.copytree(self.project_root / ".github/workflows/validator", validator)
        wheels = self.settings.project_root / "ui-check-wheels"
        wheels.mkdir()
        pip_wheel, = (Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl")
        shutil.copy2(pip_wheel, wheels / pip_wheel.name)
        requirements = self.settings.project_root / "ui-check-requirements.txt"
        requirements.write_text("--no-index\n--find-links ./ui-check-wheels\npip>=0\n", encoding="utf-8")
        settings = replace(self.settings, python_executable=sys.executable, approval_policy="auto_scoped", tool_output_chars=128)
        app = IanvsAgentApp(settings)
        async with app.run_test(size=(140, 45)) as pilot:
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            app.harness._ensure_task("核查独立环境下原始 Example 的运行与验证结果", kind="repair")
            plan = await app.harness.environment.plan_install(app.harness.current_instance, requirements_files=[requirements.name], packages=[], modules=["json"])
            environment = await app.harness.environment.install(app.harness.current_instance, plan["plan_id"])
            self.assertEqual(environment["status"], "ready")
            app.harness.current_instance = app.harness.instances.get(app.harness.current_instance["instance_id"])
            run = await app.harness.tools.execute({"id": "tui-failed-example", "name": "run_example", "arguments": {}})
            await pilot.pause()
            self.assertEqual(run["status"], "success")
            self.assertTrue(run["truncated"])
            self.assertEqual(tool_display_status(run), "error")
            self.assertIn("原任务执行失败", tool_summary(run))
            self.assertIn("独立验证未通过", tool_summary(run))
            self.assertFalse(run["outcome"]["verification_passed"])
            data = app.harness.store.read(run["full_output_ref"])
            self.assertNotEqual(data["run"]["exit_code"], 0)
            self.assertIn("No module named", app.harness.store.path(data["run"]["stderr_ref"]).read_text(encoding="utf-8"))
            self.assertFalse(data["verification"]["passed"])
            verify = await app.harness.tools.execute({"id": "tui-failed-verification", "name": "verify_run", "arguments": {"run_id": run["outcome"]["run_id"]}})
            await pilot.pause()
            self.assertEqual(verify["status"], "success")
            self.assertTrue(verify["truncated"])
            self.assertEqual(tool_display_status(verify), "error")
            self.assertIn("独立验证未通过", tool_summary(verify))
            await pilot.press("f6")
            await pilot.pause()
            table = app.query_one("#tool-table", DataTable)
            self.assertIn("error", table.get_cell("tui-failed-example", "status").plain)
            app._selected_tool_id = "tui-failed-example"
            app.show_tool_details()
            await pilot.pause()
            self.assertIn("工具调用状态：success", str(app.screen.query_one("#tool-result-status", Static).render()))
            self.assertIn("结果显示状态：error", str(app.screen.query_one("#tool-result-status", Static).render()))
            self.assertIn(run["full_output_ref"], str(app.screen.query_one("#tool-output-reference", Static).render()))
            await pilot.press("escape")
            legacy = {key: value for key, value in run.items() if key != "outcome"}
            self.assertEqual(tool_display_status(legacy), "unconfirmed")
            self.assertIn("结果待核查", tool_summary(legacy))
            self.assertNotIn("修复完成", tool_summary(legacy))
            app.write_tool_row(legacy)
            self.assertIn("结果待核查", table.get_cell("tui-failed-example", "status").plain)
            app.push_screen(ToolDetailsScreen(legacy))
            await pilot.pause()
            self.assertIn("工具调用状态：success", str(app.screen.query_one("#tool-result-status", Static).render()))
            self.assertIn("结果显示状态：unconfirmed", str(app.screen.query_one("#tool-result-status", Static).render()))
            self.assertIn(run["full_output_ref"], str(app.screen.query_one("#tool-output-reference", Static).render()))
            await pilot.press("escape")
            original = app.harness.store.read_events(app.harness.session_relative_path + "/tools.jsonl")
            app.restore_conversation()
            app.restore_conversation()
            self.assertEqual(app._tool_records["tui-failed-example"]["status"], "success")
            self.assertIn("error", table.get_cell("tui-failed-example", "status").plain)
            self.assertIn("独立验证未通过", table.get_cell("tui-failed-verification", "summary").plain)
            self.assertEqual(app.harness.store.read_events(app.harness.session_relative_path + "/tools.jsonl"), original)
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)

    async def test_environment_preparation_displays_scope_after_a_failed_run_and_restores(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            with self.assertRaises(RuntimeError):
                await app.harness.run_current()
            await pilot.pause()
            run_id = app.harness.current_instance["last_run_id"]
            app.start_action(lambda: app.harness._action(app.harness.environment.prepare(app.harness.current_instance, install=False)), "正在准备独立运行环境")
            await app.workers.wait_for_complete()
            await pilot.pause()
            environment = app.harness.current_instance["environment"]
            self.assertEqual(environment["status"], "created")
            self.assertTrue(Path(environment["python_executable"]).is_file())
            text = "\n".join(line.text for line in app.query_one("#conversation-log", RichLog).lines)
            self.assertIn("独立 Python 环境已创建", text)
            self.assertIn("尚未安装依赖", text)
            self.assertIn(run_id, text)
            self.assertIn("准备后尚未重新运行", text)
            self.assertIn("独立 Python 环境已创建", str(app.query_one("#task-status", Static).render()))
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
        restored = IanvsAgentApp(self.settings)
        async with restored.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            for _ in range(2):
                restored.restore_conversation()
                await pilot.pause()
                text = "\n".join(line.text for line in restored.query_one("#conversation-log", RichLog).lines)
                self.assertEqual(text.count("独立 Python 环境已创建"), 1)
                self.assertIn("准备后尚未重新运行", text)
            self.assertEqual(restored.harness.usage_summary()["model_calls"], 0)

    @unittest.skipIf(os.name == "nt", "真实失败命令采用 Linux false 程序")
    async def test_environment_failure_shows_real_step_and_log_reference(self):
        settings = replace(self.settings, python_executable="/usr/bin/false")
        app = IanvsAgentApp(settings)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("ctrl+e")
            await app.workers.wait_for_complete()
            await pilot.pause()
            environment = app.harness.current_instance["environment"]
            self.assertEqual(environment["status"], "failed")
            self.assertNotEqual(environment["failed_step"]["exit_code"], 0)
            reference = environment["failed_step"]["log_ref"]
            self.assertTrue(Path(reference).is_file())
            text = "\n".join(line.text for line in app.query_one("#conversation-log", RichLog).lines)
            self.assertIn("环境准备失败", text)
            self.assertIn("失败日志：", text)
            self.assertIn("create.log", text)
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)

    async def test_instance_creation_and_session_recovery_are_real(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            self.assertIsNotNone(app.selected_example)
            await pilot.press("ctrl+n")
            await pilot.press("f4")
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "usage")
            await app.workers.wait_for_complete()
            await pilot.pause()
            instance = app.harness.current_instance
            self.assertIsNotNone(instance)
            self.assertTrue(Path(instance["code_root"]).is_dir())
            self.assertTrue((Path(instance["code_root"]) / instance["benchmark_path"]).is_file())
            self.assertEqual(app.query_one("#instance-select", Select).value, instance["instance_id"])
            self.assertEqual(app._displayed_session_id, app.harness.session["session_id"])
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
            instance_id = instance["instance_id"]
            result = await app.harness.tools.execute({"id": "tui-inspect-configuration", "name": "inspect_configuration", "arguments": {}})
            self.assertEqual(result["status"], "success")
            await pilot.pause()
            self.assertEqual(app.query_one("#tool-table", DataTable).row_count, 1)
            self.assertEqual(app._tool_records["tui-inspect-configuration"]["status"], "success")
        await app.harness.close()
        app.harness.store.path("last_session.json").unlink()
        restored = IanvsAgentApp(self.settings)
        async with restored.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            self.assertIsNone(restored.harness.current_instance)
            restored.query_one("#instance-select", Select).value = instance_id
            await pilot.press("f4")
            self.assertEqual(restored.query_one("#main-tabs", TabbedContent).active, "usage")
            await restored.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(restored.harness.current_instance["instance_id"], instance_id)
            self.assertEqual(restored.harness.usage_summary()["model_calls"], 0)
            rendered = "\n".join(line.text for line in restored.query_one("#conversation-log").lines)
            self.assertNotIn("inspect_configuration", rendered)
            self.assertNotIn("工具执行完成", rendered)
            tools = restored.query_one("#tool-table", DataTable)
            self.assertEqual(tools.row_count, 1)
            self.assertIn("inspect_configuration", str(tools.get_row("tui-inspect-configuration")[0]))
            await pilot.press("f6")
            await pilot.pause()
            tools.focus()
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(restored.screen, ToolDetailsScreen)
            self.assertIn("tools.jsonl", str(restored.screen.query_one("#tool-output-reference", Static).render()))
            await pilot.press("escape")
            await pilot.pause()
            restored.restore_conversation()
            self.assertEqual(tools.row_count, 1)
            await pilot.press("ctrl+q")
            await pilot.pause()
            saved = restored.harness.store.read(restored.harness.session_relative_path + "/session.json")
            self.assertEqual(saved["instance_id"], instance_id)
        await restored.harness.close()

    async def test_real_tool_approval_and_rejection(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.resize_terminal(80, 24)
            call, target, before, after = await self.patch_call(app.harness, "tui-approved-patch")
            approved = app.run_worker(app.harness.tools.execute(call), name="真实补丁审批检查", exit_on_error=False)
            async with asyncio.timeout(20):
                while not isinstance(app.screen, ApprovalScreen):
                    await pilot.pause(0.1)
            self.assertIsInstance(app.screen, ApprovalScreen)
            self.assertEqual(app.query_one("#activity-indicator").styles.visibility, "visible")
            self.assertIn("等待操作审批", str(app.query_one("#task-status", Static).render()))
            initial_time = str(app.query_one("#activity-time", Static).render())
            await pilot.pause(1.1)
            self.assertNotEqual(str(app.query_one("#activity-time", Static).render()), initial_time)
            self.assertEqual(app._tool_records["tui-approved-patch"]["status"], "awaiting_approval")
            self.assertEqual(target.read_text(encoding="utf-8"), before)
            for selector in ("#approve", "#reject"):
                region = app.screen.query_one(selector).region
                self.assertGreater(region.width, 0)
                self.assertGreater(region.height, 0)
                self.assertGreaterEqual(region.y, 0)
                self.assertLessEqual(region.bottom, 24)
                self.assertLessEqual(region.right, 80)
            self.assertTrue(await pilot.click("#approve"))
            async with asyncio.timeout(20):
                result = await approved.wait()
            await pilot.pause()
            self.assertEqual(result["status"], "success")
            self.assertEqual(target.read_text(encoding="utf-8"), after)
            self.assertEqual(app.query_one("#tool-table", DataTable).row_count, 2)
            self.assertEqual(app._tool_records["tui-approved-patch"]["status"], "success")
            self.assertEqual(app._tool_records["tui-approved-patch"]["duration_seconds"], result["duration_seconds"])
            call, target, before, _ = await self.patch_call(app.harness, "tui-denied-patch")
            denied = app.run_worker(app.harness.tools.execute(call), name="拒绝操作检查", exit_on_error=False)
            async with asyncio.timeout(20):
                while not isinstance(app.screen, ApprovalScreen):
                    await pilot.pause(0.1)
            self.assertIsInstance(app.screen, ApprovalScreen)
            await pilot.press("escape")
            rejected = await denied.wait()
            await pilot.pause()
            self.assertEqual(rejected["status"], "denied")
            self.assertEqual(target.read_text(encoding="utf-8"), before)
            self.assertEqual(app.query_one("#tool-table", DataTable).row_count, 4)
            self.assertEqual(app._tool_records["tui-denied-patch"]["status"], "denied")
            self.assertEqual(app.query_one("#activity-indicator").styles.visibility, "hidden")
            conversation = "\n".join(line.text for line in app.query_one("#conversation-log").lines)
            self.assertNotIn('"stdout"', conversation)
            self.assertNotIn("工具执行完成", conversation)
            decisions = [event for event in app.harness.store.read_events(app.harness.session_relative_path + "/events.jsonl") if event["event_type"] == "approval.decided"]
            self.assertEqual([event["payload"]["approved"] for event in decisions], [True, False])
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
        await app.harness.close()

    async def test_compaction_queue_and_cancellation_during_real_approval(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            call, target, before, _ = await self.patch_call(app.harness, "tui-cancel-patch")
            app.start_action(lambda: app.harness._action(app.harness.tools.execute(call)), "执行受控工具")
            await pilot.pause()
            self.assertTrue(app.harness.busy)
            self.assertIsInstance(app.screen, ApprovalScreen)
            await pilot.press("ctrl+k")
            await pilot.pause()
            events = app.harness.store.read_events(app.harness.session_relative_path + "/events.jsonl")
            self.assertTrue(any(event["event_type"] == "context.compaction_queued" for event in events))
            self.assertTrue(app.harness._compact_requested)
            operation = app._active_action
            await pilot.press("ctrl+x")
            with self.assertRaises(WorkerCancelled):
                await operation.wait()
            await pilot.pause()
            self.assertFalse(app.harness.busy)
            self.assertFalse(app.harness.tools.pending)
            self.assertNotIsInstance(app.screen, ApprovalScreen)
            self.assertEqual(app._tool_records["tui-cancel-patch"]["status"], "cancelled")
            self.assertEqual(target.read_text(encoding="utf-8"), before)
            self.assertEqual(app.query_one("#activity-indicator").styles.visibility, "hidden")
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
        await app.harness.close()

    async def test_environment_approval_displays_versions_reference_and_full_commands(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(100, 35)) as pilot:
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            operation = app.run_worker(app.harness.tools.execute({"id": "tui-environment-plan", "name": "prepare_instance", "arguments": {}}), name="环境准备计划审批检查", exit_on_error=False)
            async with asyncio.timeout(30):
                while not isinstance(app.screen, ApprovalScreen):
                    await pilot.pause(0.1)
            await pilot.pause()
            screen = app.screen
            request = copy.deepcopy(screen.request)
            resolved = request["resolved_operation"]
            displayed = json.loads(screen.query_one("#approval-operation", Static).content.code)
            self.assertGreater(len(resolved["file_versions"]), 0)
            self.assertEqual(displayed["file_versions"], {"count": len(resolved["file_versions"]), "snapshot_fingerprint": resolved["snapshot_fingerprint"], "evidence_ref": resolved["evidence_ref"]})
            self.assertEqual({key: value for key, value in displayed.items() if key != "file_versions"}, {key: value for key, value in resolved.items() if key != "file_versions"})
            self.assertEqual(screen.request, request)
            formal = app.harness.store.read(resolved["evidence_ref"])
            self.assertEqual(formal, resolved)
            await pilot.press("escape")
            self.assertEqual((await operation.wait())["status"], "denied")
            self.assertEqual(app.harness.store.read(resolved["evidence_ref"]), formal)
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
        await app.harness.close()

    async def test_recorded_install_plan_approval_preserves_dependency_details(self):
        capture = os.environ.get("IANVS_APPROVAL_PLAN_CAPTURE")
        if not capture:
            self.skipTest("需要真实安装计划 IANVS_APPROVAL_PLAN_CAPTURE")
        source = Path(capture)
        original = source.read_bytes()
        resolved = json.loads(original)
        self.assertGreater(len(resolved["file_versions"]), 0)
        request = {"tool_name": "install_dependencies", "approval_id": "display-" + resolved["plan_id"], "arguments": {"action": "apply", "plan_id": resolved["plan_id"]}, "resolved_operation": resolved}
        before = copy.deepcopy(request)
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(100, 35)) as pilot:
            screen = ApprovalScreen(request)
            await app.push_screen(screen)
            await pilot.pause()
            displayed = json.loads(screen.query_one("#approval-operation", Static).content.code)
            self.assertEqual(displayed["file_versions"], {"count": len(resolved["file_versions"]), "snapshot_fingerprint": resolved["snapshot_fingerprint"], "evidence_ref": resolved["evidence_ref"]})
            for field in ("commands", "request", "requirements_file_versions", "resolution_ref", "constraints_ref"):
                self.assertEqual(displayed[field], resolved[field])
            self.assertEqual({key: value for key, value in displayed.items() if key != "file_versions"}, {key: value for key, value in resolved.items() if key != "file_versions"})
            self.assertEqual(request, before)
            self.assertEqual(source.read_bytes(), original)
            await pilot.press("escape")
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
        await app.harness.close()

    async def test_real_file_result_and_failure_have_separate_details(self):
        self.settings.tool_output_chars = 80
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            path = app.harness.current_instance["readme_path"]
            result = await app.harness.tools.execute({"id": "tui-read-readme", "name": "read_file", "arguments": {"path": path}})
            await pilot.pause()
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["truncated"])
            self.assertTrue(app.harness.store.path(result["full_output_ref"]).is_file())
            self.assertEqual(app._tool_records["tui-read-readme"]["full_output_ref"], result["full_output_ref"])
            await pilot.press("f6")
            await pilot.pause()
            table = app.query_one("#tool-table", DataTable)
            table.focus()
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, ToolDetailsScreen)
            self.assertIn(result["full_output_ref"], str(app.screen.query_one("#tool-output-reference", Static).render()))
            self.assertIn("README", app.screen.query_one("#tool-output-preview", Static).content.code)
            await pilot.press("escape")
            missing = "examples/llm_simple_qa/does-not-exist.txt"
            app.settings.tool_output_chars = 16000
            absent = await app.harness.tools.execute({"id": "tui-read-missing", "name": "read_file", "arguments": {"path": missing}})
            self.assertEqual(absent["status"], "success")
            absent_data = app.harness.store.read(absent["full_output_ref"]) if absent.get("truncated") else absent["data"]
            self.assertFalse(absent_data["exists"])
            self.assertEqual(absent_data["file_version"], "absent")
            self.assertIn("文件尚不存在，已记录 absent 版本", tool_summary(absent))
            self.assertNotIn("1–0", tool_summary(absent))
            app.settings.tool_output_chars = 80
            failed = await app.harness.tools.execute({"id": "tui-read-outside", "name": "read_file", "arguments": {"path": "../does-not-exist.txt"}})
            await pilot.pause()
            self.assertEqual(failed["status"], "error")
            self.assertEqual(table.row_count, 3)
            table.move_cursor(row=2)
            table.focus()
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, ToolDetailsScreen)
            self.assertIn("超出当前实例", str(app.screen.query_one("#tool-error", Static).render()))
            await pilot.press("escape")
            app.restore_conversation()
            app.restore_conversation()
            self.assertEqual(table.row_count, 3)
            self.assertEqual(app._tool_records["tui-read-readme"]["full_output_ref"], result["full_output_ref"])
            rendered = "\n".join(line.text for line in app.query_one("#conversation-log").lines)
            self.assertNotIn("does-not-exist.txt", rendered)
            self.assertNotIn("结果：", rendered)
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
        await app.harness.close()

    async def test_cancel_error_clears_activity_and_allows_approval_after_reconnect(self):
        config = Path(self.directory.name) / "backend.yaml"
        public = self.settings.public_dict()
        public.pop("project_root")
        config.write_text(yaml.safe_dump(public), encoding="utf-8")
        harness = RemoteHarness(self.settings, [sys.executable, "-u", "-m", "ianvs_agent.transport.server", "--project-root", str(self.settings.project_root), "--config", str(config)])
        app = IanvsAgentApp(self.settings, harness)
        async with app.run_test(size=(80, 24)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertTrue(harness.connected)
            await pilot.press("ctrl+n")
            await app.workers.wait_for_complete()
            await pilot.pause()
            instance_id = harness.current_instance["instance_id"]
            call, target, before, _ = await self.patch_call(harness, "before-disconnect")
            app.start_action(lambda: harness.execute_tool(call), "检查操作审批")
            async with asyncio.timeout(20):
                while not isinstance(app.screen, ApprovalScreen):
                    await pilot.pause(0.1)
            await harness.close()
            await pilot.pause()
            self.assertFalse(harness.connected)
            with self.assertRaises(TransportError):
                await app.cancel_action()
            await pilot.pause()
            self.assertFalse(app._cancelling)
            self.assertFalse(app._approval_open)
            self.assertNotIsInstance(app.screen, ApprovalScreen)
            self.assertEqual(app.query_one("#activity-indicator").styles.visibility, "hidden")
            self.assertEqual(target.read_text(encoding="utf-8"), before)
            app.action_connect_backend()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertTrue(harness.connected)
            self.assertEqual(harness.current_instance["instance_id"], instance_id)
            call, target, before, _ = await self.patch_call(harness, "after-reconnect")
            app.start_action(lambda: harness.execute_tool(call), "恢复后的操作审批")
            async with asyncio.timeout(20):
                while not isinstance(app.screen, ApprovalScreen):
                    await pilot.pause(0.1)
            self.assertEqual(app._tool_records["after-reconnect"]["status"], "awaiting_approval")
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(app._tool_records["after-reconnect"]["status"], "denied")
            self.assertEqual(target.read_text(encoding="utf-8"), before)
            self.assertEqual(harness.usage_summary()["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
