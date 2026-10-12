# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml
from textual.widgets import Button, DataTable, Input, RichLog, Select, Static, TabbedContent

from ianvs_agent.app.screens import DeleteInstanceScreen
from ianvs_agent.app.tui import IanvsAgentApp
from ianvs_agent.config import load_settings
from ianvs_agent.transport.remote import RemoteHarness


class DeleteInstanceTuiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[2]
        artifacts = Path(os.environ.get("IANVS_TUI_TEST_ROOT", self.project_root / "workspace/ianvs-agent/test-artifacts"))
        artifacts.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="tui-delete-", dir=artifacts)
        directory = Path(self.directory.name)
        source = directory / "source"
        shutil.copytree(self.project_root / "examples/llm_simple_qa", source / "examples/llm_simple_qa")
        self.settings = replace(load_settings(self.project_root), project_root=source, state_root=directory / "state")

    def tearDown(self):
        self.directory.cleanup()

    async def open_delete(self, app, pilot):
        await app.workers.wait_for_complete()
        app.refresh_instances()
        app.refresh_status()
        app.query_one("#sidebar").scroll_end(animate=False)
        button = app.query_one("#delete-instance", Button)
        button.scroll_visible(animate=False)
        await pilot.pause()
        self.assertFalse(button.disabled)
        self.assertTrue(await pilot.click(button))
        await pilot.pause()
        self.assertIsInstance(app.screen, DeleteInstanceScreen)

    async def test_cancel_is_default_and_escape_preserves_real_instance(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            self.assertTrue(app.query_one("#delete-instance", Button).disabled)
            await pilot.click("#example-table", offset=(2, 2))
            await pilot.pause()
            self.assertIsNone(app.harness.current_instance)
            await pilot.click("#create-instance")
            await app.workers.wait_for_complete()
            await pilot.pause()
            instance = app.harness.current_instance
            self.assertIsNotNone(instance)
            instance_root = self.settings.state_root / "instances" / instance["instance_id"]
            before = (instance_root / "instance.json").read_bytes()
            await self.open_delete(app, pilot)
            screen = app.screen
            self.assertEqual(screen.instance_id, instance["instance_id"])
            self.assertEqual(app.focused.id, "cancel-delete-instance")
            identity = str(screen.query_one("#delete-instance-identity", Static).render())
            self.assertIn(instance["instance_id"], identity)
            description = str(screen.query_one("#delete-instance-description", Static).render())
            for required in ("源码副本", "Python", "缓存", "对话", "补丁", "运行记录", "无法恢复"):
                self.assertIn(required, description)
            await pilot.press("enter")
            await pilot.pause()
            self.assertNotIsInstance(app.screen, DeleteInstanceScreen)
            self.assertEqual((instance_root / "instance.json").read_bytes(), before)
            await self.open_delete(app, pilot)
            await pilot.press("escape")
            await pilot.pause()
            self.assertEqual((instance_root / "instance.json").read_bytes(), before)
            self.assertEqual(app.harness.current_instance["instance_id"], instance["instance_id"])
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)

    async def test_confirm_deletes_instance_and_clears_all_views(self):
        await self.check_confirmed_deletion(remote=False)

    async def test_real_backend_rpc_deletion_clears_frontend_snapshot(self):
        await self.check_confirmed_deletion(remote=True)

    async def check_confirmed_deletion(self, *, remote):
        harness = None
        if remote:
            config = Path(self.directory.name) / "backend.yaml"
            public = self.settings.public_dict()
            public.pop("project_root")
            config.write_text(yaml.safe_dump(public), encoding="utf-8")
            harness = RemoteHarness(self.settings, [sys.executable, "-u", "-m", "ianvs_agent.transport.server", "--project-root", str(self.settings.project_root), "--config", str(config)])
        app = IanvsAgentApp(self.settings, harness)
        original = self.settings.project_root / "examples/llm_simple_qa/benchmarkingjob.yaml"
        original_bytes = original.read_bytes()
        async with app.run_test(size=(140, 45)) as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            survivor = app.harness.current_instance
            survivor_root = Path(survivor["code_root"])
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            instance = app.harness.current_instance
            instance_id = instance["instance_id"]
            instance_root = self.settings.state_root / "instances" / instance_id
            execute = app.harness.execute_tool if remote else app.harness.tools.execute
            result = await execute({"id": "delete-ui-read", "name": "read_file", "arguments": {"path": instance["benchmark_path"]}})
            self.assertEqual(result["status"], "success")
            await pilot.pause()
            self.assertEqual(app.query_one("#tool-table", DataTable).row_count, 1)
            app.action_show_diff()
            await pilot.pause()
            app.action_show_usage()
            await pilot.pause()
            app.action_show_verification()
            await pilot.pause()
            app.write_conversation("user", "当前实例的历史任务")
            app.query_one("#message-input", Input).value = "当前实例尚未发送的指令"
            await self.open_delete(app, pilot)
            self.assertTrue(await pilot.click("#confirm-delete-instance"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertIsNone(app.harness.current_instance)
            self.assertIsNone(app.harness.session)
            self.assertEqual(app.harness.history(), [])
            self.assertFalse(instance_root.exists())
            self.assertTrue(survivor_root.is_dir())
            self.assertEqual(original.read_bytes(), original_bytes)
            self.assertEqual([item["instance_id"] for item in app.harness.instances.list_instances()], [survivor["instance_id"]])
            self.assertIs(app.query_one("#instance-select", Select).value, Select.BLANK)
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "conversation")
            self.assertEqual(app.query_one("#message-input", Input).value, "")
            self.assertEqual(app.query_one("#tool-table", DataTable).row_count, 0)
            self.assertEqual(app._tool_records, {})
            self.assertIsNone(app._displayed_session_id)
            self.assertEqual(app._stream_text, "")
            self.assertFalse(app.query_one("#run-progress", Static).display)
            self.assertIn("尚未运行", str(app.query_one("#run-status", Static).render()))
            conversation = "\n".join(line.text for line in app.query_one("#conversation-log", RichLog).lines)
            self.assertNotIn("历史任务", conversation)
            self.assertIn("选择示例并创建实例", conversation)
            await pilot.press("f3")
            await pilot.pause()
            self.assertIn("尚无代码修改", "\n".join(line.text for line in app.query_one("#diff-log", RichLog).lines))
            self.assertIn("尚无独立验证记录", str(app.query_one("#verification-detail", Static).render()))
            self.assertIn("尚未创建任务", str(app.query_one("#usage-detail", Static).render()))
            for selector in ("#delete-instance", "#prepare-environment", "#run-example", "#resume-task", "#send-message"):
                self.assertTrue(app.query_one(selector, Button).disabled, selector)
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)
            app.query_one("#instance-select", Select).value = survivor["instance_id"]
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(app.harness.current_instance["instance_id"], survivor["instance_id"])
            self.assertFalse(app.query_one("#delete-instance", Button).disabled)

    async def test_confirmation_cannot_delete_a_different_selected_instance(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(100, 35)) as pilot:
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            first = app.harness.current_instance
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            second = app.harness.current_instance
            await pilot.pause()
            await self.open_delete(app, pilot)
            self.assertEqual(app.screen.instance_id, second["instance_id"])
            await app.harness.restore_instance(first["instance_id"])
            await pilot.pause()
            self.assertTrue(await pilot.click("#confirm-delete-instance"))
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(app.harness.current_instance["instance_id"], first["instance_id"])
            self.assertTrue(Path(first["code_root"]).is_dir())
            self.assertTrue(Path(second["code_root"]).is_dir())
            self.assertEqual(len(app.harness.instances.list_instances()), 2)
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)

    async def test_real_running_action_disables_deletion_until_cancelled(self):
        app = IanvsAgentApp(self.settings)
        process_started = asyncio.Event()

        async def run_process():
            process = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(30)")
            process_started.set()
            try:
                return await process.wait()
            finally:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()

        async with app.run_test(size=(100, 35)) as pilot:
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            instance = app.harness.current_instance
            app.start_action(lambda: app.harness._action(run_process()), "执行真实子进程")
            await asyncio.wait_for(process_started.wait(), timeout=10)
            await pilot.pause()
            self.assertTrue(app.harness.busy)
            self.assertTrue(app.query_one("#delete-instance", Button).disabled)
            app.delete_instance_button()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, DeleteInstanceScreen)
            self.assertTrue(Path(instance["code_root"]).is_dir())
            await app.cancel_action()
            async with asyncio.timeout(10):
                while app._active_action is not None or app.harness.busy:
                    await pilot.pause(.05)
            await pilot.pause()
            self.assertFalse(app.harness.busy)
            self.assertFalse(app.query_one("#delete-instance", Button).disabled)
            await self.open_delete(app, pilot)
            await pilot.press("escape")
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)

    async def test_cancel_during_real_permanent_deletion_preserves_worker_until_success(self):
        app = IanvsAgentApp(self.settings)
        async with app.run_test(size=(100, 35)) as pilot:
            await app.harness.create_instance("examples/llm_simple_qa/benchmarkingjob.yaml")
            instance_id = app.harness.current_instance["instance_id"]
            files = self.settings.state_root / "instances" / instance_id / "cache" / "delete-cancel-check"
            files.mkdir(parents=True)
            for index in range(12000):
                (files / f"entry-{index:05d}.txt").write_text(str(index), encoding="utf-8")
            await pilot.pause()
            app.start_action(lambda: app.delete_instance(instance_id), "正在永久删除实例")
            worker = app._active_action
            async with asyncio.timeout(20):
                while app.harness._deleting_instance_id != instance_id:
                    self.assertTrue(worker.is_running)
                    await asyncio.sleep(.001)
            self.assertTrue(worker.is_running)
            await app.cancel_action()
            self.assertFalse(worker.is_cancelled)
            if worker.is_running:
                self.assertIs(app._active_action, worker)
                self.assertIn("永久删除正在继续", app._last_status)
                self.assertIsNotNone(app._activity_started_at)
            await worker.wait()
            await pilot.pause()
            self.assertFalse(worker.is_cancelled)
            self.assertIsNone(app.harness.current_instance)
            self.assertIsNone(app.harness.session)
            self.assertFalse((self.settings.state_root / "instances" / instance_id).exists())
            deletion = app.harness.store.read(f"instance-deletions/{instance_id}/deletion.json")
            self.assertEqual(deletion["status"], "deleted")
            self.assertIn("实例已删除", app._last_status)
            self.assertNotIn("取消", app._last_status)
            self.assertIsNone(app._activity_started_at)
            self.assertEqual(app.harness.usage_summary()["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
