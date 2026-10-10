# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import psutil

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings
from ianvs_agent.runtime.commands import EnvironmentCommands
from ianvs_agent.runtime.credentials import SecretRedactor
from ianvs_agent.runtime.processes import recorded_command, sandboxed_command, spawn, runtime_environment, terminate_process
from ianvs_agent.storage.local_store import LocalStore


@unittest.skipUnless(os.name == "posix", "环境命令执行使用 Linux 后端")
class EnvironmentCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        project = Path(__file__).resolve().parents[2]
        control_root = Path(os.environ.get("IANVS_COMMAND_TEST_ROOT", project / "workspace/environment-command-tests"))
        control = LocalStore(control_root)
        self.root = control_root / control.allocate_id("command-case")
        self.code = self.root / "code"
        self.code.mkdir(parents=True)
        (self.code / "marker.txt").write_text("original", encoding="utf-8")
        self.settings = AgentSettings(project_root=self.code, state_root=self.root / "state")
        self.store = LocalStore(self.settings.state_root)
        self.events = EventBus(self.store)
        self.events.bind(task_id="task-command", instance_id="instance-command")
        venv = self.root / "venv"
        await recorded_command([sys.executable, "-m", "venv", str(venv)], self.code, sys.executable, self.root / "create.log")
        self.instance = {"instance_id": "instance-command", "code_root": str(self.code), "environment": {"environment_id": "environment-command", "generation": 1, "status": "created", "python_executable": str(venv / "bin/python")}, "needs_verification": False}
        self.store.write("instances/instance-command/instance.json", self.instance)
        self.commands = EnvironmentCommands(self.settings, self.store, self.events)

    async def test_source_is_readonly_and_failure_has_real_evidence(self):
        with self.assertRaises(RuntimeError):
            await self.commands.execute(self.instance, ["python", "-c", "from pathlib import Path; Path('marker.txt').write_text('changed')"])
        self.assertEqual((self.code / "marker.txt").read_text(), "original")
        saved = self.store.read("instances/instance-command/instance.json")["last_environment_command"]
        self.assertEqual(saved["status"], "failed")
        self.assertIn("Read-only file system", Path(saved["log_ref"]).read_text())

    async def test_generated_directory_is_writable_and_invalidates_verification(self):
        result = await self.commands.execute(self.instance, ["python", "-c", "from pathlib import Path; Path('generated/data.txt').write_text('actual output'); print('prepared')"], writable_directories=["generated"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual((self.code / "generated/data.txt").read_text(), "actual output")
        saved = self.store.read("instances/instance-command/instance.json")
        self.assertTrue(saved["needs_verification"])
        self.assertEqual(saved["generated_directories"], ["generated"])
        self.assertTrue(self.store.path(result["evidence_ref"]).is_file())

    async def test_host_records_and_environment_credentials_are_inaccessible(self):
        outside = self.root / "private.txt"
        outside.write_text("private host input", encoding="utf-8")
        (self.code / ".env").write_text("LOCAL_TOKEN=private", encoding="utf-8")
        command = ["python", "-c", "import json,os; from pathlib import Path; print(json.dumps({'host':Path(" + repr(str(outside)) + ").exists(),'dotenv':Path('.env').exists(),'provider':'ARK_API_KEY' in os.environ}))"]
        result = await self.commands.execute(self.instance, command)
        self.assertEqual(json.loads(result["stdout"]), {"host": False, "dotenv": False, "provider": False})

    async def test_approved_source_version_cannot_change_before_execution(self):
        arguments = ["python", "-c", "print('checked')"]
        approved = self.commands.preview(self.instance, arguments)
        (self.code / "marker.txt").write_text("updated", encoding="utf-8")
        with self.assertRaises(PermissionError):
            await self.commands.execute(self.instance, arguments, approved=approved)

    async def test_cancellation_terminates_real_child_and_saves_status(self):
        task = asyncio.create_task(self.commands.execute(self.instance, ["python", "-c", "import time; print('started',flush=True); time.sleep(120)"]))
        for _ in range(200):
            if self.commands.active and next(iter(self.commands.active.values())).get("status") == "running":
                break
            await asyncio.sleep(0.05)
        self.assertTrue(self.commands.active)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        saved = self.store.read("instances/instance-command/instance.json")["last_environment_command"]
        self.assertEqual(saved["status"], "cancelled")
        self.assertEqual(self.commands.active, {})

    async def test_reparented_process_in_owned_group_is_terminated(self):
        helper = Path(__file__).parent / "helpers/orphan_process.py"
        process = await spawn([sys.executable, str(helper)], self.code, runtime_environment(), stdout=asyncio.subprocess.PIPE)
        child_pid = int(await process.stdout.readline())
        try:
            for _ in range(100):
                if psutil.Process(child_pid).ppid() != process.pid:
                    break
                await asyncio.sleep(0.02)
            self.assertNotEqual(psutil.Process(child_pid).ppid(), process.pid)
            self.assertEqual(os.getpgid(child_pid), process.pid)
            await terminate_process(process)
            self.assertTrue(not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE)
        finally:
            if process.returncode is None:
                await terminate_process(process)
            if psutil.pid_exists(child_pid) and psutil.Process(child_pid).status() != psutil.STATUS_ZOMBIE:
                psutil.Process(child_pid).kill()

    async def test_instance_local_interpreter_is_rejected_before_execution(self):
        python = Path(self.instance["environment"]["python_executable"]).parent / "local-python"
        shutil.copy2(Path(sys.executable).resolve(), python)
        log = self.root / "untrusted-python.log"
        operation = {"operation_id": "untrusted-python-check"}
        with self.assertRaises(PermissionError):
            await sandboxed_command([str(python), "-c", "print('unreachable')"], self.code, python, log, operation=operation)
        self.assertEqual(operation["status"], "failed")
        self.assertEqual(operation["phase"], "preflight")
        self.assertFalse(operation["command_executed"])
        self.assertIsNone(operation["exit_code"])
        self.assertNotIn("pid", operation)
        self.assertIn("PermissionError", log.read_text())


class SecretRedactionTests(unittest.TestCase):
    def test_selected_secret_split_across_stream_chunks_is_redacted(self):
        redactor = SecretRedactor(["scoped-private-value"])
        chunks = [redactor.feed(b"prefix scoped-pri"), redactor.feed(b"vate-value suffix"), redactor.feed(b"", final=True)]
        output = b"".join(chunks)
        self.assertEqual(output, b"prefix [REDACTED] suffix")


if __name__ == "__main__":
    unittest.main()
