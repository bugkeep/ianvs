# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import sys
import unittest
import uuid
from pathlib import Path
from urllib.parse import quote

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.runtime.processes import spawn, terminate_process
from ianvs_agent.runtime.runner import Runner
from ianvs_agent.storage.local_store import LocalStore


PROJECT = Path(__file__).resolve().parents[2]
CONTROL = PROJECT / "ianvs_agent/tests/fixtures/runner_credential_control.py"


@unittest.skipUnless(os.name == "posix", "凭据运行需要真实 Linux/WSL 后端")
class RunnerCredentialTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        root = Path(os.environ.get("IANVS_RUNNER_CREDENTIAL_TEST_ROOT", PROJECT / "workspace/ianvs-agent/test-evidence/runner-credentials"))
        self.root = root / (self.id().split(".")[-1] + "-" + uuid.uuid4().hex[:8])
        profile = ModelProfile(name="凭据进程控制", model="unused", base_url="http://127.0.0.1", api_key_env="IANVS_CONTROL_PROVIDER")
        self.settings = AgentSettings(project_root=PROJECT, state_root=self.root / "state", profiles={"ark": profile}, runtime_credentials={"IANVS_CONTROL_BOUND": "IANVS_CONTROL_SOURCE", "IANVS_CONTROL_UNUSED": "IANVS_CONTROL_UNUSED_SOURCE"})
        self.store = LocalStore(self.settings.state_root)
        self.events = EventBus(self.store)
        self.runner = Runner(self.settings, self.store, self.events)
        self.instance = {"instance_id": "instance-0001", "credential_bindings": {"IANVS_CONTROL_BOUND": "IANVS_CONTROL_SOURCE"}}
        self.names = ("IANVS_CONTROL_SOURCE", "IANVS_CONTROL_UNUSED", "IANVS_CONTROL_UNUSED_SOURCE", "IANVS_CONTROL_PROVIDER")
        self.original_env = {name: os.environ.get(name) for name in self.names}
        self.secret = "CONTROL-" + uuid.uuid4().hex + '-"中文\n/finish'
        for name in self.names:
            os.environ[name] = self.secret if name == "IANVS_CONTROL_SOURCE" else uuid.uuid4().hex

    async def asyncTearDown(self):
        await self.runner.cancel()
        for name, value in self.original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    async def run_control(self, emitter, mode="clean", timeout=10):
        environment, credentials = self.runner._execution_environment(self.instance, sys.executable)
        root = self.store.path("instances/instance-0001/runs/control-run")
        root.mkdir(parents=True)
        run = {"run_id": "control-run", "instance_id": "instance-0001", "events": [], "testcases": [], "metrics": {}, "output_dir": str(root / "output"), "stdout_ref": "instances/instance-0001/runs/control-run/stdout.log", "stderr_ref": "instances/instance-0001/runs/control-run/stderr.log", "evidence_type": "real_process_control_for_credential_handling"}
        temporary = root / "temporary"
        temporary.mkdir()
        pipe_path = root / "events.pipe"
        os.mkfifo(pipe_path, 0o600)
        pipe = os.open(pipe_path, os.O_RDWR | os.O_NONBLOCK)
        environment.update(IANVS_RUN_ID=run["run_id"], IANVS_EVENT_FILE=str(pipe_path), IANVS_REDACT_ENV_NAMES=json.dumps(sorted(credentials)), IANVS_RUN_WORKSPACE=run["output_dir"], TMPDIR=str(temporary))
        process = await spawn([sys.executable, str(CONTROL), str(emitter), mode], PROJECT, environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.runner._process = process
        readers = [asyncio.create_task(self.runner._stream(process.stdout, self.store.path(run["stdout_ref"]), "stdout", run, credentials.values())), asyncio.create_task(self.runner._stream(process.stderr, self.store.path(run["stderr_ref"]), "stderr", run, credentials.values())), asyncio.create_task(self.runner._read_core_events(root / "core-events.jsonl", run, credentials.values(), pipe))]
        try:
            try:
                await asyncio.wait_for(process.wait(), timeout)
            except asyncio.TimeoutError:
                await self.runner.cancel()
                run["control_timed_out"] = True
            await asyncio.gather(*readers)
            run["control_exit_code"] = process.returncode
            self.runner._save(run)
        finally:
            if process.returncode is None:
                await terminate_process(process)
            await asyncio.gather(*readers, return_exceptions=True)
            self.runner._process = None
            os.close(pipe)
            pipe_path.unlink()
        return run

    def assert_public_output_safe(self, run):
        variants = (self.secret, quote(self.secret, safe=""), json.dumps(self.secret, ensure_ascii=False)[1:-1], json.dumps(self.secret, ensure_ascii=True)[1:-1])
        paths = [self.store.path(run["stdout_ref"]), self.store.path(run["stderr_ref"]), self.store.path("instances/instance-0001/runs/control-run/run.json"), self.store.path("instances/instance-0001/runs/control-run/core-events.jsonl"), self.store.path("events.jsonl")]
        for path in paths:
            data = path.read_text(encoding="utf-8")
            self.assertNotIn("\ufffd", data, str(path))
            for value in variants:
                self.assertNotIn(value, data, str(path))
        for channel in ("stdout", "stderr"):
            text = self.store.path(run[channel + "_ref"]).read_text(encoding="utf-8")
            self.assertEqual(text.count("[REDACTED]"), 4)
            self.assertEqual(text.count("中文结束"), 4)
        self.assertEqual(run["testcases"], [])

    @unittest.skipUnless(os.environ.get("IANVS_RUNNER_CAPTURE_INSTANCE"), "需要真实旧实例的事件实现来源")
    async def test_old_instance_events_and_split_output_are_redacted(self):
        capture = Path(os.environ["IANVS_RUNNER_CAPTURE_INSTANCE"])
        original = capture.read_bytes()
        instance = json.loads(original)
        emitter = Path(instance["code_root"]) / "core/common/events.py"
        old_emitter = emitter.read_bytes()
        self.assertNotIn(b"IANVS_REDACT_ENV_NAMES", old_emitter)
        run = await self.run_control(emitter)
        self.assertEqual(run["control_exit_code"], 0)
        self.assert_public_output_safe(run)
        self.assertEqual(run["events"][0]["payload"]["values"], ["[REDACTED]"] * 4)
        self.assertEqual(capture.read_bytes(), original)
        self.assertEqual(emitter.read_bytes(), old_emitter)
        self.store.write("credential-control-report.json", {"evidence_type": "real_process_control_for_credential_handling", "source_emitter": str(emitter), "model_calls": 0, "benchmark_executed": False, "credential_values_persisted": False, "split_utf8_preserved": True})

    async def test_credential_artifact_is_quarantined(self):
        run = await self.run_control(PROJECT / "core/common/events.py", mode="artifact")
        self.assertEqual(run["control_exit_code"], 0)
        self.assert_public_output_safe(run)
        restricted = self.runner._quarantine_artifacts(run["output_dir"], [self.secret])
        self.assertEqual(len(restricted), 1)
        self.assertEqual(json.loads(self.store.path(restricted[0]["private_ref"]).read_text())["value"], self.secret)
        self.assertFalse((Path(run["output_dir"]) / "restricted.txt").exists())
        self.assertEqual([path.name for path in Path(run["output_dir"]).iterdir()], ["clean.txt"])
        temporary = Path(run["output_dir"]).parent / "temporary"
        restricted = self.runner._quarantine_artifacts(temporary, [self.secret])
        self.assertEqual(len(restricted), 1)
        self.assertEqual(self.store.path(restricted[0]["private_ref"]).read_text(), self.secret)
        self.assertFalse(list(temporary.iterdir()))

    async def test_cancellation_keeps_safe_output(self):
        run = await self.run_control(PROJECT / "core/common/events.py", mode="timeout", timeout=2)
        self.assertTrue(run["control_timed_out"])
        self.assertNotEqual(run["control_exit_code"], 0)
        self.assert_public_output_safe(run)
        self.assertIsNone(self.runner._process)

    async def test_core_emitter_redacts_before_direct_file_write(self):
        environment, credentials = self.runner._execution_environment(self.instance, sys.executable)
        target = self.root / "direct-core-events.jsonl"
        environment.update(IANVS_RUN_ID="control-core", IANVS_EVENT_FILE=str(target), IANVS_REDACT_ENV_NAMES=json.dumps(sorted(credentials)), IANVS_RUN_WORKSPACE=str(self.root / "direct-output"))
        process = await spawn([sys.executable, str(CONTROL), str(PROJECT / "core/common/events.py"), "clean"], PROJECT, environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await process.communicate()
        self.assertEqual(process.returncode, 0)
        payload = json.loads(target.read_text(encoding="utf-8"))["payload"]
        self.assertEqual(payload["values"], ["[REDACTED]"] * 4)
        self.assertEqual(payload["nested"], {"[REDACTED]": {"原文": "[REDACTED]"}})


if __name__ == "__main__":
    unittest.main()
