# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

import psutil
import yaml

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings
from ianvs_agent.runtime.instances import InstanceManager
from ianvs_agent.runtime.processes import runtime_environment, spawn, terminate_process
from ianvs_agent.runtime.runner import Runner
from ianvs_agent.storage.local_store import LocalStore


@unittest.skipIf(os.name == "nt", "所属进程恢复检查需要 Linux")
class RuntimeRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        project = Path(__file__).resolve().parents[2]
        artifacts = Path(os.environ.get("IANVS_RUNTIME_RECOVERY_TEST_ROOT", project / "workspace/ianvs-agent/test-artifacts/runtime-recovery"))
        artifacts.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="进程恢复-", dir=artifacts)
        self.root = Path(self.directory.name)
        self.project = self.root / "source"
        example = self.project / "examples/process-check"
        example.mkdir(parents=True)
        self.benchmark = "examples/process-check/benchmarkingjob.yaml"
        (self.project / self.benchmark).write_text(yaml.safe_dump({"benchmarkingjob": {"name": "所属进程恢复检查", "testenv": "examples/process-check/testenv.yaml", "test_object": []}}), encoding="utf-8")
        (example / "testenv.yaml").write_text(yaml.safe_dump({"testenv": {"dataset": {}, "metrics": []}}), encoding="utf-8")
        self.settings = AgentSettings(project_root=self.project, state_root=self.root / "state", python_executable=sys.executable)
        self.store = LocalStore(self.settings.state_root)
        self.events = EventBus(self.store)
        self.instance = InstanceManager(self.settings, self.store, self.events).create(self.benchmark)

    def tearDown(self):
        self.directory.cleanup()

    async def test_naturally_exited_owned_process_allows_next_run(self):
        run_id = self.store.allocate_id("control-run")
        environment = runtime_environment()
        environment["IANVS_RUN_ID"] = run_id
        process = await spawn([sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"], self.project, environment, stdin=asyncio.subprocess.PIPE)
        relative = f"instances/{self.instance['instance_id']}/runs/{run_id}/run.json"
        record = {"run_id": run_id, "instance_id": self.instance["instance_id"], "status": "running", "pid": process.pid, "process_created_at": psutil.Process(process.pid).create_time(), "interrupted": False, "evidence_type": "real_process_control"}
        self.store.write(relative, record)
        resumed = Runner(self.settings, self.store, self.events)
        try:
            with self.assertRaisesRegex(RuntimeError, "仍活动"):
                await resumed.run(self.instance)
            self.assertEqual(resumed.get_run(self.instance["instance_id"], run_id)["status"], "orphaned_running")
            self.assertTrue(psutil.pid_exists(process.pid))
            process.stdin.write(b"\n")
            await process.stdin.drain()
            process.stdin.close()
            self.assertEqual(await asyncio.wait_for(process.wait(), 5), 0)
            self.assertFalse(psutil.pid_exists(process.pid))
            with self.assertRaisesRegex(RuntimeError, "实例环境尚未准备完成"):
                await resumed.run(self.instance)
            recovered = resumed.get_run(self.instance["instance_id"], run_id)
            self.assertEqual(recovered["status"], "interrupted")
            self.assertTrue(recovered["interrupted"])
            self.assertIsNone(recovered.get("exit_code"))
            self.assertEqual(resumed._orphaned, [])
            runs = [self.store.read(path.relative_to(self.store.root).as_posix()) for path in self.settings.state_root.glob("instances/*/runs/*/run.json")]
            following = [run for run in runs if run["run_id"] != run_id]
            self.assertEqual(len(following), 1)
            self.assertEqual(following[0]["status"], "failed")
            self.assertIsNone(following[0]["exit_code"])
            self.assertIn("实例环境尚未准备完成", self.store.path(following[0]["stderr_ref"]).read_text(encoding="utf-8"))
        finally:
            if process.returncode is None:
                await terminate_process(process)
            await resumed.cancel()


if __name__ == "__main__":
    unittest.main()
