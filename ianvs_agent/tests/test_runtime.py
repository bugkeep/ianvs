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
from ianvs_agent.runtime.environment import EnvironmentManager
from ianvs_agent.runtime.instances import ExampleCatalog, InstanceManager, configuration_evidence
from ianvs_agent.runtime.processes import recorded_command, runtime_environment, spawn, terminate_process
from ianvs_agent.runtime.runner import Runner
from ianvs_agent.runtime.verifier import Verifier
from ianvs_agent.storage.local_store import LocalStore


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[2]
        state_root = Path(os.environ.get("IANVS_RUNTIME_TEST_ROOT", cls.project / "workspace/runtime-tests"))
        cls.settings = AgentSettings(project_root=cls.project, state_root=state_root, python_executable=sys.executable)
        cls.store = LocalStore(cls.settings.state_root)
        cls.events = EventBus(cls.store)
        cls.instances = InstanceManager(cls.settings, cls.store, cls.events)
        existing = cls.instances.list_instances()
        cls.instance = existing[0] if existing else cls.instances.create("examples/llm_simple_qa/benchmarkingjob.yaml")

    def test_catalog_uses_benchmark_identity(self):
        examples = ExampleCatalog(self.project).list_examples()
        self.assertEqual(len(examples), len({item["benchmark_path"] for item in examples}))
        self.assertTrue(any(item["benchmark_path"] == self.instance["benchmark_path"] for item in examples))

    def test_runtime_record_identifiers_reject_path_components(self):
        runner = Runner(self.settings, self.store, self.events)
        for value in ("../instance-0002", "instance-0001/../instance-0002", "instance\\other", ".", "..", ""):
            with self.assertRaises(ValueError):
                self.instances.get(value)
            with self.assertRaises(ValueError):
                runner.get_run(self.instance["instance_id"], value)

    def test_existing_broken_configuration_remains_repairable(self):
        benchmark = "examples/Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning/benchmarkingjob.yaml"
        fixture = self.settings.state_root / "broken-config-source"
        target = fixture / benchmark
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.project / benchmark, target)
        evidence = configuration_evidence(fixture, benchmark)
        self.assertEqual(evidence["status"], "incomplete")
        self.assertIsNone(evidence["testenv_text"])
        settings = AgentSettings(project_root=fixture, state_root=self.settings.state_root / "broken-config-state", python_executable=sys.executable)
        store = LocalStore(settings.state_root)
        manager = InstanceManager(settings, store, EventBus(store))
        created = manager.create(benchmark)
        self.assertEqual(created["status"], "configuration_invalid")
        self.assertEqual(manager.get(created["instance_id"])["task_spec"]["status"], "incomplete")
        self.assertEqual((Path(created["code_root"]) / benchmark).read_bytes(), target.read_bytes())

    async def test_independent_venv_and_environment_metadata(self):
        environment = await EnvironmentManager(self.settings, self.store, self.events).prepare(self.instance, install=False)
        self.assertEqual(environment["status"], "created")
        self.assertNotEqual(environment["python_executable"], sys.executable)
        pyvenv = Path(environment["venv_root"]) / "pyvenv.cfg"
        self.assertIn("include-system-site-packages = false", pyvenv.read_text(encoding="utf-8"))
        self.assertEqual(self.instances.get(self.instance["instance_id"])["environment"]["environment_id"], environment["environment_id"])

    async def test_initialization_failure_is_a_persistent_run(self):
        with self.assertRaises(RuntimeError):
            await Runner(self.settings, self.store, self.events).run(self.instance)
        paths = sorted((self.settings.state_root / f"instances/{self.instance['instance_id']}/runs").glob("*/run.json"))
        run = self.store.read(paths[-1].relative_to(self.settings.state_root).as_posix())
        self.assertEqual(run["status"], "failed")
        self.assertIsNone(run["exit_code"])
        self.assertTrue((self.settings.state_root / run["stderr_ref"]).read_text(encoding="utf-8"))
        persisted = self.store.read(f"instances/{self.instance['instance_id']}/runs/{run['run_id']}/run.json")
        self.assertEqual(persisted["status"], "failed")

    async def test_static_validation_can_diagnose_unprepared_environment(self):
        before = self.store.read(f"instances/{self.instance['instance_id']}/instance.json")
        self.assertNotEqual((before.get("environment") or {}).get("status"), "ready")
        result = await Verifier(self.settings, self.store, self.events).validate(self.instance)
        self.assertEqual(result["evidence_type"], "real_static_validation")
        self.assertTrue(self.store.path(result["report_ref"]).is_file())
        self.assertTrue(result["checks"])
        self.assertTrue(all(item["source_kind"] == "static_validation" for item in result["checks"]))
        self.assertEqual(self.store.read(f"instances/{self.instance['instance_id']}/instance.json"), before)

    async def test_timeout_retains_actual_process_log(self):
        log = self.settings.state_root / "control-tests/timeout.log"
        code = "import time; print('进程已启动',flush=True); time.sleep(60)"
        with self.assertRaises(asyncio.TimeoutError):
            await recorded_command([sys.executable, "-c", code], self.project, None, log, timeout=1)
        self.assertIn("进程已启动", log.read_text(encoding="utf-8"))

    @unittest.skipIf(os.name == "nt", "重启后的进程组检查需要 Linux")
    async def test_restart_checks_live_process_before_cancellation(self):
        run_id = self.store.allocate_id("control-run")
        environment = runtime_environment()
        environment["IANVS_RUN_ID"] = run_id
        process = await spawn([sys.executable, "-c", "import time; time.sleep(60)"], self.project, environment)
        record = {"run_id": run_id, "instance_id": self.instance["instance_id"], "status": "running", "pid": process.pid, "process_created_at": psutil.Process(process.pid).create_time(), "interrupted": False, "evidence_type": "real_process_control"}
        relative = f"instances/{self.instance['instance_id']}/runs/{run_id}/run.json"
        self.store.write(relative, record)
        resumed = Runner(self.settings, self.store, self.events)
        self.assertEqual(resumed.recover()[0]["run_id"], run_id)
        self.assertTrue(psutil.pid_exists(process.pid))
        await resumed.cancel()
        await process.wait()
        saved = self.store.read(relative)
        self.assertEqual(saved["status"], "interrupted")
        self.assertTrue(saved["interrupted"])

    def test_data_change_invalidates_verification(self):
        self.instances.check_changes(self.instance["instance_id"])
        data = Path(self.instance["code_root"]) / "dataset/runtime-check.jsonl"
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text(json.dumps({"question": "数据变化检查", "answer": "A", "change_id": self.store.allocate_id("data-change")}), encoding="utf-8")
        instance = self.instances.check_changes(self.instance["instance_id"])
        self.assertTrue(instance["needs_verification"])
        self.assertIn("dataset/runtime-check.jsonl", instance["changed_files"])

    @unittest.skipIf(os.name == "nt", "进程组检查需要 Linux")
    async def test_cancel_terminates_owned_process_group(self):
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
        process = await spawn([sys.executable, "-c", code], self.project, runtime_environment(), stdout=asyncio.subprocess.PIPE)
        child_pid = int((await process.stdout.readline()).decode())
        self.assertEqual(os.getpgid(process.pid), process.pid)
        self.assertEqual(os.getpgid(child_pid), process.pid)
        await terminate_process(process)
        for _ in range(50):
            if not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
            await asyncio.sleep(0.1)
        self.assertTrue(not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE)


if __name__ == "__main__":
    unittest.main()
