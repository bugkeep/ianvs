# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
import uuid
from pathlib import Path

import yaml

from ianvs_agent.config import AgentSettings, ModelProfile
from ianvs_agent.transport.remote import RemoteHarness


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RemoteStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_command_is_reported_and_real_restart_succeeds(self):
        root = PROJECT_ROOT / "workspace/ianvs-agent/test-evidence/remote-startup" / uuid.uuid4().hex[:8]
        project = root / "project"
        project.mkdir(parents=True)
        (project / "README.md").write_text("# 真实后台启动检查\n", encoding="utf-8")
        backend_state = root / "backend-state"
        config = root / "backend.yaml"
        profile = ModelProfile("启动检查配置", "ark-code-latest", "https://ark.cn-beijing.volces.com/api/plan/v3", api_key_env="IANVS_STARTUP_TEST_UNUSED_KEY")
        config.write_text(yaml.safe_dump({"state_root": str(backend_state), "profiles": {"ark": {"name": profile.name, "model": profile.model, "base_url": profile.base_url, "api_key_env": profile.api_key_env}}}), encoding="utf-8")
        frontend_state = root / "frontend-state"
        settings = AgentSettings(project, frontend_state, profiles={"ark": profile})
        remote = RemoteHarness(settings, [str(root / "unavailable-backend-executable")], backend_label="真实子进程检查")
        self.assertFalse(frontend_state.exists())
        self.assertFalse(backend_state.exists())
        with self.assertRaises(FileNotFoundError):
            await remote.start()
        self.assertFalse(remote.connected)
        self.assertTrue(remote.connection_error)
        self.assertIn("unavailable-backend-executable", remote.connection_error)
        self.assertIsNone(remote.process)
        remote.command = [sys.executable, "-u", "-m", "ianvs_agent.transport.server", "--project-root", str(project), "--config", str(config)]
        try:
            info = await remote.start()
            self.assertTrue(remote.connected)
            self.assertIsNone(remote.connection_error)
            self.assertEqual(info["project_root"], str(project.resolve()))
            self.assertEqual(settings.project_root, project)
            self.assertEqual(settings.state_root, frontend_state)
            self.assertFalse(frontend_state.exists())
            self.assertEqual(remote.usage_summary()["model_calls"], 0)
            self.assertEqual(remote.catalog.list_examples(), [])
        finally:
            await remote.close()
        self.assertFalse(remote.connected)
        self.assertIsNone(remote.process)


if __name__ == "__main__":
    unittest.main()
