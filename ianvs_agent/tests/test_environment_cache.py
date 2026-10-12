# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import os
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ianvs_agent.runtime.dependencies import inspect_dependencies
from ianvs_agent.runtime.environment import EnvironmentManager
from ianvs_agent.runtime.instances import InstanceManager
from ianvs_agent.tests import test_environment_operations as operation_tests


class CacheHandler(SimpleHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server, directory=str(server.wheel_root))

    def do_GET(self):
        self.server.requests.append(self.path)
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()

    def log_message(self, format, *args):
        return


@unittest.skipUnless(os.name == "posix" and os.environ.get("IANVS_ENVIRONMENT_WHEEL_ROOT"), "需要 Linux Bubblewrap 和真实 wheel 目录")
class EnvironmentCacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await operation_tests.EnvironmentOperationsTests.asyncSetUp(self)

    async def test_real_http_cache_reuses_resolution_install_and_next_plan_per_instance(self):
        wheel_root = Path(os.environ["IANVS_ENVIRONMENT_WHEEL_ROOT"])
        wheel = wheel_root / "packaging-26.3-py3-none-any.whl"
        wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
        server = ThreadingHTTPServer(("127.0.0.1", 0), CacheHandler)
        server.wheel_root, server.requests = wheel_root, []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host = "127.0.0.1:" + str(server.server_port)
        requirement_text = f"--no-index\n--find-links http://{host}/\npackaging==26.3\n"
        dependency = self.root / "dependencies.txt"
        dependency.write_text(requirement_text, encoding="utf-8")
        source_hash = hashlib.sha256(dependency.read_bytes()).hexdigest()
        count = lambda: server.requests.count("/" + wheel.name)
        try:
            first = await self.manager.plan_install(self.instance, requirements_files=["dependencies.txt"], modules=["packaging"])
            after_resolve = count()
            environment = await self.manager.install(self.instance, first["plan_id"])
            after_install = count()
            next_plan = await EnvironmentManager(self.settings, self.store, self.events).plan_install(self.instance, requirements_files=["dependencies.txt"], modules=["packaging"])
            after_next_plan = count()
            cache = Path(first["pip_cache_dir"])
            cache_files = [path for path in cache.rglob("*") if path.is_file()]
            second_instance = InstanceManager(self.settings, self.store, self.events).create(self.instance["benchmark_path"])
            second_root = Path(second_instance["code_root"])
            (second_root / "dependencies.txt").write_text(requirement_text, encoding="utf-8")
            other_plan = await self.manager.plan_install(second_instance, requirements_files=["dependencies.txt"], modules=["packaging"])
            after_other_instance = count()
            resolved = self.store.read(first["resolution_ref"])["install"]
            installed = environment["pip_report"]["install"]
            resolution_hash = resolved[0]["download_info"]["archive_info"]["hashes"]["sha256"]
            install_hash = installed[0]["download_info"]["archive_info"]["hashes"]["sha256"]
            report = {
                "case_root": str(self.case_root), "source_wheel": str(wheel), "source_wheel_sha256": wheel_hash,
                "http_host": host, "wheel_requests": {"after_resolution": after_resolve, "after_install": after_install, "after_next_plan": after_next_plan, "after_other_instance": after_other_instance},
                "cache_file_count": len(cache_files), "cache_bytes": sum(path.stat().st_size for path in cache_files),
                "plan_refs": [first["evidence_ref"], next_plan["evidence_ref"], other_plan["evidence_ref"]],
                "instance_cache": str(cache), "other_instance_cache": other_plan["pip_cache_dir"],
                "resolution_sha256": resolution_hash, "installation_sha256": install_hash,
                "environment_status": environment["status"], "postcheck_passed": environment["postcheck"]["passed"],
                "actual_packages": environment["packages"], "source_unchanged": source_hash == hashlib.sha256(dependency.read_bytes()).hexdigest(),
            }
            self.store.write("pip-cache-test-result.json", report)
            print("CACHE_RESULT=" + str(self.store.path("pip-cache-test-result.json")))
            self.assertGreater(after_resolve, 0)
            self.assertEqual(after_install, after_resolve)
            self.assertEqual(after_next_plan, after_resolve)
            self.assertGreater(after_other_instance, after_next_plan)
            self.assertGreater(len(cache_files), 0)
            self.assertNotEqual(first["pip_cache_dir"], other_plan["pip_cache_dir"])
            self.assertEqual(first["pip_cache_dir"], next_plan["pip_cache_dir"])
            self.assertEqual(first["cache_trusted_hosts"], [host])
            self.assertEqual(first["resolved_packages"], {"packaging": "==26.3"})
            self.assertEqual(resolution_hash, wheel_hash)
            self.assertEqual(install_hash, wheel_hash)
            self.assertEqual(environment["status"], "ready")
            self.assertTrue(environment["postcheck"]["passed"])
            self.assertTrue(report["source_unchanged"])
            self.assertEqual(environment["steps"][1]["command"], first["commands"][1]["command"])
            for operation in [first["resolve_operation"], next_plan["resolve_operation"], environment["steps"][1]]:
                self.assertIn(str(cache), operation["isolation"]["writable_paths"])
                self.assertNotIn(str(cache.parent), operation["isolation"]["writable_paths"])
                self.assertIn("--cache-dir", operation["command"])
                self.assertIn("--trusted-host", operation["command"])
            self.assertNotIn(str(cache), other_plan["resolve_operation"]["isolation"]["writable_paths"])
            next_plan["commands"][1]["command"][next_plan["commands"][1]["command"].index("--cache-dir") + 1] = other_plan["pip_cache_dir"]
            self.store.write(next_plan["evidence_ref"], next_plan)
            with self.assertRaisesRegex(PermissionError, "当前实例批准的缓存目录"):
                self.manager.revalidate(self.instance, next_plan["plan_id"])
            self.assertFalse(Path(next_plan["venv_root"]).exists())
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join()

    async def test_only_parsed_loopback_find_links_enable_http_caching(self):
        dependency = self.root / "dependencies.txt"
        dependency.write_text("--no-index\n--find-links http://127.0.0.2:8801/\n--find-links http://localhost:8802/\n--find-links http://[::1]:8803/\n--find-links http://192.0.2.1:8804/\n--find-links https://example.org/\npackaging==26.3\n", encoding="utf-8")
        parsed = inspect_dependencies(self.root, ["dependencies.txt"], [], [])
        self.assertTrue(parsed["passed"])
        self.assertEqual(self.manager._loopback_cache_hosts(parsed), ["127.0.0.2:8801", "[::1]:8803", "localhost:8802"])
        own_cache = self.manager._pip_cache(self.instance)
        other_cache = own_cache.parent.parent / "instance-other" / "pip-cache"
        other_cache.mkdir(parents=True)
        own_cache.rmdir()
        own_cache.symlink_to(other_cache, target_is_directory=True)
        with self.assertRaisesRegex(PermissionError, "当前实例的独立目录"):
            self.manager._pip_cache(self.instance)


if __name__ == "__main__":
    unittest.main()
