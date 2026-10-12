# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

import psutil
import yaml

from ianvs_agent.agent.events import EventBus
from ianvs_agent.config import AgentSettings, load_settings
from ianvs_agent.runtime.environment import EnvironmentManager
from ianvs_agent.runtime.instances import InstanceManager, file_manifest
from ianvs_agent.runtime.processes import sandboxed_command
from ianvs_agent.runtime.sandbox import PROXY_VARIABLES, sandbox_command, sandbox_network
from ianvs_agent.storage.local_store import LocalStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def selected_environment(values):
    names = {*PROXY_VARIABLES, "ARK_API_KEY", "IANVS_HOST_PRIVATE", "PIP_INDEX_URL", "PIP_TIMEOUT", "PIP_RETRIES"}
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(values)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class SourceHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class ForwardingProxy(BaseHTTPRequestHandler):
    def do_GET(self):
        target = urlsplit(self.path)
        authorized = self.headers.get("Proxy-Authorization") == self.server.authorization
        self.server.requests.append({"hostname": target.hostname, "path": target.path, "authorized": authorized})
        if not authorized:
            self.send_response(407)
            self.send_header("Proxy-Authenticate", 'Basic realm="Ianvs test proxy"')
            self.end_headers()
            return
        if target.hostname != "ianvs-proxy-only.invalid":
            self.send_error(403)
            return
        with build_opener(ProxyHandler({})).open(self.server.origin + target.path, timeout=5) as response:
            body = response.read()
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@contextmanager
def local_proxy():
    origin = ThreadingHTTPServer(("127.0.0.1", 0), partial(SourceHandler, directory=str(PROJECT_ROOT / "resources/third_party")))
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ForwardingProxy)
    proxy.origin = f"http://127.0.0.1:{origin.server_port}"
    proxy.authorization = "Basic " + base64.b64encode(b"ianvs-proxy-user:ianvs-proxy-password").decode("ascii")
    proxy.requests = []
    workers = [threading.Thread(target=server.serve_forever, daemon=True) for server in (origin, proxy)]
    for worker in workers:
        worker.start()
    try:
        yield proxy, origin
    finally:
        for server in (proxy, origin):
            server.shutdown()
            server.server_close()
        for worker in workers:
            worker.join(timeout=5)


class PipConfigurationTests(unittest.TestCase):
    def test_defaults_and_configured_values_are_validated(self):
        root = PROJECT_ROOT / "workspace/sandbox-proxy-config-tests"
        store = LocalStore(root)
        folder = root / store.allocate_id("config-case")
        folder.mkdir()
        configuration = folder / "config.yaml"
        defaults = AgentSettings(PROJECT_ROOT, folder)
        self.assertEqual((defaults.pip_timeout_seconds, defaults.pip_retries), (60, 5))
        example = load_settings(PROJECT_ROOT, PROJECT_ROOT / "ianvs_agent/config.example.yaml", load_credentials=False)
        self.assertEqual((example.pip_timeout_seconds, example.pip_retries), (60, 5))
        configuration.write_text(yaml.safe_dump({"pip_timeout_seconds": 91, "pip_retries": 2}), encoding="utf-8")
        settings = load_settings(PROJECT_ROOT, configuration, load_credentials=False)
        self.assertEqual((settings.pip_timeout_seconds, settings.pip_retries), (91, 2))
        for name, value in (("pip_timeout_seconds", 0), ("pip_timeout_seconds", True), ("pip_timeout_seconds", float("nan")), ("pip_retries", -1), ("pip_retries", 1.5), ("pip_retries", False)):
            with self.subTest(name=name, value=value):
                configuration.write_text(yaml.safe_dump({name: value}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_settings(PROJECT_ROOT, configuration, load_credentials=False)


@unittest.skipUnless(os.name == "posix", "Bubblewrap 使用 Linux 后端")
class SandboxProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        root = Path(os.environ.get("IANVS_PROXY_TEST_ROOT", PROJECT_ROOT / "workspace/sandbox-proxy-tests"))
        store = LocalStore(root)
        self.root = root / store.allocate_id("proxy-case")
        self.code = self.root / "code"
        self.code.mkdir(parents=True)

    def proxy_values(self, proxy):
        address = f"http://ianvs-proxy-user:ianvs-proxy-password@127.0.0.1:{proxy.server_port}"
        return {"http_proxy": address, "https_proxy": address, "HTTP_PROXY": address, "HTTPS_PROXY": address, "no_proxy": "", "NO_PROXY": "", "ARK_API_KEY": "ianvs-model-private-key", "IANVS_HOST_PRIVATE": "ianvs-host-private-input", "PIP_INDEX_URL": "https://unapproved-index.invalid/simple"}

    async def test_authenticated_proxy_downloads_real_wheel_without_exposing_credentials(self):
        wheel = PROJECT_ROOT / "resources/third_party/sedna-0.6.0.1-py3-none-any.whl"
        downloads = self.root / "downloads"
        downloads.mkdir()
        operation = {"operation_id": "proxy-download"}
        updates = []
        with local_proxy() as (proxy, origin), selected_environment(self.proxy_values(proxy)):
            address = os.environ["http_proxy"]
            command = [sys.executable, "-m", "pip", "download", "--no-deps", "--no-index", "--trusted-host", "ianvs-proxy-only.invalid", "--dest", str(downloads), "http://ianvs-proxy-only.invalid/" + wheel.name]
            result = await sandboxed_command(command, self.code, sys.executable, self.root / "download.log", network=True, writable_paths=[downloads], operation=operation, on_update=lambda item: updates.append(json.dumps(item)))
            self.assertTrue(proxy.requests)
            self.assertTrue(all(item["authorized"] for item in proxy.requests))
            self.assertEqual(hashlib.sha256((downloads / wheel.name).read_bytes()).digest(), hashlib.sha256(wheel.read_bytes()).digest())
            public = json.dumps(result) + "".join(updates) + (self.root / "download.log").read_text()
            for secret in (address, "ianvs-proxy-password", "ianvs-model-private-key", "ianvs-host-private-input"):
                self.assertNotIn(secret, public)
            self.assertEqual(set(result["isolation"]["proxy_environment_names"]), {"http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"})
            self.assertEqual(result["isolation"]["pip_timeout_seconds"], 60)
            self.assertEqual(result["isolation"]["pip_retries"], 5)
            self.assertEqual(origin.server_address[0], "127.0.0.1")

    async def test_proxy_values_are_absent_from_argv_and_stream_output_is_redacted(self):
        running = asyncio.Event()
        operation = {"operation_id": "proxy-stream"}

        def update(item):
            if item.get("status") == "running":
                running.set()

        code = "import os,sys,time; from urllib.parse import urlsplit,unquote; value=os.environ['http_proxy']; sys.stdout.write(value[:19]); sys.stdout.flush(); time.sleep(.15); sys.stdout.write(value[19:]+'\\n'); sys.stdout.flush(); parsed=urlsplit(value); print(parsed.username,parsed.password,unquote(parsed.username),unquote(parsed.password),flush=True); print(os.environ['ARK_API_KEY'] if 'ARK_API_KEY' in os.environ else 'provider-absent',flush=True); time.sleep(1)"
        with local_proxy() as (proxy, _), selected_environment(self.proxy_values(proxy)):
            os.environ["http_proxy"] = f"http://ianvs-proxy%2fuser:ianvs-proxy%2fpassword@127.0.0.1:{proxy.server_port}"
            address = os.environ["http_proxy"]
            pending = asyncio.create_task(sandboxed_command([sys.executable, "-c", code], self.code, sys.executable, self.root / "stream.log", network=True, operation=operation, on_update=update))
            await asyncio.wait_for(running.wait(), 10)
            process = psutil.Process(operation["pid"])
            arguments = [process.cmdline(), *[child.cmdline() for child in process.children(recursive=True)]]
            self.assertNotIn(address, json.dumps(arguments))
            self.assertNotIn("ianvs-proxy%2fpassword", json.dumps(arguments))
            result = await pending
            output = (self.root / "stream.log").read_text()
            self.assertIn("[REDACTED]", output)
            self.assertIn("provider-absent", output)
            self.assertNotIn(address, output)
            for secret in ("ianvs-proxy%2fuser", "ianvs-proxy%2fpassword", "ianvs-proxy/user", "ianvs-proxy/password"):
                self.assertNotIn(secret, output)
            self.assertNotIn(address, json.dumps(result))

    async def test_disabled_network_clears_proxy_and_host_credentials(self):
        with local_proxy() as (proxy, origin), selected_environment(self.proxy_values(proxy)):
            code = "import json,os,urllib.request,urllib.error; names=" + repr([*PROXY_VARIABLES, "ARK_API_KEY", "IANVS_HOST_PRIVATE", "PIP_INDEX_URL"]) + "; print(json.dumps({name:name in os.environ for name in names})); urllib.request.urlopen(" + repr(f"http://127.0.0.1:{origin.server_port}/") + ",timeout=.5)"
            operation = {"operation_id": "offline-network"}
            with self.assertRaises(RuntimeError):
                await sandboxed_command([sys.executable, "-c", code], self.code, sys.executable, self.root / "offline.log", network=False, operation=operation)
            output = (self.root / "offline.log").read_text()
            self.assertFalse(any(json.loads(output.splitlines()[0]).values()))
            self.assertFalse(operation["isolation"]["network"])
            self.assertEqual(operation["isolation"]["proxy_environment_names"], [])
            self.assertEqual(proxy.requests, [])
            self.assertNotEqual(operation["exit_code"], 0)

    async def test_effective_pip_timeout_and_retry_settings_override_host_values(self):
        with selected_environment({"PIP_TIMEOUT": "1", "PIP_RETRIES": "0", "ARK_API_KEY": "ianvs-private-key", "IANVS_HOST_PRIVATE": "ianvs-host-private-input"}):
            code = "import json,os; from pip._internal.commands import create_command; options,_=create_command('download').parse_args([]); print(json.dumps({'timeout':options.timeout,'retries':options.retries,'provider':'ARK_API_KEY' in os.environ,'host_private':'IANVS_HOST_PRIVATE' in os.environ,'index':os.environ.get('PIP_INDEX_URL')}))"
            result = await sandboxed_command([sys.executable, "-c", code], self.code, sys.executable, self.root / "settings.log", network=True, pip_timeout_seconds=91, pip_retries=2)
            actual = json.loads((self.root / "settings.log").read_text())
            self.assertEqual(actual, {"timeout": 91, "retries": 2, "provider": False, "host_private": False, "index": None})
            self.assertEqual(result["isolation"]["pip_timeout_seconds"], 91)

    async def test_proxy_descriptor_is_closed_after_cancellation_and_preflight_failure(self):
        with local_proxy() as (proxy, _), selected_environment(self.proxy_values(proxy)):
            before = len(os.listdir("/proc/self/fd"))
            running = asyncio.Event()
            operation = {"operation_id": "proxy-cancellation"}
            pending = asyncio.create_task(sandboxed_command([sys.executable, "-c", "import time; time.sleep(60)"], self.code, sys.executable, self.root / "cancel.log", network=True, operation=operation, on_update=lambda item: running.set() if item.get("status") == "running" else None))
            await asyncio.wait_for(running.wait(), 10)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertEqual(operation["status"], "cancelled")
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)
            with self.assertRaises(PermissionError):
                await sandboxed_command([sys.executable, "-c", "print('unreachable')"], Path.home(), sys.executable, self.root / "preflight.log", network=True)
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_proxy_parameters_use_sealed_memory_descriptor(self):
        temporary = self.root / "temporary"
        temporary.mkdir()
        with selected_environment({"http_proxy": "http://ianvs-proxy-user:ianvs-proxy-password@127.0.0.1:34567", "ARK_API_KEY": "ianvs-private-key"}):
            with sandbox_network(True) as (descriptor, names, _):
                arguments, isolation = sandbox_command([sys.executable, "-c", "print('ready')"], self.code, sys.executable, temporary, network=True, proxy_fd=descriptor, proxy_names=names)
                self.assertIn("--clearenv", arguments)
                self.assertIn("--args", arguments)
                self.assertNotIn("ianvs-proxy-password", " ".join(arguments))
                self.assertNotIn("ianvs-private-key", " ".join(arguments))
                self.assertNotIn("ianvs-proxy-password", json.dumps(isolation))
                with self.assertRaises(PermissionError):
                    os.write(descriptor, b"changed")
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    @unittest.skipUnless(os.environ.get("IANVS_PROXY_LIVE_PYPI") == "1", "真实 PyPI 小型下载验收需要显式启用")
    async def test_actual_pypi_small_wheel_download(self):
        downloads = self.root / "downloads"
        downloads.mkdir()
        cache = self.root / "pip-cache"
        cache.mkdir()
        result = await sandboxed_command([sys.executable, "-m", "pip", "download", "--no-deps", "--cache-dir", str(cache), "--dest", str(downloads), "prettytable==2.5.0"], self.code, sys.executable, self.root / "pypi-download.log", timeout=180, network=True, writable_paths=[downloads, cache])
        wheel = downloads / "prettytable-2.5.0-py3-none-any.whl"
        self.assertTrue(wheel.is_file())
        self.assertEqual(wheel.stat().st_size, 24774)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["isolation"]["proxy_environment_names"])

    @unittest.skipUnless(os.environ.get("IANVS_PROXY_LIVE_PYPI") == "1", "正式联网安装验收需要显式启用真实 PyPI")
    async def test_actual_pypi_formal_instance_installation(self):
        started = time.monotonic()
        source = self.root / "source"
        example = "examples/cifar100/federated_class_incremental_learning/fedavg"
        shutil.copytree(PROJECT_ROOT / example, source / example)
        inventory = ".github/workflows/validator/data/example_inventory.yaml"
        (source / inventory).parent.mkdir(parents=True)
        shutil.copy2(PROJECT_ROOT / inventory, source / inventory)
        settings = AgentSettings(source, self.root / "state", python_executable=sys.executable)
        store = LocalStore(settings.state_root)
        events = EventBus(store, task_id="pip-proxy-formal-installation")
        instance = InstanceManager(settings, store, events).create(example + "/benchmarkingjob.yaml")
        code = Path(instance["code_root"])
        (code / "network-requirements.txt").write_text("prettytable~=2.5.0\n", encoding="utf-8")
        before = file_manifest(code)
        manager = EnvironmentManager(settings, store, events)
        plan_started = time.monotonic()
        plan = await manager.plan_install(instance, requirements_files=["network-requirements.txt"], modules=["prettytable"])
        plan_seconds = time.monotonic() - plan_started
        install_started = time.monotonic()
        environment = await manager.install(instance, plan["plan_id"])
        install_seconds = time.monotonic() - install_started
        self.assertEqual(environment["status"], "ready")
        self.assertTrue(environment["postcheck"]["passed"])
        self.assertTrue(store.path(environment["readiness_ref"]).is_file())
        self.assertEqual(next(item["version"] for item in environment["packages"] if item["name"].lower() == "prettytable"), "2.5.0")
        saved_plan = store.read(plan["evidence_ref"])
        self.assertEqual(saved_plan["status"], "completed")
        self.assertEqual(saved_plan["postcheck_ref"], environment["postcheck"]["evidence_ref"])
        reloaded = store.read(f"instances/{instance['instance_id']}/instance.json")
        checked = await EnvironmentManager(settings, store, events).inspect(reloaded)
        self.assertTrue(checked["passed"])
        self.assertTrue(checked["readiness"]["passed"])
        self.assertEqual(before, file_manifest(code))
        trace = [json.loads(line) for line in store.path("events.jsonl").read_text().splitlines()]
        self.assertFalse(any(item["event_type"].startswith("llm.") for item in trace))
        install_operation = environment["steps"][1]
        self.assertTrue(install_operation["isolation"]["proxy_environment_names"])
        self.assertEqual(install_operation["isolation"]["pip_timeout_seconds"], 60)
        self.assertEqual(install_operation["isolation"]["pip_retries"], 5)
        report = {"passed": True, "instance_id": instance["instance_id"], "environment_id": environment["environment_id"], "example": example, "validated_dependency_scope": "prettytable~=2.5.0 与其依赖；本次未安装 Ianvs 完整依赖或运行 CIFAR Example", "plan_ref": plan["evidence_ref"], "postcheck_ref": environment["postcheck"]["evidence_ref"], "readiness_ref": environment["readiness_ref"], "restored_check_ref": checked["evidence_ref"], "installed_packages": environment["packages"], "proxy_environment_names": install_operation["isolation"]["proxy_environment_names"], "pip_timeout_seconds": 60, "pip_retries": 5, "step_timeout_seconds": plan["timeout"], "plan_seconds": plan_seconds, "install_seconds": install_seconds, "total_seconds": time.monotonic() - started, "source_unchanged": before == file_manifest(code), "model_calls": 0}
        store.write("proxy-installation-result.json", report)
        print("PROXY_INSTALLATION_RESULT=" + str(store.path("proxy-installation-result.json")))


if __name__ == "__main__":
    unittest.main()
