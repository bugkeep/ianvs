# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("Ianvs Agent 需要 Python 3.11 或更新版本")

from ianvs_agent.config import load_settings


def build_wsl_command(settings, config_path=None, distribution=None, python_executable=None):
    if os.name != "nt":
        raise ValueError("WSL 前端启动需要 Windows Python")
    distribution = distribution or settings.wsl_distribution
    backend_python = python_executable or os.environ.get("IANVS_BACKEND_PYTHON") or settings.wsl_python_executable
    command = ["wsl.exe", "--distribution", distribution, "--cd", str(settings.project_root),
               "--exec", backend_python, "-u", "-m",
               "ianvs_agent.transport.server", "--project-root", "."]
    if config_path:
        config = Path(config_path).resolve()
        converted = subprocess.run(
            ["wsl.exe", "--distribution", distribution, "--exec", "wslpath", "-a", "-u", str(config)],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=30,
        ).stdout.strip()
        command.extend(["--config", converted])
    return command


def start_agent(project_root=None, config_path=None, backend=None, distribution=None, python_executable=None):
    from ianvs_agent.app.tui import run_app

    backend = backend or ("wsl" if os.name == "nt" else "local")
    settings = load_settings(project_root, config_path, load_credentials=backend == "local")
    if backend == "wsl":
        from ianvs_agent.transport.remote import RemoteHarness

        command = build_wsl_command(settings, config_path, distribution, python_executable)
        run_app(settings, harness=RemoteHarness(settings, command, backend_label="WSL"))
    else:
        run_app(settings)


def main():
    parser = argparse.ArgumentParser(description="Ianvs Agent 中文终端界面")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Ianvs 源码目录")
    parser.add_argument("--config", type=Path, help="Agent 配置文件")
    parser.add_argument("--backend", choices=("local", "wsl"), help="执行端：Windows 默认 WSL，Linux 默认本机")
    parser.add_argument("--wsl-distro", help="WSL 发行版，默认 Ubuntu-24.04")
    parser.add_argument("--backend-python", help="WSL 内安装了 Agent 依赖的 Python 解释器")
    args = parser.parse_args()
    start_agent(args.project_root, args.config, args.backend, args.wsl_distro, args.backend_python)


if __name__ == "__main__":
    main()
