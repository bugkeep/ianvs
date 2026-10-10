# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
from pathlib import Path


def python_layout(python):
    executable = Path(python).resolve(strict=True)
    prefixes = (Path(sys.base_prefix).resolve(), Path("/usr/local"), Path("/usr"))
    for prefix in prefixes:
        if executable.parent == (prefix / "bin").resolve() and executable.is_file():
            return {"base_prefix": str(prefix), "executable": str(executable)}
    raise PermissionError("实例解释器必须引用系统或当前 Agent 基础 Python；不能在宿主执行实例内替换的程序")


def sandbox_command(command, cwd, python, temporary, *, writable_paths=(), readonly_paths=(), readonly_mappings=(), network=False):
    if os.name != "posix" or not shutil.which("bwrap"):
        raise RuntimeError("环境命令需要 Linux Bubblewrap；请在已配置的 WSL 后端安装并启用 bubblewrap")
    if not isinstance(command, (list, tuple)) or not command or not all(isinstance(item, str) and "\x00" not in item for item in command):
        raise ValueError("环境命令必须是非空参数列表")
    cwd = Path(cwd).resolve(strict=True)
    temporary = Path(temporary).resolve(strict=True)
    readonly = {cwd, *(Path(path).resolve(strict=True) for path in readonly_paths)}
    writable = {temporary, *(Path(path).resolve(strict=True) for path in writable_paths)}
    if any(path == Path("/") or path == Path.home() for path in readonly | writable):
        raise PermissionError("环境命令不能挂载宿主根目录或用户主目录")
    runtime = set()
    for name in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
        path = Path(name)
        if path.exists():
            runtime.add(path)
    if python:
        layout = python_layout(str(python))
        prefix = Path(layout["base_prefix"]).resolve(strict=True)
        if prefix not in {Path("/"), Path("/usr"), Path("/usr/local")}:
            runtime.update(path for name in ("bin", "lib", "lib64", "include", "share") if (path := prefix / name).exists())
        interpreter = Path(python).absolute()
        environment = interpreter.parent.parent
        if (environment / "pyvenv.cfg").is_file():
            readonly.add(environment.resolve())
    arguments = [shutil.which("bwrap"), "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--disable-userns", "--new-session", "--die-with-parent", "--cap-drop", "ALL", "--clearenv"]
    if not network:
        arguments.append("--unshare-net")
    for path in sorted(runtime | readonly, key=lambda item: (len(item.parts), str(item))):
        arguments.extend(("--ro-bind", str(path), str(path)))
    for name in ("/etc/ld.so.cache", "/etc/nsswitch.conf", "/etc/hosts", "/etc/resolv.conf", "/etc/ssl", "/etc/pki"):
        if Path(name).exists():
            arguments.extend(("--ro-bind", name, name))
    for mapping in readonly_mappings:
        source = Path(mapping["source"]).resolve(strict=True)
        destination = Path(mapping["destination"])
        if not destination.is_absolute() or destination == Path("/") or destination == Path.home():
            raise PermissionError("执行快照挂载位置无效")
        arguments.extend(("--ro-bind", str(source), str(destination)))
    for path in sorted(writable, key=lambda item: (len(item.parts), str(item))):
        arguments.extend(("--bind", str(path), str(path)))
    mappings = {Path(item["destination"]): Path(item["source"]) for item in readonly_mappings}
    for destination in sorted(readonly):
        source = mappings.get(destination, destination)
        if not source.is_dir():
            continue
        for directory, folders, files in os.walk(source, followlinks=False):
            for name in tuple(folders):
                if name.startswith(".env") or name in {".git", "workspace", "sessions", "changes", "environments"}:
                    target = destination / (Path(directory) / name).relative_to(source)
                    arguments.extend(("--tmpfs", str(target)))
                    folders.remove(name)
            for name in files:
                if name.startswith(".env"):
                    target = destination / (Path(directory) / name).relative_to(source)
                    arguments.extend(("--ro-bind", "/dev/null", str(target)))
    arguments.extend(("--proc", "/proc", "--dev", "/dev", "--chdir", str(cwd)))
    home = temporary / "home"
    home.mkdir(exist_ok=True)
    environment = {
        "PATH": (str(Path(python).parent) + os.pathsep if python else "") + "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home), "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary),
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1", "PIP_CONFIG_FILE": "/dev/null",
        "PIP_CACHE_DIR": str(temporary / "pip-cache"), "XDG_CACHE_HOME": str(temporary / "cache"),
    }
    for name, value in environment.items():
        arguments.extend(("--setenv", name, value))
    arguments.extend(("--", *command))
    isolation = {"backend": "bubblewrap", "filesystem": True, "process": True, "network": bool(network), "readonly_paths": [str(path) for path in sorted(readonly)], "writable_paths": [str(path) for path in sorted(writable)], "private_home": True}
    return arguments, isolation
