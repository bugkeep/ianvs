# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


def source_path(root, value):
    root = Path(root).resolve()
    target = (root / value).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError:
        raise PermissionError("依赖输入超出当前实例源码目录")
    if any(part.startswith(".env") or part in {".git", "workspace", "environments", "sessions", "changes"} for part in relative.parts):
        raise PermissionError("依赖输入属于受保护目录")
    return target


def check_url(value):
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise PermissionError("依赖地址不能包含凭据；请使用已登记的凭据引用")


def option_value(value):
    if dataclasses.is_dataclass(value):
        return option_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: option_value(item) for key, item in value.items()}
    if isinstance(value, (set, tuple)):
        return [option_value(item) for item in sorted(value)]
    if isinstance(value, list):
        return [option_value(item) for item in value]
    return value


def inspect_dependencies(root, requirements_files, constraints_files, packages):
    from pip._internal.exceptions import InstallationError
    from pip._internal.network.session import PipSession
    from pip._internal.req.constructors import install_req_from_line
    from pip._internal.req.req_file import RequirementsFileParser, get_line_parser

    root = Path(root).resolve()
    files = {}

    class ScopedParser(RequirementsFileParser):
        def _parse_file(self, filename, constraint):
            if urlsplit(str(filename)).scheme:
                raise PermissionError("requirements 引用需要使用实例内的本地文件")
            target = source_path(root, filename)
            files[target.relative_to(root).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
            yield from super()._parse_file(str(target), constraint)

    requirements, options, errors, directories = [], [], [], []

    def inspect_spec(value, source, line, constraint=False):
        check_url(value)
        requirement = install_req_from_line(value)
        link = requirement.link
        if link is not None:
            check_url(link.url)
            if link.is_file:
                target = source_path(root, link.file_path)
                if target.is_file():
                    files[target.relative_to(root).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
                elif target.is_dir():
                    directories.append(target.relative_to(root).as_posix())
                else:
                    raise FileNotFoundError("本地依赖制品缺失：" + target.relative_to(root).as_posix())
        requirement_data = {"specification": value, "source": source, "line": line, "constraint": constraint}
        if requirement.req is not None:
            requirement_data.update(name=requirement.req.name, specifier=str(requirement.req.specifier), marker=str(requirement.req.marker) if requirement.req.marker else None)
            if requirement.req.marker is None or requirement.req.marker.evaluate():
                try:
                    installed = importlib.metadata.version(requirement.req.name)
                except importlib.metadata.PackageNotFoundError:
                    installed = None
                requirement_data.update(installed_version=installed, satisfied=(constraint and installed is None) or (installed is not None and requirement.req.specifier.contains(installed, prereleases=True)))
            else:
                requirement_data.update(installed_version=None, satisfied=True, skipped_marker=True)
        requirements.append(requirement_data)

    with PipSession() as session:
        parser = ScopedParser(session, get_line_parser(None))
        for name, constraint in [(name, False) for name in requirements_files] + [(name, True) for name in constraints_files]:
            try:
                for parsed in parser.parse(str(source_path(root, name)), constraint):
                    relative = Path(parsed.filename).relative_to(root).as_posix()
                    if parsed.is_editable:
                        raise ValueError("依赖计划需要可重复安装的发行制品，不能使用 editable 安装")
                    if parsed.requirement:
                        inspect_spec(parsed.requirement, relative, parsed.lineno, parsed.constraint)
                    for key, value in vars(parsed.opts).items():
                        if value in (None, False, [], {}):
                            continue
                        if key in {"requirements", "constraints"}:
                            continue
                        if isinstance(value, (str, list)):
                            for item in value if isinstance(value, list) else [value]:
                                if isinstance(item, str):
                                    check_url(item)
                                    if key == "find_links" and urlsplit(item).scheme in {"", "file"}:
                                        from pip._internal.utils.urls import url_to_path
                                        target = source_path(root, url_to_path(item) if urlsplit(item).scheme == "file" else item)
                                        if not target.exists():
                                            raise FileNotFoundError("本地 find-links 目录不存在")
                        if hasattr(value, "no_binary") and hasattr(value, "only_binary"):
                            value = {"no_binary": sorted(value.no_binary), "only_binary": sorted(value.only_binary)}
                            if not any(value.values()):
                                continue
                        if isinstance(value, set):
                            value = sorted(value)
                        value = option_value(value)
                        if isinstance(value, dict) and not any(value.values()):
                            continue
                        options.append({"source": relative, "line": parsed.lineno, "name": key, "value": value})
            except (InstallationError, ValueError, FileNotFoundError, PermissionError) as error:
                errors.append({"source": str(name), "error": str(error)})
        for value in packages:
            try:
                inspect_spec(value, "packages", None)
            except (InstallationError, ValueError, FileNotFoundError, PermissionError) as error:
                errors.append({"source": "packages", "error": str(error)})
    return {"passed": not errors, "parser": {"name": "pip", "version": importlib.metadata.version("pip")}, "requirements": requirements, "options": options, "errors": errors, "file_versions": files, "local_directories": sorted(set(directories))}


def observe_environment(root, request):
    dependencies = inspect_dependencies(root, request.get("requirements_files", []), request.get("constraints_files", []), request.get("packages", []))
    packages = sorted([{"name": item.metadata["Name"], "version": item.version} for item in importlib.metadata.distributions()], key=lambda item: item["name"].lower())
    checked = subprocess.run([sys.executable, "-m", "pip", "--disable-pip-version-check", "check"], capture_output=True, text=True, timeout=request.get("timeout", 60), check=False)
    imports = []
    for name in request.get("modules", []):
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name) is None:
            raise ValueError("检查模块名称无效")
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                module = importlib.import_module(name)
            imports.append({"module": name, "passed": True, "version": str(getattr(module, "__version__", "unknown")), "output": output.getvalue()[-2000:], "error": None})
        except (ImportError, OSError) as error:
            imports.append({"module": name, "passed": False, "version": None, "output": output.getvalue()[-2000:], "error": str(error)})
    return {
        "passed": dependencies["passed"] and all(item.get("satisfied", True) for item in dependencies["requirements"]) and checked.returncode == 0 and all(item["passed"] for item in imports),
        "interpreter": {"python_executable": sys.executable, "version": platform.python_version(), "platform": platform.platform(), "base_prefix": sys.base_prefix, "prefix": sys.prefix},
        "packages": packages,
        "pip_check": {"passed": checked.returncode == 0, "exit_code": checked.returncode, "output": (checked.stdout + checked.stderr)[-8000:]},
        "imports": imports,
        "dependencies": dependencies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--mode", choices=("dependencies", "inspect"), default="inspect")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(args.request)
    result = inspect_dependencies(args.root, request.get("requirements_files", []), request.get("constraints_files", []), request.get("packages", [])) if args.mode == "dependencies" else observe_environment(args.root, request)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
