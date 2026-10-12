# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import json
from urllib.parse import quote


def credential_bindings(settings, instance):
    configured = settings.runtime_credentials
    selected = instance.get("credential_bindings") or {}
    provider_names = {profile.api_key_env for profile in settings.profiles.values()}
    for name, reference in selected.items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(reference, str) or not reference.isidentifier():
            raise ValueError("运行凭据必须使用环境变量名称映射")
        if configured.get(name) != reference:
            raise PermissionError(f"运行凭据引用未获配置授权：{name}")
        if name in provider_names or reference in provider_names:
            raise PermissionError("模型服务凭据不能传给 Example")
    return selected


def credential_status(settings, instance):
    selected = credential_bindings(settings, instance)
    return [{"name": name, "source_env": reference, "available": bool(os.environ.get(reference))} for name, reference in sorted(selected.items())]


def resolve_credentials(settings, instance):
    result = {}
    for name, reference in credential_bindings(settings, instance).items():
        value = os.environ.get(reference)
        if not value:
            raise ValueError(f"运行凭据尚未配置：{name}（后端环境变量 {reference}）")
        result[name] = value
    return result


class SecretRedactor:
    def __init__(self, values=()):
        variants = {variant.encode("utf-8") for value in values if value for variant in (value, quote(value, safe=""), json.dumps(value, ensure_ascii=False)[1:-1], json.dumps(value, ensure_ascii=True)[1:-1])}
        self.pattern = re.compile(b"|".join(re.escape(value) for value in sorted(variants, key=len, reverse=True))) if variants else None
        self.buffer = b""
        self.keep = max((len(value) for value in variants), default=1) - 1

    def feed(self, data, final=False):
        if self.pattern is None:
            return data
        self.buffer += data
        limit = len(self.buffer) if final else max(0, len(self.buffer) - self.keep)
        end = limit
        for match in self.pattern.finditer(self.buffer):
            if match.start() >= limit:
                break
            end = max(end, match.end())
        output, self.buffer = self.buffer[:end], self.buffer[end:]
        return self.pattern.sub(b"[REDACTED]", output)
