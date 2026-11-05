# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=missing-module-docstring

import json
import os
from datetime import datetime, timezone
from urllib.parse import quote


_CALLBACK = None


def set_event_callback(callback):
    """设置可选的阶段事件订阅者。"""
    # pylint: disable=global-statement
    global _CALLBACK
    _CALLBACK = callback


def _json_value(value):
    """转换指标中的 numpy 值。"""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _redact_event(event):
    """只通过名称读取授权运行凭据，在写入事件前移除值。"""
    names = json.loads(os.environ.get("IANVS_REDACT_ENV_NAMES", "[]"))
    if not isinstance(names, list) or any(
            not isinstance(name, str) or not name.isidentifier() for name in names):
        raise ValueError("事件凭据名称配置无效")
    values = {os.environ[name] for name in names if os.environ.get(name)}
    if not values:
        return event
    variants = {
        variant for value in values for variant in (
            value, quote(value, safe=""),
            json.dumps(value, ensure_ascii=False)[1:-1],
            json.dumps(value, ensure_ascii=True)[1:-1])
    }

    def clean(value):
        if isinstance(value, str):
            for secret in sorted(variants, key=len, reverse=True):
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {clean(key): clean(item) for key, item in value.items()}
        return value

    return clean(json.loads(json.dumps(event, ensure_ascii=False, default=_json_value)))


def emit_event(event_type, **payload):
    """将真实阶段事件交给配置的文件通道和订阅者。"""
    event = {
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("IANVS_RUN_ID"),
        "payload": payload,
    }
    event = _redact_event(event)
    path = os.environ.get("IANVS_EVENT_FILE")
    if path:
        with open(path, "a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False, default=_json_value) + "\n")
            output.flush()
    if _CALLBACK is not None:
        _CALLBACK(event)


def stage(name, **payload):
    """记录当前执行阶段。"""
    emit_event("run.stage_changed", stage=name, **payload)
