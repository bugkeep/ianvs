# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

from ianvs_agent.llm.tokens import TokenEstimator


class BudgetExceededError(RuntimeError):
    pass


class UsageTracker:
    def __init__(self, store, session_relative_path, events):
        self.store = store
        self.session_relative_path = str(session_relative_path).rstrip("/")
        self.events = events
        self.settings = None
        self.estimator = TokenEstimator(store.root / "cache")
        self.path = self.session_relative_path + "/usage.jsonl"
        self.context_path = self.session_relative_path + "/context_usage.json"

    def record(self, record):
        saved = dict(record)
        saved["task_id"] = self.events.identities.get("task_id")
        saved.setdefault("kind", "llm")
        saved.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        saved.setdefault("input_tokens", None)
        saved.setdefault("output_tokens", None)
        saved.setdefault("cache_read_tokens", None)
        saved.setdefault("estimated_input_tokens", None)
        saved.setdefault("estimated_output_tokens", None)
        saved.setdefault("cost", None)
        saved.setdefault("cost_source", None)
        for field in ("input_tokens", "output_tokens", "cache_read_tokens", "estimated_input_tokens", "estimated_output_tokens"):
            value = saved.get(field)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"{field} 必须为非负整数或 null")
        if saved["input_tokens"] is not None and saved["cache_read_tokens"] is not None and saved["cache_read_tokens"] > saved["input_tokens"]:
            raise ValueError("缓存读取 Token 不能超过输入 Token")
        if saved["cost"] is None:
            estimated = self._estimate_cost(saved)
            if estimated is not None:
                saved["cost"] = estimated
                saved["cost_source"] = "estimated"
        self.store.append(self.path, saved)
        self.events.emit("llm.usage", saved)
        return saved

    def record_tool(self, status, duration=0):
        self.store.append(self.path, {
            "kind": "tool",
            "task_id": self.events.identities.get("task_id"),
            "status": status,
            "duration_seconds": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_compaction(self, checkpoint_id):
        self.store.append(self.path, {
            "kind": "compaction",
            "task_id": self.events.identities.get("task_id"),
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def update_context(self, estimated_tokens, context_fraction, context_window):
        self.store.write(self.context_path, {
            "estimated_tokens": estimated_tokens,
            "context_fraction": context_fraction,
            "context_window": context_window,
            "estimator": "cl100k_base_with_request_margin",
        })

    @staticmethod
    def _estimate_cost(record):
        prices = record.get("prices") or {}
        input_tokens = record.get("input_tokens")
        output_tokens = record.get("output_tokens")
        input_price = prices.get("input_per_million")
        output_price = prices.get("output_per_million")
        if None in (input_tokens, output_tokens, input_price, output_price):
            return None
        cached = record.get("cache_read_tokens")
        cache_price = prices.get("cache_per_million")
        if cached is not None and cached > 0:
            if cache_price is None:
                return None
            input_cost = (input_tokens - cached) * input_price + cached * cache_price
        else:
            input_cost = input_tokens * input_price
        return (input_cost + output_tokens * output_price) / 1000000

    def summary(self):
        entries = self.store.read_events(self.path)
        calls = [entry for entry in entries if entry.get("kind") == "llm"]
        tools = [entry for entry in entries if entry.get("kind") == "tool"]
        compactions = [entry for entry in entries if entry.get("kind") == "compaction"]
        task_id = self.events.identities.get("task_id")
        task_calls = [entry for entry in calls if entry.get("task_id") == task_id]
        task_tools = [entry for entry in tools if entry.get("task_id") == task_id]
        missing = sum(entry.get("input_tokens") is None or entry.get("output_tokens") is None for entry in calls)
        input_known = sum(entry.get("input_tokens") or 0 for entry in calls)
        output_known = sum(entry.get("output_tokens") or 0 for entry in calls)
        cache_known = sum(entry.get("cache_read_tokens") or 0 for entry in calls)
        cost_missing = sum(entry.get("cost") is None for entry in calls)
        context = self.store.read(self.context_path, {})
        result = {
            "input_tokens": None if missing else input_known,
            "output_tokens": None if missing else output_known,
            "input_tokens_known": input_known,
            "output_tokens_known": output_known,
            "cache_read_tokens": None if any(entry.get("cache_read_tokens") is None for entry in calls) else cache_known,
            "cache_read_tokens_known": cache_known,
            "estimated_input_tokens": sum(entry.get("estimated_input_tokens") or 0 for entry in calls),
            "estimated_output_tokens": sum(entry.get("estimated_output_tokens") or 0 for entry in calls),
            "effective_input_tokens": sum(entry.get("input_tokens") if entry.get("input_tokens") is not None else entry.get("estimated_input_tokens") or 0 for entry in calls),
            "effective_output_tokens": sum(entry.get("output_tokens") if entry.get("output_tokens") is not None else entry.get("estimated_output_tokens") or 0 for entry in calls),
            "missing_usage_calls": missing,
            "cost": None if cost_missing else sum(entry.get("cost") or 0 for entry in calls),
            "cost_known": sum(entry.get("cost") or 0 for entry in calls),
            "cost_source": "estimated" if any(entry.get("cost_source") == "estimated" for entry in calls) else "reported" if calls and not cost_missing else None,
            "estimated_cost": any(entry.get("cost_source") == "estimated" for entry in calls),
            "missing_cost_calls": cost_missing,
            "model_calls": len(calls),
            "model_calls_succeeded": sum(entry.get("status") == "completed" for entry in calls),
            "model_calls_failed": sum(entry.get("status") in {"failed", "cancelled"} for entry in calls),
            "model_retries": sum((entry.get("attempt") or 1) > 1 or bool(entry.get("retry_reason")) for entry in calls),
            "compaction_calls": sum(entry.get("purpose") == "compaction" for entry in calls),
            "compaction_input_tokens_known": sum(entry.get("input_tokens") or 0 for entry in calls if entry.get("purpose") == "compaction"),
            "compaction_output_tokens_known": sum(entry.get("output_tokens") or 0 for entry in calls if entry.get("purpose") == "compaction"),
            "tool_calls": len(tools),
            "tool_calls_failed": sum(entry.get("status") not in {"success", "completed", "approved"} for entry in tools),
            "last_request": calls[-1] if calls else None,
            "context_estimated_tokens": context.get("estimated_tokens"),
            "context_fraction": context.get("context_fraction"),
            "context_window": context.get("context_window"),
            "usage_records": len(entries),
        }
        result.update({
            "model_failures": result["model_calls_failed"],
            "unknown_usage_calls": result["missing_usage_calls"],
            "compactions": len(compactions),
            "compaction_tokens": None if any(entry.get("input_tokens") is None or entry.get("output_tokens") is None for entry in calls if entry.get("purpose") == "compaction") else result["compaction_input_tokens_known"] + result["compaction_output_tokens_known"],
            "current_task_id": task_id,
            "task_input_tokens": None if any(entry.get("input_tokens") is None for entry in task_calls) else sum(entry["input_tokens"] for entry in task_calls),
            "task_output_tokens": None if any(entry.get("output_tokens") is None for entry in task_calls) else sum(entry["output_tokens"] for entry in task_calls),
            "task_model_calls": len(task_calls),
            "task_tool_calls": len(task_tools),
            "task_effective_input_tokens": sum(entry.get("input_tokens") if entry.get("input_tokens") is not None else entry.get("estimated_input_tokens") or 0 for entry in task_calls),
            "task_effective_output_tokens": sum(entry.get("output_tokens") if entry.get("output_tokens") is not None else entry.get("estimated_output_tokens") or 0 for entry in task_calls),
        })
        return result

    def check_budget(self, settings, estimated_input=0, output_reserve=0, *, reserve_model_calls=0, reserve_tool_calls=0):
        return self.summary()
