# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import time

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState


class ContextLimitError(RuntimeError):
    pass


class Provider:
    def __init__(self, profile, events, usage):
        self.profile = profile
        self.events = events
        self.usage = usage
        self.client = None

    def _get_client(self):
        if self.client is None:
            api_key = os.environ.get(self.profile.api_key_env)
            if not api_key:
                raise ValueError(f"缺少模型凭据环境变量 {self.profile.api_key_env}")
            timeout = self.profile.timeout_seconds
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.profile.base_url,
                timeout=httpx.Timeout(timeout, connect=min(timeout, 10), write=min(timeout, 30), pool=min(timeout, 10)),
                max_retries=0,
            )
        return self.client

    async def close(self):
        if self.client is not None:
            await self.client.close()
            self.client = None

    @staticmethod
    async def _io(operation, *arguments, **keywords):
        task = asyncio.create_task(asyncio.to_thread(operation, *arguments, **keywords))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    def output_limit(self, purpose):
        return min(self.profile.output_tokens, self.profile.auxiliary_output_tokens) if purpose in {"observe", "compaction"} else self.profile.output_tokens

    def _error_message(self, error):
        value = str(error)
        api_key = os.environ.get(self.profile.api_key_env)
        if api_key:
            value = value.replace(api_key, "[凭据已隐藏]")
        return value[:4000]

    @staticmethod
    def _failure_stage(error, telemetry):
        if isinstance(error, TimeoutError):
            return "absolute_deadline"
        if isinstance(error, httpx.ReadTimeout) or isinstance(error.__cause__, httpx.ReadTimeout):
            return "idle_timeout"
        return telemetry["phase"]

    @staticmethod
    def _context_limit(error):
        if not isinstance(error, APIStatusError):
            return False
        body = error.body or {}
        detail = body.get("error", body) if isinstance(body, dict) else {}
        code = str(detail.get("code", "")).lower() if isinstance(detail, dict) else ""
        message = str(detail.get("message", "")).lower() if isinstance(detail, dict) else ""
        return Provider._context_limit_detail(error.status_code, code, message)

    @staticmethod
    def _context_limit_detail(status_code, code, message):
        known_codes = {"context_length_exceeded", "context_window_exceeded", "context_limit_exceeded", "input_tokens_exceeded", "outofcontexterror"}
        return code.lower() in known_codes or (status_code in {400, 413} and any(value in message.lower() for value in (
            "maximum context length", "context length exceeded", "context window exceeded", "exceeds the context window", "上下文长度超过", "上下文超限",
        )))

    @staticmethod
    def _can_retry(error, telemetry):
        if telemetry["phase"] not in {"connecting", "receiving"}:
            return False
        if isinstance(error, APIStatusError):
            return error.status_code in {408, 409, 429} or error.status_code >= 500
        if isinstance(error, TimeoutError):
            return True
        if isinstance(error, APIConnectionError):
            error = error.__cause__
        # 完整响应交付前不会执行工具；接收中断可重新请求，原始片段单独保留。
        return isinstance(error, (
            httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError,
            httpx.ReadError, httpx.ReadTimeout, httpx.WriteError, httpx.WriteTimeout,
            httpx.CloseError, httpx.RemoteProtocolError,
        ))

    def _request(self, messages, tools, purpose):
        request = {
            "model": self.profile.model,
            "messages": messages,
            "max_tokens": self.output_limit(purpose),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = tools
        else:
            request["tool_choice"] = "none"
        if purpose == "observe":
            request["response_format"] = {"type": "json_object"}
        if purpose in {"observe", "compaction"} and self.profile.auxiliary_thinking is not None:
            request["extra_body"] = {"thinking": {"type": self.profile.auxiliary_thinking}}
        return request

    async def _flush_progress(self, telemetry, *, force=False):
        now = time.monotonic()
        if not force and now - telemetry["last_progress"] < 0.5:
            return
        identity = {key: telemetry[key] for key in ("request_group_id", "purpose", "attempt")}
        if telemetry["pending_text"]:
            text = "".join(telemetry["pending_text"])
            telemetry["pending_text"].clear()
            await self._io(self.events.emit, "llm.delta", {**identity, "text": text})
        await self._io(self.events.emit, "llm.progress", {
            **identity,
            "phase": telemetry["phase"],
            "elapsed_seconds": now - telemetry["started"],
            "first_chunk_seconds": telemetry["first_chunk_seconds"],
            "last_chunk_seconds": telemetry["last_chunk_seconds"],
            "chunk_count": telemetry["chunk_count"],
            "content_characters": telemetry["content_characters"],
            "idle_timeout_seconds": self.profile.timeout_seconds,
            "deadline_seconds": self.profile.request_timeout_seconds,
        })
        telemetry["last_progress"] = time.monotonic()

    async def _receive(self, client, request, state, telemetry):
        raw_stream = await client.chat.completions.create(**request)
        telemetry["response_received"] = True
        telemetry["phase"] = "receiving"
        telemetry["http_status"] = raw_stream.response.status_code
        telemetry["http_request_id"] = raw_stream.response.headers.get("x-request-id")
        async with raw_stream:
            async for chunk in raw_stream:
                received = time.monotonic() - telemetry["started"]
                if telemetry["first_chunk_seconds"] is None:
                    telemetry["first_chunk_seconds"] = received
                telemetry["last_chunk_seconds"] = received
                telemetry["chunk_count"] += 1
                telemetry["raw_chunks"].append(chunk.model_dump(mode="json", exclude_none=True))
                self._accumulate(state, chunk, telemetry["choice_roles"])
                telemetry["snapshot_available"] = True
                if chunk.usage is not None:
                    telemetry["raw_usage"] = chunk.usage.model_dump(mode="json", exclude_none=True)
                for choice in chunk.choices:
                    if choice.index == 0 and choice.delta.content:
                        telemetry["pending_text"].append(choice.delta.content)
                        telemetry["content_characters"] += len(choice.delta.content)
                await self._flush_progress(telemetry)
        telemetry["phase"] = "response_validation"
        await self._flush_progress(telemetry, force=True)
        return state.get_final_completion()

    @staticmethod
    def _accumulate(state, chunk, choice_roles):
        normalized = chunk.model_copy(deep=True)
        for choice in normalized.choices:
            role = choice.delta.role
            if role is None:
                continue
            previous = choice_roles.get(choice.index)
            if previous is not None and previous != role:
                raise ValueError("模型流式响应在同一个候选消息中改变了 role")
            if role != "assistant":
                raise ValueError("模型流式响应必须属于 assistant 消息")
            if previous is None:
                choice_roles[choice.index] = role
            else:
                # role 属于消息身份；部分兼容服务会在每个 chunk 重复返回。
                choice.delta.role = None
        state.handle_chunk(normalized)

    @staticmethod
    def _validate_response(response, allow_tools):
        if len(response.choices) != 1:
            raise ValueError("模型服务必须返回一个候选消息")
        choice = response.choices[0]
        if choice.message.role != "assistant":
            raise ValueError("模型响应必须属于 assistant 消息")
        if choice.finish_reason not in {"stop", "tool_calls"}:
            raise ValueError(f"模型响应未完整结束：{choice.finish_reason}")
        content = choice.message.content or ""
        calls = choice.message.tool_calls or []
        if calls and (not allow_tools or choice.finish_reason != "tool_calls"):
            raise ValueError("当前响应包含未开放或未完整结束的工具调用")
        if not calls and choice.finish_reason == "tool_calls":
            raise ValueError("模型声明工具调用结束，却没有提供工具调用")
        tool_calls, seen = [], set()
        for call in calls:
            if call.type != "function" or not call.id or call.id in seen or not call.function.name:
                raise ValueError("模型工具调用缺少有效名称或唯一调用 ID")
            arguments = json.loads(call.function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须为 JSON object")
            seen.add(call.id)
            tool_calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
        if not content.strip() and not tool_calls:
            raise ValueError("模型返回空内容且没有工具调用")
        return content, tool_calls

    @staticmethod
    def _usage_fields(record, raw_usage):
        record["raw_usage"] = raw_usage
        if raw_usage is not None:
            prompt_details = raw_usage.get("prompt_tokens_details") or {}
            record.update({
                "input_tokens": raw_usage.get("prompt_tokens"),
                "output_tokens": raw_usage.get("completion_tokens"),
                "cache_read_tokens": prompt_details.get("cached_tokens"),
            })
            reported_cost = raw_usage.get("cost", raw_usage.get("total_cost"))
            if isinstance(reported_cost, (int, float)) and not isinstance(reported_cost, bool):
                record.update({"cost": reported_cost, "cost_source": "reported"})

    async def _failed(self, error, state, telemetry, request, attempt_path, estimated_input, retry_reason, *, cancelled=False):
        await self._flush_progress(telemetry, force=True)
        status = "cancelled" if cancelled else "failed"
        record = self._record(status, telemetry["purpose"], telemetry["attempt"], telemetry["request_group_id"], telemetry["started"], estimated_input, retry_reason)
        response = state.current_completion_snapshot.model_dump(mode="json", exclude_none=True) if telemetry["snapshot_available"] else None
        cause_type = type(error.__cause__).__name__ if error.__cause__ is not None else None
        record.update({
            "error_type": type(error).__name__,
            "error": self._error_message(error) or ("模型请求已取消" if cancelled else "模型请求失败"),
            "cause_type": cause_type,
            "failure_stage": self._failure_stage(error, telemetry),
            "context_limit": self._context_limit(error),
            "http_status": getattr(error, "status_code", None) or telemetry["http_status"],
            "http_request_id": getattr(error, "request_id", None) or telemetry["http_request_id"],
            "retry_after": error.response.headers.get("retry-after") if isinstance(error, APIStatusError) else None,
            "request_id": response.get("id") if response else None,
            "first_chunk_seconds": telemetry["first_chunk_seconds"],
            "last_chunk_seconds": telemetry["last_chunk_seconds"],
            "chunk_count": telemetry["chunk_count"],
            "deadline_seconds": self.profile.request_timeout_seconds,
            "idle_timeout_seconds": self.profile.timeout_seconds,
            "attempt_ref": attempt_path,
        })
        if not cancelled and record["failure_stage"] in {"absolute_deadline", "idle_timeout"}:
            label = "总请求时限" if record["failure_stage"] == "absolute_deadline" else "连续无数据时限"
            seconds = self.profile.request_timeout_seconds if record["failure_stage"] == "absolute_deadline" else self.profile.timeout_seconds
            record["error"] = f"模型 {telemetry['purpose']} 请求达到{label} {seconds:g} 秒；收到 {telemetry['chunk_count']} 个数据片段，未完成的工具调用未执行。已有任务与证据保留，可通过 /resume 继续。"
        self._usage_fields(record, telemetry["raw_usage"])
        record["retryable"] = not cancelled and not record["context_limit"] and self._can_retry(error, telemetry)
        will_retry = record["retryable"] and telemetry["attempt"] < telemetry.get("max_attempts", self.profile.max_retries + 1)
        await self._io(self.usage.record, record)
        await self._io(self.usage.store.write, attempt_path, {
            "schema_version": 2, "purpose": telemetry["purpose"], "attempt": telemetry["attempt"],
            "status": status, "request": request, "error": record["error"], "diagnostics": record,
            "partial_response": response, "raw_usage": telemetry["raw_usage"], "raw_chunks": telemetry["raw_chunks"],
        })
        await self._io(self.events.emit, "llm.failed", {**record, "will_retry": will_retry})
        return record, will_retry

    def _save_completed(self, record, result, request, attempt_path, telemetry, response):
        result["usage"] = self.usage.record(record)
        self.usage.store.write(self.usage.session_relative_path + f"/requests/{telemetry['request_group_id']}.json", {
            "schema_version": 2, "purpose": telemetry["purpose"], "model": self.profile.model,
            "messages": request["messages"], "tools": request.get("tools", []), "response": result,
        })
        self.usage.store.write(attempt_path, {
            "schema_version": 2, "purpose": telemetry["purpose"], "attempt": telemetry["attempt"], "status": "completed",
            "request": request, "response": result, "raw_response": response.model_dump(mode="json", exclude_none=True),
            "raw_chunks": telemetry["raw_chunks"],
        })
        self.events.emit("llm.completed", {
            "request_group_id": telemetry["request_group_id"], "request_id": response.id, "model": response.model,
            "purpose": telemetry["purpose"], "tool_calls": len(result["tool_calls"]), "duration_seconds": record["duration_seconds"],
            "first_chunk_seconds": telemetry["first_chunk_seconds"], "chunk_count": telemetry["chunk_count"],
        })
        return result

    async def complete(self, messages, tools=None, purpose="plan", retry_reason=None, *, max_attempts=None):
        if purpose == "observe" and tools:
            raise ValueError("Observe 请求不允许开放工具")
        if max_attempts is not None and (isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0):
            raise ValueError("模型请求尝试次数必须为空值或正整数")
        attempt_limit = self.profile.max_retries + 1 if max_attempts is None else min(self.profile.max_retries + 1, max_attempts)
        estimated_input = await asyncio.to_thread(self.usage.estimator.request, messages, tools)
        output_limit = self.output_limit(purpose)
        request_group_id = await self._io(self.usage.store.allocate_id, "LLM")
        request = self._request(messages, tools, purpose)
        for attempt in range(1, attempt_limit + 1):
            if self.usage.settings is not None:
                await asyncio.to_thread(self.usage.check_budget, self.usage.settings, estimated_input, output_limit, reserve_model_calls=1)
            client = self._get_client()
            started = time.monotonic()
            telemetry = {
                "request_group_id": request_group_id, "purpose": purpose, "attempt": attempt,
                "started": started, "last_progress": started - 0.5, "phase": "connecting", "max_attempts": attempt_limit,
                "response_received": False, "http_status": None, "http_request_id": None,
                "first_chunk_seconds": None, "last_chunk_seconds": None, "chunk_count": 0,
                "content_characters": 0, "pending_text": [], "raw_chunks": [], "raw_usage": None, "snapshot_available": False,
                "choice_roles": {},
            }
            attempt_path = self.usage.session_relative_path + f"/requests/{request_group_id}/attempt-{attempt}.json"
            state = ChatCompletionStreamState()
            try:
                await self._io(self.events.emit, "llm.started", {
                    "request_group_id": request_group_id, "model": self.profile.model,
                    "purpose": purpose, "attempt": attempt, "max_attempts": attempt_limit, "estimated_input_tokens": estimated_input,
                    "deadline_seconds": self.profile.request_timeout_seconds, "idle_timeout_seconds": self.profile.timeout_seconds,
                })
                await self._io(self.usage.store.write, attempt_path, {
                    "schema_version": 2, "purpose": purpose, "attempt": attempt, "status": "started", "request": request,
                })
                # httpx read 时限只计算流的空闲时间；总时限单独约束持续响应。
                async with asyncio.timeout(self.profile.request_timeout_seconds):
                    response = await self._receive(client, request, state, telemetry)
                content, tool_calls = self._validate_response(response, bool(tools))
                raw_message = response.choices[0].message.model_dump(mode="json", exclude_none=True)
                estimated_output = await asyncio.to_thread(self.usage.estimator.text, json.dumps(raw_message, ensure_ascii=False))
            except asyncio.CancelledError as error:
                await self._failed(error, state, telemetry, request, attempt_path, estimated_input, retry_reason, cancelled=True)
                raise
            except Exception as error:
                record, will_retry = await self._failed(error, state, telemetry, request, attempt_path, estimated_input, retry_reason)
                error.request_failure = record
                if record["context_limit"]:
                    raise ContextLimitError("模型服务明确报告上下文超限") from error
                if will_retry:
                    delay = min(2 ** (attempt - 1), 8)
                    await self._io(self.events.emit, "llm.retry", {
                        "request_group_id": request_group_id, "purpose": purpose, "attempt": attempt,
                        "next_attempt": attempt + 1, "delay_seconds": delay, "error_type": record["error_type"],
                    })
                    await asyncio.sleep(delay)
                    continue
                if record["failure_stage"] in {"absolute_deadline", "idle_timeout"}:
                    timeout_error = TimeoutError(record["error"])
                    timeout_error.request_failure = record
                    raise timeout_error from error
                raise
            record = self._record("completed", purpose, attempt, request_group_id, started, estimated_input, retry_reason)
            record.update({
                "request_id": response.id, "returned_model": response.model,
                "http_request_id": telemetry["http_request_id"], "http_status": telemetry["http_status"],
                "first_chunk_seconds": telemetry["first_chunk_seconds"], "last_chunk_seconds": telemetry["last_chunk_seconds"],
                "chunk_count": telemetry["chunk_count"], "deadline_seconds": self.profile.request_timeout_seconds,
                "idle_timeout_seconds": self.profile.timeout_seconds,
                "estimated_output_tokens": estimated_output,
            })
            self._usage_fields(record, telemetry["raw_usage"])
            result = {
                "content": content, "tool_calls": tool_calls, "usage": None,
                "model": response.model, "request_id": response.id,
                "finish_reason": response.choices[0].finish_reason, "raw_message": raw_message,
            }
            return await self._io(self._save_completed, record, result, request, attempt_path, telemetry, response)
        raise RuntimeError("模型请求重试状态不完整")

    def _record(self, status, purpose, attempt, request_group_id, started, estimated_input, retry_reason=None):
        return {
            "status": status,
            "purpose": purpose,
            "attempt": attempt,
            "request_group_id": request_group_id,
            "retry_reason": retry_reason,
            "model": self.profile.model,
            "duration_seconds": time.monotonic() - started,
            "estimated_input_tokens": estimated_input,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_output_tokens": self.output_limit(purpose) if status in {"failed", "cancelled"} else None,
            "cache_read_tokens": None,
            "prices": {
                "input_per_million": self.profile.input_price_per_million,
                "output_per_million": self.profile.output_price_per_million,
                "cache_per_million": self.profile.cache_price_per_million,
            },
        }
