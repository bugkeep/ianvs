# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import os
from pathlib import Path

from filelock import FileLock

from ianvs_agent.config import load_settings
from ianvs_agent.transport.protocol import decode_frame


async def interrupt_backend(settings, failure_mode):
    from ianvs_agent.transport.server import BackendService

    class InterruptedBackendService(BackendService):
        async def send(self, frame):
            if failure_mode == "publisher" and frame["type"] == "event" and frame["event"]["event_type"] == "approval.requested":
                raise RuntimeError("通信检查：事件发布失败")
            if failure_mode == "closing-publisher" and frame["type"] == "event" and frame["event"]["event_type"] == "agent.cancelled":
                raise RuntimeError("通信检查：关闭期间事件发布失败")
            await super().send(frame)

        @staticmethod
        def write_bytes(payload):
            frame = decode_frame(payload)
            if failure_mode == "closing-writer" and frame["type"] == "event" and frame["event"]["event_type"] == "agent.cancelled":
                raise BrokenPipeError("通信检查：关闭期间发送失败")
            BackendService.write_bytes(payload)
            if failure_mode == "fragment" and frame["type"] == "fragment":
                # 注入真实进程中断，前端只会取得完整消息的第一个片段。
                os._exit(31)

    settings.state_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(settings.state_root / "transport-backend.lock"), timeout=0):
        await InterruptedBackendService(settings).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通信分片中断检查后台")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--failure-mode", choices=("fragment", "publisher", "closing-publisher", "closing-writer"), default="fragment")
    arguments = parser.parse_args()
    asyncio.run(interrupt_backend(load_settings(arguments.project_root, arguments.config), arguments.failure_mode))
