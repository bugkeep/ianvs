# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class CapturedStreamServer(ThreadingHTTPServer):
    def __init__(self, capture, modes):
        if capture.get("status") != "completed" or not capture.get("raw_chunks"):
            raise ValueError("网络恢复检查需要已完成真实请求的 raw_chunks")
        self.frames = [b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n" for chunk in capture["raw_chunks"]]
        self.body = b"".join(self.frames) + b"data: [DONE]\n\n"
        self.prefix_count = next((index + 1 for index, chunk in enumerate(capture["raw_chunks"])
                                  if any(choice.get("delta", {}).get("tool_calls") and not choice.get("finish_reason")
                                         for choice in chunk.get("choices", []))), 1)
        self.modes = modes
        self.requests = []
        self.lock = threading.Lock()
        self.received = threading.Event()
        self.sent_prefix = threading.Event()
        self.release = threading.Event()
        super().__init__(("127.0.0.1", 0), CapturedStreamHandler)
        self.worker = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.worker.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server_port}/v1"

    def close(self):
        self.release.set()
        self.shutdown()
        self.server_close()
        self.worker.join(timeout=5)
        if self.worker.is_alive():
            raise RuntimeError("网络故障检查服务没有退出")


class CapturedStreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            index = len(self.server.requests)
            self.server.requests.append(request)
            mode = self.server.modes[min(index, len(self.server.modes) - 1)]
        self.server.received.set()
        self.close_connection = True
        if mode == "wait_headers":
            self.server.release.wait(timeout=5)
            return
        if mode == "unauthorized":
            body = json.dumps({"error": {"message": "本地连接故障检查：认证拒绝", "code": "invalid_api_key"}}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"".join(self.server.frames[:self.server.prefix_count]) if mode == "unfinished" else self.server.body
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            if mode == "disconnect":
                self.wfile.write(b"".join(self.server.frames[:self.server.prefix_count]))
                self.wfile.flush()
                self.server.sent_prefix.set()
            elif mode == "wait_body":
                self.wfile.write(self.server.frames[0])
                self.wfile.flush()
                self.server.sent_prefix.set()
                self.server.release.wait(timeout=5)
            elif mode in {"complete", "unfinished"}:
                self.wfile.write(body)
                self.wfile.flush()
            else:
                raise ValueError(f"未知连接故障模式：{mode}")
        except (BrokenPipeError, ConnectionResetError):
            # 请求超时或取消会关闭真实 socket，服务端结束当前连接。
            return
