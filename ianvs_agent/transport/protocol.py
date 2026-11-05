# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import uuid


PROTOCOL_VERSION = 2
MAX_FRAME_BYTES = 16 * 1024 * 1024
FRAGMENT_BYTES = 1024 * 1024
MESSAGE_TYPES = {"request", "response", "snapshot", "event", "ready"}


class TransportError(RuntimeError):
    pass


class RemoteCallError(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        self.error_type = detail.get("type", "RemoteError")
        super().__init__(detail.get("message", "后台操作失败"))


def serialize_message(frame):
    return json.dumps({"protocol_version": PROTOCOL_VERSION, **frame}, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"


def encode_frame(frame):
    payload = serialize_message(frame)
    if len(payload) > MAX_FRAME_BYTES:
        raise TransportError("后台通信消息超过 16 MiB 上限")
    return payload


def decode_frame(payload):
    if len(payload) > MAX_FRAME_BYTES or not payload.endswith(b"\n"):
        raise TransportError("后台通信消息超过上限或缺少完整分隔")
    return decode_message(payload, allow_fragment=True)


def decode_message(payload, allow_fragment=False):
    frame = json.loads(payload.decode("utf-8"))
    if not isinstance(frame, dict) or frame.get("protocol_version") != PROTOCOL_VERSION:
        raise TransportError("后台通信协议版本或消息结构不匹配")
    if frame.get("type") not in MESSAGE_TYPES | ({"fragment"} if allow_fragment else set()):
        raise TransportError("后台通信消息类型无效")
    return frame


def encode_message(frame):
    payload = serialize_message(frame)
    if len(payload) <= MAX_FRAME_BYTES:
        yield payload
        return
    message_id = uuid.uuid4().hex
    count = (len(payload) + FRAGMENT_BYTES - 1) // FRAGMENT_BYTES
    for index in range(count):
        content = payload[index * FRAGMENT_BYTES:(index + 1) * FRAGMENT_BYTES]
        yield encode_frame({"type": "fragment", "message_id": message_id, "index": index, "count": count, "content": base64.b64encode(content).decode("ascii")})


class MessageAssembler:
    def __init__(self):
        self.message_id = None
        self.count = 0
        self.parts = []

    def feed(self, payload):
        frame = decode_frame(payload)
        if frame["type"] != "fragment":
            if self.message_id is not None:
                raise TransportError("后台通信分片未完成就收到其他消息")
            return frame
        message_id, index, count = frame.get("message_id"), frame.get("index"), frame.get("count")
        if not isinstance(message_id, str) or not message_id or type(index) is not int or type(count) is not int or count < 2 or not 0 <= index < count or not isinstance(frame.get("content"), str):
            raise TransportError("后台通信分片结构无效")
        if self.message_id is None:
            if index != 0:
                raise TransportError("后台通信分片缺少起始片段")
            self.message_id, self.count = message_id, count
        if message_id != self.message_id or count != self.count or index != len(self.parts):
            raise TransportError("后台通信分片身份或顺序无效")
        content = base64.b64decode(frame["content"], validate=True)
        if not content or len(content) > FRAGMENT_BYTES or (index < count - 1 and len(content) != FRAGMENT_BYTES):
            raise TransportError("后台通信分片长度无效")
        self.parts.append(content)
        if len(self.parts) < self.count:
            return None
        complete = b"".join(self.parts)
        self.message_id, self.count, self.parts = None, 0, []
        if not complete.endswith(b"\n"):
            raise TransportError("后台通信完整消息缺少分隔")
        return decode_message(complete)
