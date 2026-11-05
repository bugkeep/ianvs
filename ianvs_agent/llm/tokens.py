# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from pathlib import Path

import tiktoken


class TokenEstimator:
    def __init__(self, cache_root):
        cache_path = Path(cache_root) / "tiktoken"
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_path)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def text(self, value):
        return len(self.encoding.encode(str(value), disallowed_special=()))

    def request(self, messages, tools=None):
        payload = {"messages": messages, "tools": tools or []}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        # 自定义模型使用通用 tokenizer 估算，并预留消息包装空间。
        return math.ceil((self.text(encoded) + 4 * len(messages) + 3) * 1.15)
