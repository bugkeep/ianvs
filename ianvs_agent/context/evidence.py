# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

from markdown_it import MarkdownIt


class ReadmeEvidence:
    def __init__(self, output_chars=16000):
        self.output_chars = output_chars
        self.parser = MarkdownIt()

    def read(self, path):
        source = Path(path).resolve()
        if not source.is_file():
            return {"status": "missing", "source": str(source), "content": None, "sections": []}
        text = source.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        tokens = self.parser.parse(text)
        headings = []
        for index, token in enumerate(tokens):
            if token.type == "heading_open" and token.map is not None:
                headings.append({"title": tokens[index + 1].content, "start_line": token.map[0] + 1, "level": int(token.tag[1:])})
        for index, heading in enumerate(headings):
            heading["end_line"] = headings[index + 1]["start_line"] - 1 if index + 1 < len(headings) else len(lines)
        if len(text) <= self.output_chars:
            return {"status": "available", "source": str(source), "content": text, "sections": headings, "ranges": [[1, len(lines)]], "truncated": False, "revision": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        wanted = ("install", "require", "depend", "prepar", "config", "quick", "run", "usage", "metric", "result", "environment", "数据", "环境", "依赖", "准备", "安装", "配置", "运行", "指标", "结果", "介绍", "说明")
        selected = [heading for heading in headings if any(word in heading["title"].lower() for word in wanted)]
        ranges = []
        parts = []
        used = 0
        prefix_end = headings[1]["start_line"] - 1 if len(headings) > 1 else min(len(lines), 30)
        intro = "".join(lines[:prefix_end])[:min(2000, self.output_chars)]
        if intro:
            parts.append(intro)
            ranges.append([1, min(prefix_end, len(intro.splitlines()))])
            used += len(intro)
        for heading in selected:
            if heading["end_line"] <= prefix_end:
                continue
            start = max(heading["start_line"], prefix_end + 1)
            section = "".join(lines[start - 1:heading["end_line"]])
            available = self.output_chars - used
            if available <= 0:
                break
            excerpt = section[:available]
            parts.append(f"\n[原文第 {start} 行起]\n" + excerpt)
            end = start + len(excerpt.splitlines()) - 1
            ranges.append([start, end])
            used += len(excerpt)
        return {
            "status": "available",
            "source": str(source),
            "revision": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "content": "\n".join(parts),
            "sections": headings,
            "ranges": ranges,
            "truncated": True,
            "continuation": "使用 read_file 按 source 与行范围继续读取，未提供的章节保持未知",
        }

    @staticmethod
    def locate(task, state, project_root):
        target = task.get("target") or {}
        instance = task.get("instance") or {}
        explicit = task.get("readme_path") or target.get("readme_path") or instance.get("readme_path") or state.get("readme_path")
        root = Path(task.get("code_root") or target.get("code_root") or instance.get("code_root") or state.get("code_root") or project_root).resolve()
        if explicit:
            path = Path(explicit)
            source = (path if path.is_absolute() else root / path).resolve()
            return source if source.is_relative_to(root) else None
        benchmark = task.get("benchmark_path") or target.get("benchmark_path") or target.get("config")
        if benchmark is None:
            return None
        config = Path(benchmark)
        if not config.is_absolute():
            config = root / config
        directory = config.parent.resolve()
        if not directory.is_relative_to(root):
            return None
        while directory != root.parent:
            matches = sorted(path for path in directory.iterdir() if path.is_file() and path.name.lower() == "readme.md") if directory.is_dir() else []
            if matches:
                for source in matches:
                    resolved = source.resolve()
                    if resolved.is_relative_to(root):
                        return resolved
            if directory == root:
                break
            directory = directory.parent
        return None
