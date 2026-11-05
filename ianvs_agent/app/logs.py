# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

from collections import deque

from rich.ansi import AnsiDecoder
from rich.text import Text
from textual.widgets import RichLog


class RunOutputBuffer:
    def __init__(self):
        self.pending = {}
        self.decoders = {}

    def feed(self, run_id, channel, text):
        key = (run_id, channel)
        decoder = self.decoders.setdefault(key, AnsiDecoder())
        lines = (self.pending.get(key, "") + text).split("\n")
        pending = lines.pop()
        self.pending[key] = pending.rstrip("\r").rsplit("\r", 1)[-1] + ("\r" if pending.endswith("\r") else "")
        return [decoder.decode_line(line.rstrip("\r")) for line in lines]

    def preview(self):
        preview = Text()
        for key, line in self.pending.items():
            if line:
                decoder = AnsiDecoder()
                decoder.style = self.decoders[key].style
                if preview:
                    preview.append("\n")
                preview.append(f"{key[0]} · {key[1]}\n", style="#93c5fd")
                preview.append(decoder.decode_line(line.rstrip("\r")))
        return preview

    def finish(self, run_id):
        completed = []
        for key in list(self.pending):
            if key[0] == run_id:
                line = self.pending.pop(key)
                decoder = self.decoders.pop(key)
                if line:
                    completed.append((key, decoder.decode_line(line.rstrip("\r"))))
        return completed


class ReflowRichLog(RichLog):
    def __init__(self, *args, **kwargs):
        self._entries = deque(maxlen=kwargs.get("max_lines"))
        self._render_width = 0
        self._needs_reflow = False
        super().__init__(*args, **kwargs)

    def write(self, content, **kwargs):
        content = content.copy() if isinstance(content, Text) else content
        self._entries.append((content, kwargs))
        width = self.scrollable_content_region.width
        if width > 0 and self._size_known:
            if self._needs_reflow or width != self._render_width:
                self.reflow()
            else:
                super().write(content, **{**kwargs, "width": width})
        else:
            self._needs_reflow = True
        return self

    def clear(self):
        self._entries.clear()
        self._needs_reflow = False
        return super().clear()

    def on_resize(self, event):
        super().on_resize(event)
        self.call_after_refresh(self.reflow)

    def reflow(self):
        width = self.scrollable_content_region.width
        if width <= 0 or not self._size_known or (width == self._render_width and not self._needs_reflow):
            return
        self._render_width = width
        self._needs_reflow = False
        super().clear()
        for content, kwargs in self._entries:
            super().write(content, **{**kwargs, "width": width})
