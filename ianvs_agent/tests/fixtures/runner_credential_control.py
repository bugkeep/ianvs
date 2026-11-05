# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

def main():
    # 该进程只验证凭据传递与输出处理，不执行 Ianvs 评测。
    specification = importlib.util.spec_from_file_location("control_events", sys.argv[1])
    emitter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(emitter)
    value = os.environ["IANVS_CONTROL_BOUND"]
    forbidden = ("IANVS_CONTROL_SOURCE", "IANVS_CONTROL_UNUSED", "IANVS_CONTROL_UNUSED_SOURCE", "IANVS_CONTROL_PROVIDER")
    if any(name in os.environ for name in forbidden):
        raise RuntimeError("未绑定的配置引用进入子进程")
    representations = (value, quote(value, safe=""), json.dumps(value, ensure_ascii=False)[1:-1], json.dumps(value, ensure_ascii=True)[1:-1])
    for descriptor in (1, 2):
        for representation in representations:
            data = ("输出开始：" + representation + "：中文结束\n").encode("utf-8")
            for index in range(0, len(data), 3):
                os.write(descriptor, data[index:index + 3])
                time.sleep(0.001)
    emitter.emit_event("credential.control", values=list(representations), nested={value: {"原文": value}}, evidence_type="real_process_control_for_credential_handling")
    output = Path(os.environ["IANVS_RUN_WORKSPACE"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "clean.txt").write_text("真实凭据输出控制已执行\n", encoding="utf-8")
    mode = sys.argv[2]
    if mode == "artifact":
        (output / "restricted.txt").write_text(json.dumps({"value": value}), encoding="utf-8")
        (Path(os.environ["TMPDIR"]) / "restricted-temporary.txt").write_text(value, encoding="utf-8")
    if mode == "timeout":
        print("等待运行取消", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
