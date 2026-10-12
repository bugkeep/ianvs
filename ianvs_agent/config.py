# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class ModelProfile:
    name: str
    model: str
    base_url: str
    api_key_env: str = "ARK_API_KEY"
    context_window: int = 256000
    output_tokens: int = 8192
    auxiliary_output_tokens: int = 2048
    auxiliary_thinking: str | None = None
    timeout_seconds: float = 120
    request_timeout_seconds: float = 600
    max_retries: int = 2
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cache_price_per_million: float | None = None


@dataclass
class AgentSettings:
    project_root: Path
    state_root: Path
    python_executable: str = sys.executable
    selected_profile: str = "ark"
    profiles: dict[str, ModelProfile] = field(default_factory=dict)
    context_threshold: float = 0.78
    context_input_budget: int | None = None
    keep_recent_turns: int = 3
    # 保留既有配置字段用于统计参考；累计数值不终止任务。
    max_model_calls: int = 40
    max_tool_calls: int = 100
    max_input_tokens: int = 500000  # 自动压缩周期的输入参考量。
    max_output_tokens: int = 100000
    max_task_seconds: float = 1800
    max_repair_attempts: int = 5
    tool_output_chars: int = 16000
    run_timeout_seconds: float = 600
    pip_timeout_seconds: float = 60
    pip_retries: int = 5
    approval_policy: str = "manual"
    dependency_mode: str = "isolated"
    wsl_distribution: str = "Ubuntu-24.04"
    wsl_python_executable: str = "python3"
    runtime_credentials: dict[str, str] = field(default_factory=dict)

    @property
    def profile(self):
        return self.profiles[self.selected_profile]

    def public_dict(self):
        result = asdict(self)
        result["project_root"] = str(self.project_root)
        result["state_root"] = str(self.state_root)
        return result


def load_settings(project_root=None, config_path=None, *, load_credentials=True):
    root = Path(project_root or Path.cwd()).resolve()
    config = Path(config_path) if config_path else root / "workspace/ianvs-agent/config.yaml"
    if load_credentials:
        load_dotenv(root / ".env", override=False)
        load_dotenv(root / "workspace/ianvs-agent/.env", override=False)
    data = yaml.safe_load(config.read_text(encoding="utf-8")) if config.exists() else {}
    if data is None:
        data = {}
    profiles = data.pop("profiles", {
        "ark": {"name": "方舟主模型", "model": "ark-code-latest", "base_url": "https://ark.cn-beijing.volces.com/api/plan/v3", "auxiliary_thinking": "disabled"},
        "glm": {"name": "方舟 GLM", "model": "glm-5.3", "base_url": "https://ark.cn-beijing.volces.com/api/plan/v3", "context_window": 1024000},
    })
    resolved = {name: ModelProfile(**profile) for name, profile in profiles.items()}
    state_root = Path(data.pop("state_root", root / "workspace/ianvs-agent"))
    if not state_root.is_absolute():
        state_root = root / state_root
    settings = AgentSettings(project_root=root, state_root=state_root.resolve(), profiles=resolved, **data)
    if settings.selected_profile not in settings.profiles:
        raise ValueError("选中的模型配置不存在")
    if not 0 < settings.context_threshold < 1:
        raise ValueError("上下文压缩阈值必须在 0 和 1 之间")
    if isinstance(settings.pip_timeout_seconds, bool) or not isinstance(settings.pip_timeout_seconds, (int, float)) or not math.isfinite(settings.pip_timeout_seconds) or settings.pip_timeout_seconds <= 0:
        raise ValueError("pip 网络请求时限必须为有限正数")
    if isinstance(settings.pip_retries, bool) or not isinstance(settings.pip_retries, int) or settings.pip_retries < 0:
        raise ValueError("pip 网络重试次数必须为非负整数")
    if settings.context_input_budget is not None and (isinstance(settings.context_input_budget, bool) or not isinstance(settings.context_input_budget, int) or settings.context_input_budget <= 0):
        raise ValueError("单次上下文输入预算必须为空值或正整数")
    for profile in resolved.values():
        for value in (profile.timeout_seconds, profile.request_timeout_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("模型空闲与总请求时限必须为正数")
        for value in (profile.context_window, profile.output_tokens, profile.auxiliary_output_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("模型上下文和输出预算必须为正整数")
        if profile.output_tokens >= profile.context_window:
            raise ValueError("输出预算必须小于模型上下文窗口")
        if profile.auxiliary_output_tokens >= profile.context_window:
            raise ValueError("诊断和压缩输出预算必须小于模型上下文窗口")
        if profile.auxiliary_thinking not in (None, "enabled", "disabled"):
            raise ValueError("辅助请求思考配置必须为 enabled、disabled 或空值")
        if not profile.base_url.startswith("https://") and not profile.base_url.startswith("http://localhost") and not profile.base_url.startswith("http://127.0.0.1"):
            raise ValueError("模型地址需要 HTTPS 或本地服务地址")
    if not isinstance(settings.runtime_credentials, dict) or any(not isinstance(name, str) or not name.isidentifier() or not isinstance(reference, str) or not reference.isidentifier() for name, reference in settings.runtime_credentials.items()):
        raise ValueError("运行凭据必须使用环境变量名称映射，禁止保存凭据值")
    provider_credentials = {profile.api_key_env for profile in resolved.values()}
    if any(name in provider_credentials or reference in provider_credentials for name, reference in settings.runtime_credentials.items()):
        raise ValueError("模型服务凭据不能作为 Example 运行凭据")
    return settings
