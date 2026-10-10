# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from ianvs_agent.runtime.commands import EnvironmentCommands
from ianvs_agent.runtime.credentials import credential_status
from ianvs_agent.runtime.environment import environment_summary


def check_summary(check):
    fields = ("check_id", "instance_id", "task_id", "environment_id", "generation", "status", "passed", "checked_at", "evidence_ref", "source_kind", "python_executable", "dependency_fingerprint", "requirements_files", "file_versions", "pip_check", "imports", "log_ref", "package_drift", "drift_reason", "error")
    result = {key: check[key] for key in fields if key in check}
    result["package_count"] = len(check.get("packages") or [])
    dependencies = check.get("dependencies") or {}
    result["dependencies"] = {**dependencies, "requirements": dependencies.get("requirements", [])[:30]}
    result["dependencies"]["requirement_count"] = len(dependencies.get("requirements") or [])
    result["continuation"] = "完整包版本及依赖记录使用 read_evidence 读取 evidence_ref"
    return result


def plan_summary(plan):
    fields = ("plan_id", "instance_id", "task_id", "action", "status", "requires_install", "evidence_ref", "base_environment_id", "base_generation", "candidate_environment_id", "candidate_generation", "request", "commands", "requirements_file_versions", "dependency_fingerprint", "resolution", "resolution_ref", "resolved_packages", "constraints_ref", "error")
    result = {key: plan[key] for key in fields if key in plan}
    result["source_file_count"] = len(plan.get("file_versions") or {})
    result["continuation"] = "完整输入版本与安装依据使用 read_evidence 读取 evidence_ref"
    return result


def register_operations(harness, registry, code):
    harness.environment_commands = EnvironmentCommands(harness.settings, harness.store, harness.events)
    string = {"type": "string", "minLength": 1}
    strings = {"type": "array", "items": string, "uniqueItems": True}

    def fresh():
        return harness.instances.get(code.instance["instance_id"])

    async def refresh():
        harness.current_instance = await asyncio.to_thread(fresh)
        harness.events.bind(environment_id=(harness.current_instance.get("environment") or {}).get("environment_id"))
        if harness.current_instance.get("needs_verification"):
            harness.state["verification"] = harness.current_instance.get("verification") or {"passed": False, "status": "needs_verification"}
        await harness._save_session_async()

    async def inspect_environment(requirements_files=None, modules=None):
        try:
            return check_summary(await harness.environment.inspect(await asyncio.to_thread(fresh), requirements_files, modules))
        finally:
            await refresh()

    async def install_approval(action, plan_id=None, **parameters):
        if action != "apply" or not plan_id:
            raise ValueError("安装操作需要已生成的 plan_id")
        if parameters:
            raise ValueError("apply 仅接受 action 和 plan_id；变更安装内容需要生成新计划")
        return await asyncio.to_thread(harness.environment.revalidate, fresh(), plan_id)

    async def install_dependencies(action, plan_id=None, _approved=None, **parameters):
        if action == "plan":
            if plan_id:
                raise ValueError("生成依赖计划时不接受 plan_id")
            return plan_summary(await harness.environment.plan_install(await asyncio.to_thread(fresh), **parameters))
        if parameters or not plan_id or _approved is None or _approved["plan_id"] != plan_id:
            raise ValueError("apply 需要已审批的 plan_id，不能附带新的安装参数")
        try:
            result = await harness.environment.install(await asyncio.to_thread(fresh), plan_id)
            return {**environment_summary(result), "postcheck": check_summary(result["postcheck"])}
        finally:
            await refresh()

    async def prepare_approval():
        instance = await asyncio.to_thread(fresh)
        plan = await asyncio.to_thread(harness.environment.preview_prepare, instance)
        return await asyncio.to_thread(harness.environment.revalidate, instance, plan["plan_id"])

    async def prepare_instance(_approved=None):
        if _approved is None:
            raise PermissionError("准备环境需要具体计划的审批")
        try:
            result = await harness.environment.prepare(await asyncio.to_thread(fresh), plan_id=_approved["plan_id"])
            return {**environment_summary(result), "postcheck": check_summary(result["postcheck"])}
        finally:
            await refresh()

    async def command_approval(**parameters):
        return await asyncio.to_thread(harness.environment_commands.preview, await asyncio.to_thread(fresh), **parameters)

    async def execute_environment_command(_approved=None, **parameters):
        if _approved is None:
            raise PermissionError("环境命令需要具体参数和版本的审批")
        try:
            return await harness.environment_commands.execute(await asyncio.to_thread(fresh), approved=_approved, **parameters)
        finally:
            await refresh()

    async def credential_approval(credential_names):
        configured = harness.settings.runtime_credentials
        unknown = set(credential_names) - set(configured)
        if unknown:
            raise PermissionError("运行凭据尚未获 YAML 配置授权：" + ", ".join(sorted(unknown)))
        instance = await asyncio.to_thread(fresh)
        instance["credential_bindings"] = {name: configured[name] for name in credential_names}
        selected = credential_status(harness.settings, instance)
        if any(not item["available"] for item in selected):
            raise ValueError("运行凭据未在后端环境变量中配置：" + ", ".join(item["source_env"] for item in selected if not item["available"]))
        return {"instance_id": instance["instance_id"], "previous_bindings": (await asyncio.to_thread(fresh)).get("credential_bindings", {}), "credential_bindings": instance["credential_bindings"], "availability": selected}

    async def configure_runtime_environment(credential_names, _approved=None):
        if _approved is None:
            raise PermissionError("绑定运行凭据需要审批")
        instance = await asyncio.to_thread(fresh)
        instance["credential_bindings"] = _approved["credential_bindings"]
        instance.update(needs_verification=True, status="needs_verification")
        if instance.get("verification"):
            instance["verification"]["stale"] = True
        harness.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        harness.events.emit("environment.credentials_configured", _approved, instance_id=instance["instance_id"])
        await refresh()
        return _approved

    registry.register("inspect_environment", "核查当前实例的真实 Python、包版本、pip check、requirements 和指定模块导入；保存正式检查和日志。检查成功仅表示环境准备成功。", {"requirements_files": strings, "modules": strings}, [], inspect_environment)
    registry.register("install_dependencies", "依赖安装到当前实例独立环境。action=plan 时提供 requirements_files/packages/constraints_files/index_url/timeout/modules，核查真实声明并保存具体 pip 计划；action=apply 时仅传 plan_id，审批后创建新环境代次、执行真实安装及检查，成功后激活；失败保留原环境。", {"action": {"enum": ["plan", "apply"]}, "plan_id": string, "requirements_files": strings, "packages": strings, "constraints_files": strings, "index_url": string, "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": 1800}, "modules": strings}, ["action"], install_dependencies, modifies=lambda args: not isinstance(args, dict) or args.get("action") != "plan", approval_context=install_approval)
    registry.register("prepare_instance", "按当前实例 YAML 声明生成具体准备计划，审批后准备独立环境，保留安装、检查和日志；未声明依赖通过 install_dependencies 纳入。", {}, [], prepare_instance, modifies=True, approval_context=prepare_approval)
    registry.register("execute_environment_command", "在当前实例 Python 环境执行参数列表命令；源码、验证程序和现有环境只读，可指定实例内新生成目录；默认禁止网络。需要审批实际命令、文件版本、生成目录和网络权限。pip 安装请使用 install_dependencies。", {"argv": {**strings, "minItems": 1, "uniqueItems": False}, "cwd": string, "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": 1800}, "writable_directories": strings, "network": {"type": "boolean"}}, ["argv"], execute_environment_command, modifies=True, approval_context=command_approval)
    registry.register("configure_runtime_environment", "仅将 YAML runtime_credentials 允许的环境变量引用绑定到当前实例，后端运行时加载；工具仅接收名称，禁止提供凭据值。空列表清除绑定。", {"credential_names": strings}, ["credential_names"], configure_runtime_environment, modifies=True, approval_context=credential_approval)
