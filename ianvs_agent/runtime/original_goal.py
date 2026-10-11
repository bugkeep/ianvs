# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import io
import json
import math
import os
import re
import uuid
from importlib.machinery import BYTECODE_SUFFIXES, EXTENSION_SUFFIXES
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml
from filelock import FileLock

from .instances import utc_now, validate_identity


def python_module_candidates(declared, manifest):
    if not isinstance(declared, str) or not declared or any(not part.isidentifier() for part in declared.split(".")):
        return [], []
    parts = declared.split(".")
    chain = []
    for index in range(1, len(parts) + 1):
        path = "/".join(parts[:index])
        if any(path + suffix in manifest or path + "/__init__" + suffix in manifest for suffix in EXTENSION_SUFFIXES + BYTECODE_SUFFIXES):
            return [], chain
    for index in range(1, len(parts)):
        parent = "/".join(parts[:index])
        package = parent + "/__init__.py"
        if package in manifest:
            chain.append({"kind": "package", "path": package})
        elif parent + ".py" in manifest:
            return [], chain
        elif any(name.startswith(parent + "/") for name in manifest):
            chain.append({"kind": "namespace", "path": parent})
        else:
            return [], chain
    name = "/".join(parts)
    return [candidate for candidate in (name + ".py", name + "/__init__.py") if candidate in manifest], chain


class OriginalGoalStore:
    _MODEL_FIELDS = {"initial_model_url", "model", "model_url", "model_path", "model_name", "model_identity", "model_id", "checkpoint", "checkpoint_path", "weights", "weights_path", "pretrained_model_name_or_path", "tokenizer", "tokenizer_path", "tokenizer_name_or_path"}
    _DATA_FIELDS = {"train_url", "test_url", "train_index", "test_index", "train_data", "test_data", "train_data_info", "test_data_info", "label"}

    def __init__(self, settings, store):
        self.settings, self.store = settings, store

    @staticmethod
    def _digest(value):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_file(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def _relative(value):
        if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
            raise ValueError("原目标文件路径无效")
        path = PurePosixPath(value)
        if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or path.as_posix() != value:
            raise ValueError("原目标文件路径需要实例内的规范相对路径")
        return value

    def _path(self, reference):
        self._relative(reference)
        path = self.store.root / reference
        if path.resolve() != path or not path.is_relative_to(self.store.root):
            raise PermissionError("原目标记录路径不属于程序管理目录")
        return path

    def _prefix(self, instance):
        return "instances/" + validate_identity(instance["instance_id"], "实例") + "/original-goals"

    def _manifest(self, instance):
        manifest = instance.get("source_manifest")
        if not isinstance(manifest, dict) or not manifest:
            return {}
        for name, digest in manifest.items():
            self._relative(name)
            if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
                raise ValueError("创建时文件摘要无效：" + name)
        return manifest

    @staticmethod
    def _object_reference(digest):
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("原目标对象摘要无效")
        return f"original-goals/objects/{digest[:2]}/{digest}.bin"

    def _object(self, digest, reference=None):
        expected = self._object_reference(digest)
        if reference is not None and reference != expected:
            raise PermissionError("原目标对象引用与文件摘要不一致")
        path = self._path(expected)
        if not path.is_file():
            raise FileNotFoundError("原目标归档对象缺失：" + expected)
        if self._hash_file(path) != digest:
            raise ValueError("原目标归档对象内容损坏：" + expected)
        return path

    def _archive(self, stream, digest):
        reference = self._object_reference(digest)
        destination = self._path(reference)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(destination) + ".lock"):
            if destination.exists():
                self._object(digest)
                return reference
            temporary = destination.with_name(uuid.uuid4().hex + ".part")
            observed = hashlib.sha256()
            try:
                with temporary.open("xb") as output:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        observed.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if observed.hexdigest() != digest:
                    raise ValueError("原目标来源字节与创建时文件摘要不同")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return reference

    def _write_once(self, reference, value):
        path = self._path(reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock"):
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != value:
                    raise ValueError("原目标不可变记录内容冲突：" + reference)
            else:
                temporary = path.with_name(uuid.uuid4().hex + ".part")
                try:
                    with temporary.open("x", encoding="utf-8", newline="\n") as output:
                        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
                        output.write("\n")
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)

    def _read_source(self, instance, reference):
        prefix = self._prefix(instance) + "/manifests/"
        if not isinstance(reference, str) or not reference.startswith(prefix) or not reference.endswith(".json"):
            raise PermissionError("原目标来源清单不属于当前实例")
        self._path(reference)
        value = self.store.read(reference)
        if not isinstance(value, dict) or value.get("instance_id") != instance["instance_id"]:
            raise ValueError("原目标来源清单身份无效")
        digest = self._digest(value)
        if reference != prefix + digest + ".json":
            raise ValueError("原目标来源清单内容损坏")
        manifest = self._manifest(instance)
        if value.get("source_manifest") != manifest or value.get("source_manifest_digest") != self._digest(manifest):
            raise ValueError("原目标来源清单与实例创建记录不同")
        for name, item in value.get("file_objects", {}).items():
            if name not in manifest or item.get("sha256") != manifest[name] or item.get("object_ref") != self._object_reference(manifest[name]):
                raise ValueError("原目标来源清单文件身份无效：" + name)
        return value

    def _save_source(self, instance, manifest, objects):
        record = {"schema_version": 1, "instance_id": instance["instance_id"], "source_manifest": manifest, "source_manifest_digest": self._digest(manifest), "file_objects": objects}
        reference = self._prefix(instance) + "/manifests/" + self._digest(record) + ".json"
        self._write_once(reference, record)
        instance["original_source_ref"] = reference
        return reference

    def capture(self, instance):
        manifest = self._manifest(instance)
        if not manifest:
            raise ValueError("新实例缺少创建时文件清单")
        root = Path(instance["code_root"]).resolve()
        objects = {}
        for name, digest in sorted(manifest.items()):
            path = root / name
            if path.resolve() != path or not path.is_file():
                raise ValueError("创建时原目标文件缺失或包含符号链接：" + name)
            with path.open("rb") as stream:
                reference = self._archive(stream, digest)
            # 已有去重对象仍需要证明当前新实例复制的是同一文件版本。
            if self._hash_file(path) != digest:
                raise ValueError("归档期间创建文件内容发生变化：" + name)
            objects[name] = {"sha256": digest, "object_ref": reference, "source_kind": "creation_snapshot", "evidence_ref": f"instances/{instance['instance_id']}/instance.json"}
        self._save_source(instance, manifest, objects)
        return self.resolve(instance)

    @staticmethod
    def _local_reference(instance, declared):
        if not isinstance(declared, str) or not declared or "\x00" in declared or "://" in declared:
            return None
        root = Path(instance["code_root"]).resolve()
        # 仅原本指向实例内的绝对路径可以转换为相对文件身份。
        if PureWindowsPath(declared).drive and os.name != "nt":
            return None
        path = Path(os.path.abspath(root / declared))
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else None

    def _candidates(self, instance, declared, marker, manifest, *, python_module=False):
        direct = self._local_reference(instance, declared)
        if direct in manifest:
            return [direct], "original_reference"
        if python_module and isinstance(declared, str) and declared and all(part.isidentifier() for part in declared.split(".")):
            candidates, _ = python_module_candidates(declared, manifest)
            return candidates, "python_module_reference"
        if not marker or not isinstance(declared, str) or "://" in declared:
            return [], "original_reference"
        parts = PurePosixPath(declared.replace("\\", "/")).parts
        if ".." in parts or parts.count(marker) != 1:
            return [], "original_reference"
        suffix = PurePosixPath(*parts[parts.index(marker):]).as_posix()
        example = PurePosixPath(instance["benchmark_path"]).parent
        candidates = [name for name in manifest if PurePosixPath(name).is_relative_to(example) and (name == suffix or name.endswith("/" + suffix))]
        return sorted(candidates), "approved_unique_example_suffix"

    def _history_sources(self, instance, name, digest):
        prefix = f"instances/{instance['instance_id']}"
        specification = instance.get("task_spec") or {}
        if name == instance["benchmark_path"] and isinstance(specification.get("benchmark_text"), str):
            yield specification["benchmark_text"].encode("utf-8"), "initial_task_text", prefix + "/instance.json"
        benchmark = specification.get("benchmark")
        if not isinstance(benchmark, dict) and isinstance(specification.get("benchmark_text"), str):
            try:
                benchmark = yaml.safe_load(specification["benchmark_text"])
            except yaml.YAMLError:
                benchmark = None
        job = benchmark.get("benchmarkingjob") if isinstance(benchmark, dict) else None
        if isinstance(job, dict) and isinstance(specification.get("testenv_text"), str):
            candidates, _ = self._candidates(instance, job.get("testenv"), "testenv", self._manifest(instance))
            if candidates == [name]:
                yield specification["testenv_text"].encode("utf-8"), "initial_task_text", prefix + "/instance.json"
        for path in sorted(self.store.path(prefix + "/changes").glob("*/change.json")):
            record = self.store.read(path.relative_to(self.store.root).as_posix())
            for change in record.get("changes", []):
                if change.get("path") == name and isinstance(change.get("before"), str):
                    yield change["before"].encode("utf-8"), "patch_before", path.relative_to(self.store.root).as_posix()
        for path in sorted(self.store.path(prefix + "/runs").glob("*/run.json")):
            record = self.store.read(path.relative_to(self.store.root).as_posix())
            if (record.get("source_manifest") or {}).get(name) != digest:
                continue
            snapshot = record.get("config_snapshot") or {}
            if name == instance["benchmark_path"] and isinstance(snapshot.get("benchmark_text"), str):
                yield snapshot["benchmark_text"].encode("utf-8"), "run_configuration", path.relative_to(self.store.root).as_posix()
            job = (snapshot.get("benchmark") or {}).get("benchmarkingjob") or {}
            if self._local_reference(instance, job.get("testenv")) == name and isinstance(snapshot.get("testenv_text"), str):
                yield snapshot["testenv_text"].encode("utf-8"), "run_configuration", path.relative_to(self.store.root).as_posix()

    def _recover(self, instance, name, manifest, objects):
        digest = manifest[name]
        if name in objects:
            self._object(digest, objects[name]["object_ref"])
            return objects[name]
        for field, kind in (("code_root", "legacy_instance_file"), ("source_root", "legacy_source_file")):
            if not instance.get(field):
                continue
            root = Path(instance[field]).resolve()
            path = root / name
            if path.resolve() != path or not path.is_file() or self._hash_file(path) != digest:
                continue
            with path.open("rb") as stream:
                reference = self._archive(stream, digest)
            objects[name] = {"sha256": digest, "object_ref": reference, "source_kind": kind, "evidence_ref": str(path)}
            return objects[name]
        for data, kind, evidence in self._history_sources(instance, name, digest):
            if hashlib.sha256(data).hexdigest() == digest:
                reference = self._archive(io.BytesIO(data), digest)
                objects[name] = {"sha256": digest, "object_ref": reference, "source_kind": kind, "evidence_ref": evidence}
                return objects[name]
        return None

    @classmethod
    def _json_value(cls, value):
        if value is None or isinstance(value, (str, bool, int)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(cls._json_value(item) for item in value)
        return isinstance(value, dict) and all(isinstance(key, str) and cls._json_value(item) for key, item in value.items())

    def _bind(self, instance, declared, field, marker, manifest, objects, bindings, unresolved, *, document=False, python_module=False):
        candidates, basis = self._candidates(instance, declared, marker, manifest, python_module=python_module)
        if len(candidates) != 1:
            unresolved.append({"field": field, "declared": declared, "reason": "ambiguous_reference" if candidates else "creation_source_missing", "candidates": candidates})
            return None
        name = candidates[0]
        original = self._recover(instance, name, manifest, objects)
        if original is None:
            unresolved.append({"field": field, "declared": declared, "path": name, "reason": "creation_bytes_unavailable", "expected_sha256": manifest[name]})
            return None
        item = {"field": field, "declared": declared, "path": name, "binding_basis": basis, **original}
        if basis == "python_module_reference":
            _, item["python_module_chain"] = python_module_candidates(declared, manifest)
        bindings.append(copy.deepcopy(item))
        if document:
            data = self._object(item["sha256"], item["object_ref"]).read_bytes()
            try:
                text = data.decode("utf-8")
                parsed = yaml.safe_load(text)
            except (UnicodeDecodeError, yaml.YAMLError) as error:
                unresolved.append({"field": field, "path": name, "reason": "unsupported_document", "error": str(error)})
                return {**item, "text": None if isinstance(error, UnicodeDecodeError) else text, "document": None}
            if not isinstance(parsed, dict) or not self._json_value(parsed):
                unresolved.append({"field": field, "path": name, "reason": "unsupported_document"})
                parsed = None
            return {**item, "text": text, "document": parsed}
        return item

    def _artifacts(self, instance, value, field, manifest, objects, result, unresolved, fields, selected=False):
        if isinstance(value, dict):
            for key, item in value.items():
                self._artifacts(instance, item, field + "/" + key, manifest, objects, result, unresolved, fields, selected or key in fields)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._artifacts(instance, item, field + "/" + str(index), manifest, objects, result, unresolved, fields, selected)
        elif selected and isinstance(value, str):
            name = self._local_reference(instance, value)
            directory = name if name and name not in manifest else None
            members = [name] if name in manifest else sorted(item for item in manifest if directory and item.startswith(directory.rstrip("/") + "/"))
            for member in members:
                original = self._recover(instance, member, manifest, objects)
                if original:
                    item = {"field": field, "declared": value, "path": member, **original}
                    if directory:
                        item["artifact_directory"] = directory
                    result.append(item)
                else:
                    unresolved.append({"field": field, "path": member, "reason": "creation_bytes_unavailable", "expected_sha256": manifest[member]})

    def _check_initial_text(self, instance, item, field, unresolved):
        specification = instance.get("task_spec") or {}
        text = specification.get(field)
        document = specification.get(field.removesuffix("_text"))
        if item and item.get("document") is not None and document is not None and document != item["document"]:
            unresolved.append({"field": "task_spec/" + field.removesuffix("_text"), "reason": "initial_record_conflict"})
        if not text or not item or item.get("document") is None:
            return
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError:
            unresolved.append({"field": "task_spec/" + field, "reason": "initial_record_conflict"})
            return
        if parsed != item["document"]:
            unresolved.append({"field": "task_spec/" + field, "reason": "initial_record_conflict"})

    def _metric(self, instance, metric, field, manifest, objects, bindings, unresolved, sources):
        if not isinstance(metric, dict) or not isinstance(metric.get("name"), str) or not metric["name"] or "url" in metric and not isinstance(metric["url"], str):
            unresolved.append({"field": field, "reason": "unsupported_metrics"})
            return
        declared = metric.get("url")
        if declared:
            item = self._bind(instance, declared, field + "/url", "testenv", manifest, objects, bindings, unresolved)
        else:
            item = self._bind(instance, "core/testcasecontroller/metrics/metrics.py", field + "/url", None, manifest, objects, bindings, unresolved)
        if item:
            sources.append({**item, "name": metric["name"], "metric_kind": "custom_metric" if declared else "builtin_metric"})

    @staticmethod
    def _contents(goal):
        return {key: value for key, value in goal.items() if key not in {"goal_id", "goal_digest", "evidence_ref", "created_at", "previous_goal_id", "previous_goal_ref"}}

    def resolve(self, instance):
        prefix = self._prefix(instance)
        previous = self.load(instance) if instance.get("original_goal_ref") else None
        if previous and previous["status"] == "complete":
            return previous
        manifest = self._manifest(instance)
        source = self._read_source(instance, instance["original_source_ref"]) if instance.get("original_source_ref") else None
        objects = copy.deepcopy(source["file_objects"]) if source else {}
        bindings, unresolved = [], []
        if not manifest:
            unresolved.append({"field": "source_manifest", "reason": "creation_manifest_missing"})
        benchmark = self._bind(instance, instance["benchmark_path"], "benchmark", None, manifest, objects, bindings, unresolved, document=True)
        self._check_initial_text(instance, benchmark, "benchmark_text", unresolved)
        job = ((benchmark or {}).get("document") or {}).get("benchmarkingjob")
        testenv, algorithms, metric_sources, artifacts = None, [], [], []
        if not isinstance(job, dict):
            unresolved.append({"field": "benchmarkingjob", "reason": "unsupported_configuration"})
            job = {}
        if isinstance(job.get("testenv"), str):
            testenv = self._bind(instance, job["testenv"], "benchmarkingjob/testenv", "testenv", manifest, objects, bindings, unresolved, document=True)
            self._check_initial_text(instance, testenv, "testenv_text", unresolved)
        else:
            unresolved.append({"field": "benchmarkingjob/testenv", "reason": "unsupported_configuration"})
        test_object = job.get("test_object") or {}
        declared_algorithms = test_object.get("algorithms") if isinstance(test_object, dict) else None
        if not isinstance(test_object, dict) or test_object.get("type") != "algorithms" or not isinstance(declared_algorithms, list) or not declared_algorithms:
            unresolved.append({"field": "benchmarkingjob/test_object", "reason": "unsupported_configuration"})
            declared_algorithms = []
        for index, declaration in enumerate(declared_algorithms):
            field = f"benchmarkingjob/test_object/algorithms/{index}/url"
            if not isinstance(declaration, dict) or not isinstance(declaration.get("name"), str) or not isinstance(declaration.get("url"), str):
                unresolved.append({"field": field, "reason": "unsupported_configuration"})
                continue
            item = self._bind(instance, declaration["url"], field, "testalgorithms", manifest, objects, bindings, unresolved, document=True)
            if item:
                item.update(name=declaration["name"], declared_url=declaration["url"])
                algorithms.append(item)
                algorithm = (item.get("document") or {}).get("algorithm")
                if not isinstance(algorithm, dict) or not isinstance(algorithm.get("paradigm_type"), str):
                    unresolved.append({"field": field, "reason": "unsupported_configuration"})
                else:
                    self._artifacts(instance, algorithm, f"algorithms/{index}/algorithm", manifest, objects, artifacts, unresolved, self._MODEL_FIELDS)
                    modules = algorithm.get("modules")
                    if not isinstance(modules, list):
                        unresolved.append({"field": field + "/algorithm/modules", "reason": "unsupported_configuration"})
                    else:
                        for module_index, module in enumerate(modules):
                            if not isinstance(module, dict):
                                unresolved.append({"field": field + "/algorithm/modules", "reason": "unsupported_configuration"})
                            elif module.get("url"):
                                self._bind(instance, module["url"], f"algorithms/{index}/algorithm/modules/{module_index}/url", "testalgorithms", manifest, objects, bindings, unresolved, python_module=True)
        environment = ((testenv or {}).get("document") or {}).get("testenv")
        if not isinstance(environment, dict):
            unresolved.append({"field": "testenv", "reason": "unsupported_configuration"})
            environment = {}
        dataset, metrics = environment.get("dataset"), environment.get("metrics")
        if not isinstance(dataset, dict):
            unresolved.append({"field": "testenv/dataset", "reason": "unsupported_configuration"})
        else:
            self._artifacts(instance, dataset, "testenv/dataset", manifest, objects, artifacts, unresolved, self._DATA_FIELDS)
        if not isinstance(metrics, list) or not metrics or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in metrics):
            unresolved.append({"field": "testenv/metrics", "reason": "unsupported_metrics"})
        else:
            for index, metric in enumerate(metrics):
                self._metric(instance, metric, f"testenv/metrics/{index}", manifest, objects, bindings, unresolved, metric_sources)
        if "model_eval" in environment:
            model_eval = environment["model_eval"]
            if not isinstance(model_eval, dict) or "model_metric" not in model_eval:
                unresolved.append({"field": "testenv/model_eval", "reason": "unsupported_configuration"})
            else:
                self._metric(instance, model_eval["model_metric"], "testenv/model_eval/model_metric", manifest, objects, bindings, unresolved, metric_sources)
        source_ref = self._save_source(instance, manifest, objects) if manifest else None
        reasons = {item["reason"] for item in unresolved}
        status = "complete" if not reasons else "needs_binding" if reasons & {"ambiguous_reference", "initial_record_conflict"} else "source_missing" if reasons & {"creation_manifest_missing", "creation_source_missing", "creation_bytes_unavailable"} else "unsupported"
        goal = {"schema_version": 1, "instance_id": instance["instance_id"], "source_manifest_digest": self._digest(manifest), "source_manifest_ref": source_ref, "status": status, "documents": {"benchmark": benchmark, "testenv": testenv, "algorithms": algorithms}, "bindings": bindings, "requirements": {"benchmark": (benchmark or {}).get("document"), "testenv": (testenv or {}).get("document"), "algorithms": [{"name": item["name"], "document": item["document"]} for item in algorithms], "dataset": dataset, "metrics": metrics}, "metric_sources": metric_sources, "artifact_bindings": artifacts, "unresolved": unresolved, "supported_checks": ["creation_file_identity", "configuration_declarations", "metric_source_identity"], "limitations": ["配置来源完整仅证明创建时声明与文件身份；模型加载、数据内容与实际指标仍需本次运行和独立验证证据", "创建清单未包含的模型与数据制品仅保留原声明，不推断其内容身份"]}
        if any(item["binding_basis"] == "python_module_reference" for item in bindings):
            goal["limitations"].append("Python 模块声明绑定创建源码候选；实际 import origin 与 namespace 搜索位置仍需运行证据")
        if previous and self._contents(previous) == goal:
            return previous
        identity = self.store.allocate_id("goal")
        reference = prefix + "/" + identity + ".json"
        goal.update(goal_id=identity, evidence_ref=reference, created_at=utc_now(), previous_goal_id=previous["goal_id"] if previous else None, previous_goal_ref=previous["evidence_ref"] if previous else None)
        goal["goal_digest"] = self._digest(goal)
        self._write_once(reference, goal)
        instance.update(original_goal_ref=reference, goal_id=identity, goal_digest=goal["goal_digest"], needs_verification=True)
        if instance.get("verification"):
            instance["verification"]["stale"] = True
        self.store.write(f"instances/{instance['instance_id']}/instance.json", instance)
        return goal

    def load(self, instance, reference=None):
        current = reference is None
        reference = reference or instance.get("original_goal_ref")
        if not reference:
            return None
        prefix = self._prefix(instance) + "/"
        path = self._path(reference)
        if path.parent != self.store.root / prefix:
            raise PermissionError("原目标记录不属于当前实例")
        goal = self.store.read(reference)
        if not isinstance(goal, dict) or goal.get("instance_id") != instance["instance_id"]:
            raise ValueError("原目标记录身份无效")
        identity = validate_identity(goal.get("goal_id"), "原目标")
        if reference != prefix + identity + ".json" or goal.get("evidence_ref") != reference:
            raise ValueError("原目标记录引用无效")
        digest = self._digest({key: value for key, value in goal.items() if key != "goal_digest"})
        if digest != goal.get("goal_digest") or current and (instance.get("goal_id") != identity or instance.get("goal_digest") != digest):
            raise ValueError("原目标记录摘要与正式绑定不同")
        if goal.get("source_manifest_digest") != self._digest(self._manifest(instance)):
            raise ValueError("原目标记录与创建时文件清单不同")
        if goal.get("source_manifest_ref"):
            source = self._read_source(instance, goal["source_manifest_ref"])
            for item in [*goal.get("bindings", []), *goal.get("artifact_bindings", [])]:
                if item.get("path") not in source["file_objects"] or source["source_manifest"].get(item["path"]) != item.get("sha256"):
                    raise ValueError("原目标绑定没有对应创建来源")
                self._object(item["sha256"], item["object_ref"])
                if item.get("binding_basis") == "python_module_reference":
                    candidates, chain = python_module_candidates(item["declared"], source["source_manifest"])
                    if candidates != [item["path"]] or chain != item.get("python_module_chain"):
                        raise ValueError("原目标 Python 模块来源与创建清单不同")
            directories = {(item["field"], item["artifact_directory"]) for item in goal.get("artifact_bindings", []) if item.get("artifact_directory")}
            for field, directory in directories:
                self._relative(directory)
                expected = {name for name in source["source_manifest"] if name.startswith(directory + "/")}
                recorded = {item["path"] for item in goal["artifact_bindings"] if item["field"] == field and item.get("artifact_directory") == directory}
                if not expected or expected != recorded:
                    raise ValueError("原目标模型或数据目录的创建文件集合不同")
            for item in [goal["documents"].get("benchmark"), goal["documents"].get("testenv"), *goal["documents"].get("algorithms", [])]:
                if item and isinstance(item.get("text"), str) and hashlib.sha256(item["text"].encode("utf-8")).hexdigest() != item["sha256"]:
                    raise ValueError("原目标配置文本与归档字节不同")
            conflicts = []
            self._check_initial_text(instance, goal["documents"].get("benchmark"), "benchmark_text", conflicts)
            self._check_initial_text(instance, goal["documents"].get("testenv"), "testenv_text", conflicts)
            if conflicts and not any(item.get("reason") == "initial_record_conflict" for item in goal.get("unresolved", [])):
                raise ValueError("原目标配置与初始 task_spec 内容冲突")
        elif goal.get("status") == "complete":
            raise ValueError("完整原目标缺少创建来源清单")
        return goal

    def bound_object(self, instance, reference, original_goal=None):
        self._relative(reference)
        if original_goal is not None:
            if not isinstance(original_goal, dict) or not original_goal.get("evidence_ref"):
                raise ValueError("任务原目标绑定缺少正式引用")
            goal = self.load(instance, original_goal["evidence_ref"])
            if any(original_goal.get(key) != goal.get(key) for key in ("goal_id", "goal_digest", "evidence_ref")):
                raise ValueError("任务原目标绑定与正式版本不同")
        else:
            goal = self.load(instance)
        binding = next((item for item in (goal or {}).get("bindings", []) if item.get("object_ref") == reference), None)
        if binding is None:
            raise PermissionError("只能读取当前任务原目标已绑定的来源对象")
        if self._manifest(instance).get(binding["path"]) != binding["sha256"]:
            raise ValueError("原目标来源对象与创建时文件摘要不同")
        path = self._object(binding["sha256"], reference)
        return path, {"path": binding["path"], "file_version": binding["sha256"], "object_ref": reference, "source_kind": "creation_snapshot", "original_goal_ref": goal["evidence_ref"]}
