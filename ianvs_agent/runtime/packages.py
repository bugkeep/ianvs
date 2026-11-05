# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

from packaging.utils import canonicalize_name


def package_identity(packages):
    if not isinstance(packages, list) or any(not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("version"), str) for item in packages):
        return None
    return sorted((canonicalize_name(item["name"]), item["version"]) for item in packages)
