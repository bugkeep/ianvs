# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--validator-root", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    validator_path = Path(args.validator_root) / ".github/workflows/validator/static_validator.py"
    spec = importlib.util.spec_from_file_location("ianvs_independent_static_validator", validator_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    entry = json.loads(Path(args.entry).read_text(encoding="utf-8"))
    report = module.validate_examples(Path(args.repo_root), [entry])
    Path(args.report).write_text(module.render_json(report), encoding="utf-8")


if __name__ == "__main__":
    main()
