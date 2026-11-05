# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys

from core.common.utils import load_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module")
    args = parser.parse_args()
    load_module(args.module)
    module = sys.modules[args.module]
    print(json.dumps({"origin": module.__spec__.origin, "file": module.__file__}), flush=True)


if __name__ == "__main__":
    main()
