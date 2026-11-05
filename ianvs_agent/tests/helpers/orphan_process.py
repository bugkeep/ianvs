# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import os
import time


middle = os.fork()
if middle == 0:
    leaf = os.fork()
    if leaf == 0:
        time.sleep(120)
        os._exit(0)
    print(leaf, flush=True)
    os._exit(0)
os.waitpid(middle, 0)
time.sleep(120)
