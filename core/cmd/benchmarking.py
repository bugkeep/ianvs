# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""main"""
# pylint: disable=import-outside-toplevel

import sys
import argparse
import os

from core.__version__ import __version__
from core.common.events import emit_event, stage


def main():
    """ main command-line interface to ianvs"""
    parser = _generate_parser()
    args = parser.parse_args()
    if args.agent:
        if sys.version_info < (3, 11):
            raise SystemExit("Ianvs Agent 需要 Python 3.11 或更新版本")
        from ianvs_agent.__main__ import start_agent
        start_agent(config_path=args.agent_config, backend=args.agent_backend,
                    distribution=args.agent_wsl_distro,
                    python_executable=args.agent_backend_python)
        return
    try:
        stage("runtime_import")
        from core.common.log import LOGGER
        from core.common import utils
        from core.cmd.obj import BenchmarkingJob

        stage("configuration_load", benchmark_path=args.benchmarking_config_file)
        config_file = args.benchmarking_config_file
        if not utils.is_local_file(config_file):
            raise SystemExit(f"not found benchmarking config({config_file}) file in local")

        config = utils.yaml2dict(args.benchmarking_config_file)
        workspace = os.environ.get("IANVS_RUN_WORKSPACE")
        if workspace:
            config["benchmarkingjob"]["workspace"] = workspace
        job = BenchmarkingJob(config[str.lower(BenchmarkingJob.__name__)])
        job.run()

        LOGGER.info("benchmarkingjob runs successfully.")
        emit_event("run.pipeline_completed", workspace=job.workspace)
    except Exception as err:
        emit_event("run.pipeline_failed", error_type=type(err).__name__, error=str(err))
        raise RuntimeError(f"benchmarkingjob runs failed, error: {err}.") from err


def _generate_parser():
    parser = argparse.ArgumentParser(description='AI Benchmarking Tool')
    parser.prog = "ianvs"

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-f",
                        "--benchmarking_config_file",
                        nargs="?",
                        type=str,
                        help="run a benchmarking job, "
                             "and the benchmarking config file must be yaml/yml file.")
    mode.add_argument("--agent", action="store_true", help="打开 Ianvs Agent 中文界面")
    parser.add_argument("--agent-config", help="Agent 配置文件路径")
    parser.add_argument("--agent-backend", choices=("local", "wsl"), help="Agent 执行端")
    parser.add_argument("--agent-wsl-distro", help="Agent WSL 发行版")
    parser.add_argument("--agent-backend-python", help="WSL Agent 控制解释器")

    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version=__version__,
                        help='show program version info and exit.')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser


if __name__ == '__main__':
    main()
