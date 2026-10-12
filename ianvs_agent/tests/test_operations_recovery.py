# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest

from ianvs_agent.runtime.processes import runtime_environment


@unittest.skipUnless(os.environ.get('IANVS_OPERATIONS_EVIDENCE_ROOT'),
                     '需要已完成的真实环境运维实验目录')
class OperationsEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['IANVS_OPERATIONS_EVIDENCE_ROOT']).resolve()
        cls.report = json.loads((cls.root / 'trial.json').read_text(encoding='utf-8'))

    def test_original_records_and_model_artifacts_remain_unchanged(self):
        for field in ('protected_records', 'model_records'):
            self.assertTrue(self.report[field])
            for filename, expected in self.report[field].items():
                with Path(filename).open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                self.assertEqual(actual, expected, filename)

    def test_autonomous_results_and_postverification_preserve_original_task(self):
        self.assertEqual({case['name'] for case in self.report['cases']},
                         {'undeclared_dependency', 'installed_version_conflict'})
        for case in self.report['cases']:
            self.assertTrue(case['baseline_verification']['passed'])
            self.assertTrue(case['fault_confirmed'])
            self.assertEqual(case['passed'], case['autonomous_result']['passed'])
            self.assertTrue(case['model_installed_dependencies'])
            self.assertTrue(case['model_inspected_imports'])
            self.assertTrue(case['model_checked_example_declaration'])
            if case['name'] == 'undeclared_dependency':
                self.assertTrue(case['model_patched_declaration'])
            self.assertTrue(case['model_reran_original_task'])
            formal = json.loads((self.root / 'state/instances' / case['instance_id'] / 'runs' /
                                  case['repaired_run_id'] / 'verification.json').read_text(encoding='utf-8'))
            task = json.loads((self.root / 'state' / case['session_path'] / 'task.json').read_text(encoding='utf-8'))
            self.assertEqual(task['task_id'], case['task_id'])
            self.assertEqual(formal['acceptance']['repair_task_id'], case['task_id'])
            if case['passed']:
                self.assertEqual(case['status'], 'completed')
                self.assertTrue(formal['passed'])
                self.assertTrue(formal['repair_completed'])
            if case['name'] == 'undeclared_dependency':
                attempts = [item for item in self.report['attempts'] if item['case'] == case['name']]
                self.assertEqual([item['status'] for item in attempts], ['blocked', 'blocked'])
                self.assertTrue(all(not item['passed'] for item in attempts))
                self.assertEqual(attempts[1]['attempt_usage']['model_calls'], 0)
                self.assertEqual(attempts[0]['usage']['task_effective_input_tokens'],
                                 attempts[1]['usage']['task_effective_input_tokens'])
                self.assertTrue(case['independent_postverification'])
                for record in case['independent_postverification']:
                    self.assertTrue(record['passed'])
                    self.assertFalse(record['model_repair_score'])
                    self.assertEqual(record['model_calls'], 0)
                    self.assertTrue(record['session_records_unchanged'])
                self.assertTrue(formal['passed'])
                self.assertTrue(formal['repair_completed'])
            self.assertEqual(case['protected_source_changes'], [])
            self.assertEqual(case['unauthorized_executions'], [])

    def test_active_environments_have_real_packages_and_imports(self):
        for case in self.report['cases']:
            environment = case['final_environment']
            healthy = case['healthy_environment']
            self.assertEqual(environment['status'], 'ready')
            self.assertGreater(environment['generation'], healthy['generation'])
            self.assertNotEqual(environment['venv_root'], healthy['venv_root'])
            python = environment['python_executable']
            result = subprocess.run([python, '-m', 'pip', 'check'], cwd=case['code_root'],
                                    env=runtime_environment(python), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            imports = case['final_check']['imports']
            self.assertTrue(imports)
            self.assertTrue(all(item['passed'] for item in imports))
            expression = (
                "import json,sys,torch,transformers,accelerate,safetensors,sedna; "
                "print(json.dumps({'prefix':sys.prefix,'files':{k:globals()[k].__file__ "
                "for k in ('torch','transformers','accelerate','safetensors','sedna')}}))"
            )
            result = subprocess.run([python, '-c', expression], cwd=case['code_root'],
                                    env=runtime_environment(python), capture_output=True, text=True, check=True)
            probe = json.loads(result.stdout)
            venv = Path(environment['venv_root']).resolve()
            self.assertEqual(Path(probe['prefix']).resolve(), venv)
            self.assertTrue(all(Path(path).resolve().is_relative_to(venv) for path in probe['files'].values()))

    def test_actual_cancellation_recovery_and_cross_instance_isolation(self):
        self.assertTrue(all(check['passed'] for check in self.report['recovery_checks']))
        checks = {check['name']: check for check in self.report['recovery_checks']}
        for required in ('cancel_active_install', 'resume_cancelled_install', 'restore_session',
                         'failed_install_then_continue', 'cross_instance_isolation',
                         'interrupted_backend_recovery'):
            self.assertIn(required, checks)
            self.assertTrue(checks[required]['passed'], required)
            self.assertTrue(checks[required]['evidence_refs'], required)
        cancelled = checks['cancel_active_install']
        self.assertTrue(cancelled['actual_process_started'])
        self.assertTrue(cancelled['actual_pip_process_ended'])
        self.assertTrue(cancelled['owner_process_ended'])
        self.assertEqual(cancelled['actual_pip_command'][1:4], ['-m', 'pip', 'install'])
        self.assertTrue(cancelled['owned_processes_ended'])
        self.assertFalse(cancelled['candidate_activated'])
        restored = checks['restore_session']
        self.assertTrue(restored['same_task_id'])
        self.assertTrue(restored['usage_continued'])
        self.assertTrue(restored['original_acceptance_preserved'])
        for restored in (check for check in self.report['recovery_checks'] if check['name'] == 'restore_session'):
            if restored['case'] == 'installed_version_conflict':
                self.assertTrue(restored['all_attempt_task_ids_preserved'])
                self.assertTrue(restored['original_acceptance_across_attempts'])
                self.assertTrue(restored['cumulative_budgets_preserved'])

    def test_attempts_keep_real_usage_and_failed_process(self):
        self.assertTrue(self.report['attempts'])
        for attempt in self.report['attempts']:
            self.assertTrue(Path(attempt['prompt_ref']).is_file())
            self.assertTrue(Path(attempt['attempt_ref']).is_dir())
            self.assertTrue(attempt['notified_at'])
            self.assertTrue(attempt['stopped_at'])
            self.assertGreaterEqual(attempt['wall_seconds'], 0)
            usage = attempt['usage']
            self.assertLessEqual(usage['task_effective_input_tokens'], 500000)
            self.assertIn('missing_usage_calls', usage)
            self.assertIn('cache_read_tokens', usage)
            self.assertIn('cost', usage)
            if not attempt['passed']:
                self.assertIsNone(attempt['repair_success_seconds'])
        self.assertTrue(self.report['evidence_checks_passed'])
        summary = self.report['autonomous_summary']
        self.assertEqual(summary['passed_cases'], sum(bool(case['passed']) for case in self.report['cases']))
        self.assertEqual(summary['total_cases'], len(self.report['cases']))
        self.assertEqual(summary['passed'], all(case['passed'] for case in self.report['cases']))
        self.assertEqual(self.report['passed'], summary['passed'] and self.report['evidence_checks_passed'])


if __name__ == '__main__':
    unittest.main()
