"""GPU allocation changes scheduling without changing scientific settings"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment_runtime as subject


class RuntimeTests(unittest.TestCase):
    def args(self, *values):
        parser = argparse.ArgumentParser()
        subject.add_arguments(parser)
        parser.add_argument("--dry-run", action="store_true")
        return parser.parse_args(values)

    def test_no_visible_gpu_rejects_real_execution(self):
        for values in ((), ("--device", "gpu"), ("--gpus", "0")):
            with self.subTest(values=values), self.assertRaises(ValueError):
                subject.select_runtime(self.args(*values), gpu_count=0)

    def test_cpu_auto_and_allow_cpu_arguments_are_rejected(self):
        self.assertEqual(self.args().device, "gpu")
        for values in (("--device", "cpu"), ("--device", "auto"), ("--allow-cpu",)):
            with self.subTest(values=values), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*values)

    def test_any_gpu_count_and_optional_concurrency_limit(self):
        for count in (1, 2, 3, 4, 8):
            with self.subTest(count=count):
                runtime = subject.select_runtime(self.args(), gpu_count=count)
                self.assertEqual(runtime.device, "gpu")
                self.assertEqual(len(runtime.workers), count)
        runtime = subject.select_runtime(self.args("--workers", "2"), gpu_count=8)
        self.assertEqual(runtime.workers, ("0", "1"))

    def test_gpu_ids_resolve_within_scheduler_allocation(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}):
            runtime = subject.select_runtime(self.args("--gpus", "1", "0"), gpu_count=2)
            self.assertEqual(runtime.cuda_tokens, {"1": "GPU-b", "0": "GPU-a"})

    def test_explicit_dry_gpu_plan_does_not_import_tensorflow(self):
        with patch.object(subject, "visible_gpu_count", side_effect=AssertionError("must not discover")):
            runtime = subject.select_runtime(self.args("--device", "gpu", "--gpus", "0", "1", "--dry-run"))
        self.assertFalse(runtime.visibility_checked)
        self.assertEqual(runtime.workers, ("0", "1"))

    def test_implicit_dry_plan_still_requires_gpu_discovery(self):
        with patch.object(subject, "visible_gpu_count", return_value=0) as discover:
            with self.assertRaises(ValueError):
                subject.select_runtime(self.args("--dry-run"))
        discover.assert_called_once_with()

    def test_controller_rejects_unavailable_gpu(self):
        parser = argparse.ArgumentParser()
        subject.add_arguments(parser)
        with patch.object(subject, "visible_gpu_count", return_value=0):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                subject.configure_controller(parser.parse_args([]), parser)

    def test_invalid_and_conflicting_requests_fail(self):
        for values, count in [(("--workers", "0"), 2), (("--gpus", "0", "0"), 2),
                              (("--device", "gpu"), 0), (("--gpus", "2"), 2),
                              (("--workers", "3"), 2), (("--gpus", "0", "--workers", "2"), 2)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                subject.select_runtime(self.args(*values), gpu_count=count)

    def test_training_worker_requires_exactly_one_visible_gpu(self):
        subject.validate_worker_devices([object()])
        for devices in ([], [object(), object()]):
            with self.assertRaises(RuntimeError):
                subject.validate_worker_devices(devices)


if __name__ == "__main__":
    unittest.main()
