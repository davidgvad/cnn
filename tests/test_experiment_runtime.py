"""Hardware choices must change scheduling without changing scientific settings."""
from __future__ import annotations

import argparse
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

    def test_auto_falls_back_to_cpu(self):
        runtime = subject.select_runtime(self.args(), gpu_count=0)
        self.assertEqual(runtime.device, "cpu")
        self.assertEqual(runtime.workers, ("cpu0",))
        self.assertEqual(runtime.cuda_tokens, {"cpu0": ""})

    def test_any_gpu_count_and_optional_concurrency_limit(self):
        for count in (1, 2, 3, 4, 8):
            with self.subTest(count=count):
                runtime = subject.select_runtime(self.args(), gpu_count=count)
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

    def test_invalid_and_conflicting_requests_fail(self):
        for values, count in [(("--workers", "0"), 2), (("--gpus", "0", "0"), 2),
                              (("--device", "gpu"), 0), (("--gpus", "2"), 2),
                              (("--workers", "3"), 2), (("--device", "cpu", "--gpus", "0"), 2)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                subject.select_runtime(self.args(*values), gpu_count=count)

    def test_cpu_worker_requires_no_visible_gpus(self):
        subject.validate_worker_devices([], allow_cpu=True)
        subject.validate_worker_devices([object()], allow_cpu=False)
        for devices, cpu in (([], False), ([object()], True), ([object(), object()], False)):
            with self.assertRaises(RuntimeError):
                subject.validate_worker_devices(devices, allow_cpu=cpu)


if __name__ == "__main__":
    unittest.main()
