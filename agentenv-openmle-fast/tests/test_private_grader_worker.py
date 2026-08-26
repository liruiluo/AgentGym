from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from agentenv_openmle_fast import _private_grader_worker as worker


class NativeEvaluatorClassificationTest(unittest.TestCase):
    def _write(self, root: Path, name: str, payload: bytes) -> Path:
        path = root / name
        path.write_bytes(payload)
        return path

    def _grade_metric(self, metric: bytes):
        class FakeFrame:
            def copy(self):
                return self

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metric_path = self._write(root, "metric.py", metric)
            answer = self._write(root, "answer.csv", b"y\n1\n")
            submission = self._write(root, "submission.csv", b"y\n1\n")
            fake_pandas = types.SimpleNamespace(read_csv=lambda _path: FakeFrame())
            with mock.patch.dict(sys.modules, {"pandas": fake_pandas}):
                return worker._grade(metric_path, answer, submission, True, ())

    def test_admitted_spearman_domain_fault_is_invalid(self) -> None:
        metric = b'''\
class NativeMetric:
    higher_is_better = True
    def validate_submission(self, prediction, truth): return True
    def evaluate(self, y_true, y_pred):
        raise Exception("Spearman correlation is undefined for the provided data.")
'''
        result = self._grade_metric(metric)
        self.assertEqual(result["classification"], "invalid_submission")
        self.assertIsNone(result["native_score"])

    def test_only_exact_admitted_exception_is_invalid(self) -> None:
        variants = (
            b'raise Exception("Spearman correlation is undefined for the provided data!")',
            b'raise RuntimeError("Spearman correlation is undefined for the provided data.")',
            b'raise Exception("unrecognized evaluator fault")',
        )
        for statement in variants:
            metric = (
                b"class NativeMetric:\n"
                b"    higher_is_better = True\n"
                b"    def validate_submission(self, prediction, truth): return True\n"
                b"    def evaluate(self, y_true, y_pred): " + statement + b"\n"
            )
            with self.subTest(statement=statement):
                with self.assertRaises((Exception, RuntimeError)):
                    self._grade_metric(metric)

    def test_nonfinite_scores_are_invalid_but_score_contract_drift_escapes(self) -> None:
        for expression in (b"float('nan')", b"float('inf')", b"float('-inf')"):
            metric = (
                b"class NativeMetric:\n"
                b"    higher_is_better = True\n"
                b"    def validate_submission(self, prediction, truth): return True\n"
                b"    def evaluate(self, y_true, y_pred): return " + expression + b"\n"
            )
            with self.subTest(expression=expression):
                result = self._grade_metric(metric)
                self.assertEqual(result["classification"], "invalid_submission")
        for expression in (b"True", b"'0.5'"):
            metric = (
                b"class NativeMetric:\n"
                b"    higher_is_better = True\n"
                b"    def validate_submission(self, prediction, truth): return True\n"
                b"    def evaluate(self, y_true, y_pred): return " + expression + b"\n"
            )
            with self.subTest(expression=expression):
                with self.assertRaises(TypeError):
                    self._grade_metric(metric)

    def test_evaluator_runtime_and_control_faults_escape_to_infrastructure_boundary(self) -> None:
        for fault_name in (
            b"RuntimeError",
            b"MemoryError",
            b"SystemExit",
            b"KeyboardInterrupt",
            b"GeneratorExit",
        ):
            metric = (
                b"class NativeMetric:\n"
                b"    higher_is_better = True\n"
                b"    def validate_submission(self, prediction, truth): return True\n"
                b"    def evaluate(self, y_true, y_pred): raise "
                + fault_name
                + b"('probe')\n"
            )
            fault_type = getattr(__builtins__, fault_name.decode(), None)
            if fault_type is None:
                import builtins

                fault_type = getattr(builtins, fault_name.decode())
            with self.subTest(fault=fault_name):
                with self.assertRaises(fault_type):
                    self._grade_metric(metric)

    def test_metric_load_failure_remains_an_infrastructure_exception(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metric_path = self._write(root, "metric.py", b"this is not valid python !")
            answer = self._write(root, "answer.csv", b"score\n1\n")
            submission = self._write(root, "submission.csv", b"score\n1\n")
            fake_pandas = types.SimpleNamespace(read_csv=lambda _path: object())
            with mock.patch.dict(sys.modules, {"pandas": fake_pandas}):
                with self.assertRaises(SyntaxError):
                    worker._grade(metric_path, answer, submission, True, ())


if __name__ == "__main__":
    unittest.main()
