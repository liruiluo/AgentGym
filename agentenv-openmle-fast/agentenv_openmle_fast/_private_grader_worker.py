from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import math
import numbers
import os
import resource
import sys
import uuid
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "openmle_fast_private_worker_request_v1"
RESULT_SCHEMA = "openmle_fast_private_worker_result_v1"
_SAFE_EXCEPTION = Exception
_SAFE_TYPE = type
_ADMITTED_EVALUATOR_DOMAIN_ERROR_ARGS = (
    "Spearman correlation is undefined for the provided data.",
)


def main() -> int:
    try:
        request = json.loads(sys.stdin.buffer.read(256 * 1024))
        metric, answer, submission, direction, forms = _validate_request(request)
        _apply_limits(request["resource_limits"])
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = _grade(metric, answer, submission, direction, forms)
    except BaseException:  # noqa: BLE001 - native metric faults stay in worker
        result = _result("infrastructure_fault", None, False)
    os.write(
        1,
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return 0


def _validate_request(
    value: Any,
) -> tuple[Path, Path, Path, bool, tuple[str, ...]]:
    required = {
        "schema",
        "metric_path",
        "answer_path",
        "submission_path",
        "higher_is_better",
        "validator_success_forms",
        "resource_limits",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("invalid request")
    if value["schema"] != REQUEST_SCHEMA:
        raise ValueError("invalid request schema")
    direction = value["higher_is_better"]
    forms = value["validator_success_forms"]
    if (
        type(direction) is not bool
        or not isinstance(forms, list)
        or any(not isinstance(item, str) for item in forms)
    ):
        raise ValueError("invalid grading contract")
    paths = tuple(
        Path(value[name]).resolve()
        for name in (
            "metric_path",
            "answer_path",
            "submission_path",
        )
    )
    root = paths[0].parent
    if any(
        path.parent != root or path.is_symlink() or not path.is_file() for path in paths
    ):
        raise ValueError("worker inputs are not selected regular files")
    return paths[0], paths[1], paths[2], direction, tuple(forms)


def _apply_limits(value: Any) -> None:
    required = {"cpu_vcpus", "memory_bytes", "max_processes", "wall_ms", "input_bytes"}
    if (
        not isinstance(value, dict)
        or set(value) != required
        or any(type(item) is not int or item <= 0 for item in value.values())
    ):
        raise ValueError("invalid worker limits")
    cpu_seconds = max(1, math.ceil(value["wall_ms"] / 1000.0))
    _set_limit(resource.RLIMIT_CPU, cpu_seconds)
    _set_limit(resource.RLIMIT_CORE, 0)
    _set_limit(resource.RLIMIT_FSIZE, value["input_bytes"])
    _set_limit(resource.RLIMIT_NOFILE, 64)
    if sys.platform != "darwin":
        _set_limit(resource.RLIMIT_AS, value["memory_bytes"])


def _set_limit(kind: int, maximum: int) -> None:
    _soft, hard = resource.getrlimit(kind)
    target = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
    resource.setrlimit(kind, (target, target))


def _grade(
    metric_path: Path,
    answer_path: Path,
    submission_path: Path,
    expected_direction: bool,
    success_forms: tuple[str, ...],
) -> dict[str, Any]:
    import pandas as pd

    try:
        prediction = pd.read_csv(submission_path)
    except BaseException:  # noqa: BLE001 - malformed policy CSV is invalid
        return _result("invalid_submission", None, expected_direction)
    truth = pd.read_csv(answer_path)
    instance, module_name = _load_native_scorer(metric_path)
    try:
        direction = getattr(instance, "higher_is_better", None)
        if type(direction) is not bool or direction != expected_direction:
            raise ValueError("direction drift")
        try:
            validation = instance.validate_submission(prediction.copy(), truth.copy())
        except BaseException:  # noqa: BLE001 - validator exceptions are invalid
            return _result("invalid_submission", None, expected_direction)
        if not _canonicalize_validation(validation, success_forms):
            return _result("invalid_submission", None, expected_direction)
        # Sealed admission proves that this metric/runtime pair grades a known
        # good submission. Only explicitly admitted evaluator-domain outcomes
        # are policy-invalid; unknown scorer/runtime/control faults must reach
        # the worker's infrastructure-fault boundary.
        try:
            score_value = instance.evaluate(y_true=truth.copy(), y_pred=prediction.copy())
        except BaseException as error:  # noqa: BLE001 - classify exact native outcomes
            if (
                _SAFE_TYPE(error) is _SAFE_EXCEPTION
                and error.args == _ADMITTED_EVALUATOR_DOMAIN_ERROR_ARGS
            ):
                return _result("invalid_submission", None, expected_direction)
            raise
        if isinstance(score_value, bool) or not isinstance(score_value, numbers.Real):
            raise TypeError("invalid native score")
        score = float(score_value)
        if not math.isfinite(score):
            return _result("invalid_submission", None, expected_direction)
        return _result("graded", score, expected_direction)
    finally:
        sys.modules.pop(module_name, None)


def _load_native_scorer(path: Path) -> tuple[Any, str]:
    module_name = f"_openmle_fast_native_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load scorer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    eligible = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and value.__module__ == module_name
        and callable(getattr(value, "validate_submission", None))
        and callable(getattr(value, "evaluate", None))
    ]
    if len(eligible) != 1:
        sys.modules.pop(module_name, None)
        raise ValueError("scorer class contract drift")
    return eligible[0](), module_name


def _canonicalize_validation(value: Any, success_forms: tuple[str, ...]) -> bool:
    if type(value) is bool:
        return value
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and type(value[0]) is bool
        and isinstance(value[1], str)
    ):
        return value[0]
    if isinstance(value, str):
        if not success_forms:
            raise ValueError("unadmitted string convention")
        return value in success_forms
    raise ValueError("validator convention drift")


def _result(
    classification: str,
    score: float | None,
    direction: bool,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "classification": classification,
        "native_score": score,
        "higher_is_better": direction,
    }


if __name__ == "__main__":
    raise SystemExit(main())
