from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SmokeHttpError(RuntimeError):
    pass


JsonRequest = Callable[[str, str, Optional[dict[str, Any]], float], Any]


@dataclass(frozen=True)
class SmokeServiceExpectation:
    surface: str
    runtime_source_id: str
    memoryarena_base_commit: str
    product_pool_file_sha256: str
    product_pool_semantic_sha256: str
    catalog_sha256: str
    attributes_sha256: str
    lucene_manifest_sha256: str
    generator_seed: int
    split: str
    price_seed: int
    memory_prompt_mode: str
    minimum_task_count: int
    provider_mode: str = "fixed_window"
    start_orbit: int = 0
    first_valid_add_reward: float = 0.0
    first_valid_later_retrieve_reward: float = 0.0


class AgentMemorySmokeHttpClient:
    """Small client for a dedicated, resident native-WebShop smoke server."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 300.0,
        request_json: JsonRequest | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        self.timeout = float(timeout)
        self._request_json = request_json or _request_json
        self.request_trace: list[dict[str, Any]] = []

    def metadata(self) -> dict[str, Any]:
        payload = self._call("GET", "/metadata")
        if not isinstance(payload, dict):
            raise SmokeHttpError("/metadata did not return an object")
        return payload

    def open(self, data_idx: int) -> "AgentMemorySmokeSession":
        before = self.metadata()
        baseline_count = _active_environment_count(before)
        created = self._call("POST", "/create", {})
        if not isinstance(created, dict) or not isinstance(created.get("id"), int):
            raise SmokeHttpError("/create did not return an integer environment id")
        session = AgentMemorySmokeSession(
            client=self,
            env_id=created["id"],
            baseline_active_count=baseline_count,
        )
        try:
            reset = session.reset(data_idx)
            assert_clean_reset(reset)
        except BaseException:
            session.close(verify_count=False)
            raise
        return session

    def _call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        response = self._request_json(
            method,
            self.base_url + path,
            body,
            self.timeout,
        )
        self.request_trace.append(
            {
                "method": method,
                "path": path,
                "request": body,
                "response": response,
            }
        )
        return response


class AgentMemorySmokeSession:
    def __init__(
        self,
        *,
        client: AgentMemorySmokeHttpClient,
        env_id: int,
        baseline_active_count: int,
    ) -> None:
        self.client = client
        self.env_id = env_id
        self.baseline_active_count = baseline_active_count
        self.closed = False
        self.last_payload: dict[str, Any] | None = None

    def __enter__(self) -> "AgentMemorySmokeSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def reset(self, data_idx: int) -> dict[str, Any]:
        payload = self.client._call(
            "POST",
            "/reset",
            {"id": self.env_id, "data_idx": int(data_idx)},
        )
        self.last_payload = _require_payload(payload, endpoint="/reset")
        return self.last_payload

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        payload = self.client._call(
            "POST",
            "/step",
            {"id": self.env_id, "action": action},
        )
        self.last_payload = _require_payload(payload, endpoint="/step")
        observation = self.last_payload.get("observation")
        info = self.last_payload.get("info")
        if not isinstance(observation, str) or not isinstance(info, dict):
            raise SmokeHttpError("/step omitted observation or info")
        return (
            observation,
            float(self.last_payload.get("reward", 0.0)),
            bool(self.last_payload.get("done", False)),
            False,
            info,
        )

    def close(self, *, verify_count: bool = True) -> None:
        if self.closed:
            return
        try:
            result = self.client._call("POST", "/close", {"id": self.env_id})
            if result is not True:
                raise SmokeHttpError("/close did not return true")
        finally:
            self.closed = True
        if verify_count:
            observed = _active_environment_count(self.client.metadata())
            if observed != self.baseline_active_count:
                raise SmokeHttpError(
                    "smoke environment leaked after /close: "
                    f"before={self.baseline_active_count} after={observed}"
                )


def validate_smoke_service(
    metadata: Mapping[str, Any],
    expected: SmokeServiceExpectation,
) -> str:
    service = _mapping(metadata.get("service"), "service")
    provider = _mapping(metadata.get("provider"), "provider")
    runtime_inputs = _mapping(metadata.get("runtime_inputs"), "runtime_inputs")
    backend = _mapping(metadata.get("backend"), "backend")

    checks = {
        "service.role": (service.get("role"), "smoke"),
        "service.runtime_source_id": (
            service.get("runtime_source_id"),
            expected.runtime_source_id,
        ),
        "surface": (metadata.get("surface"), expected.surface),
        "provider.provider_mode": (
            provider.get("provider_mode"),
            expected.provider_mode,
        ),
        "provider.split": (provider.get("split"), expected.split),
        "provider.generator_base_seed": (
            provider.get("generator_base_seed"),
            expected.generator_seed,
        ),
        "provider.product_pool_sha256": (
            provider.get("product_pool_sha256"),
            expected.product_pool_semantic_sha256,
        ),
        "runtime_inputs.product_pool_file_sha256": (
            runtime_inputs.get("product_pool_file_sha256"),
            expected.product_pool_file_sha256,
        ),
        "runtime_inputs.catalog_sha256": (
            runtime_inputs.get("catalog_sha256"),
            expected.catalog_sha256,
        ),
        "runtime_inputs.attributes_sha256": (
            runtime_inputs.get("attributes_sha256"),
            expected.attributes_sha256,
        ),
        "runtime_inputs.lucene_manifest_sha256": (
            runtime_inputs.get("lucene_manifest_sha256"),
            expected.lucene_manifest_sha256,
        ),
        "backend.price_seed": (backend.get("price_seed"), expected.price_seed),
        "memory_prompt_mode": (
            metadata.get("memory_prompt_mode"),
            expected.memory_prompt_mode,
        ),
    }
    fixed_window = provider.get("fixed_window")
    if expected.provider_mode == "fixed_window":
        if not isinstance(fixed_window, Mapping):
            mismatches = ["provider.fixed_window: missing"]
        else:
            mismatches = []
            if fixed_window.get("start_orbit") != expected.start_orbit:
                mismatches.append(
                    "provider.fixed_window.start_orbit: "
                    f"observed={fixed_window.get('start_orbit')!r} "
                    f"expected={expected.start_orbit!r}"
                )
    else:
        mismatches = []
    reward_contract = _mapping(metadata.get("reward_contract"), "reward_contract")
    checks["reward_contract.first_valid_add_reward"] = (
        reward_contract.get("first_valid_add_reward"),
        expected.first_valid_add_reward,
    )
    checks["reward_contract.first_valid_later_session_retrieve_reward"] = (
        reward_contract.get("first_valid_later_session_retrieve_reward"),
        expected.first_valid_later_retrieve_reward,
    )
    upstream = _mapping(backend.get("upstream_provenance"), "upstream_provenance")
    observed_commit = (
        upstream.get("memoryarena_commit")
        or upstream.get("commit")
        or upstream.get("head_commit")
    )
    checks["backend.upstream_provenance.commit"] = (
        observed_commit,
        expected.memoryarena_base_commit,
    )

    mismatches.extend(
        [
        f"{name}: observed={observed!r} expected={wanted!r}"
        for name, (observed, wanted) in checks.items()
        if observed != wanted
        ]
    )
    task_count = provider.get("task_count")
    if not isinstance(task_count, int) or task_count < expected.minimum_task_count:
        mismatches.append(
            "provider.task_count: "
            f"observed={task_count!r} minimum={expected.minimum_task_count!r}"
        )
    fingerprint = service.get("fingerprint_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        mismatches.append(f"service.fingerprint_sha256: observed={fingerprint!r}")
    if mismatches:
        raise SmokeHttpError(
            "resident smoke service fingerprint inputs do not match:\n- "
            + "\n- ".join(mismatches)
        )
    return fingerprint


def assert_clean_reset(payload: Mapping[str, Any]) -> None:
    info = _mapping(payload.get("info"), "reset.info")
    memory_diff = _mapping(info.get("memory_state_diff"), "memory_state_diff")
    checks = {
        "reward": payload.get("reward") == 0.0,
        "done": payload.get("done") is False,
        "current_subtask_index": info.get("current_subtask_index") == 0,
        "ltm_inventory_count": info.get("ltm_inventory_count") == 0,
        "session_trace": info.get("session_trace") == [],
        "tool_ops": info.get("tool_ops") == [],
        "memory_state_diff.added": memory_diff.get("added") == [],
        "memory_state_diff.updated": memory_diff.get("updated") == [],
        "memory_state_diff.deleted": memory_diff.get("deleted") == [],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SmokeHttpError(
            "resident smoke reset retained episode state: " + ", ".join(failed)
        )


def require_correct_buy(info: Mapping[str, Any], *, session_index: int) -> None:
    tool_ops = info.get("tool_ops")
    if not isinstance(tool_ops, list):
        raise SmokeHttpError("BUY info omitted tool_ops")
    buy_ops = [item for item in tool_ops if isinstance(item, dict) and item.get("op") == "BUY"]
    if len(buy_ops) != 1:
        raise SmokeHttpError(f"expected one BUY receipt, observed {buy_ops!r}")
    receipt = buy_ops[0]
    expected = {
        "raw_action": "click[Buy Now]",
        "committed": True,
        "purchase_correct": True,
        "session_advanced": True,
        "session_index": session_index,
    }
    mismatches = {
        key: (receipt.get(key), value)
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise SmokeHttpError(f"BUY receipt mismatch: {mismatches!r}")
    if "actual_asin" in json.dumps(info, ensure_ascii=True, sort_keys=True):
        raise SmokeHttpError("private purchase identity leaked through HTTP info")


def _request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None,
    timeout: float,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise SmokeHttpError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise SmokeHttpError(f"cannot reach resident smoke service {url}: {exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SmokeHttpError(f"non-JSON response from {url}") from exc


def _require_payload(payload: Any, *, endpoint: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SmokeHttpError(f"{endpoint} did not return an object")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SmokeHttpError(f"resident smoke metadata omitted {name}")
    return value


def _active_environment_count(metadata: Mapping[str, Any]) -> int:
    count = metadata.get("active_environment_count")
    if not isinstance(count, int) or count < 0:
        raise SmokeHttpError("metadata omitted active_environment_count")
    return count
