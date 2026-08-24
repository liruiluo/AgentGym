from __future__ import annotations

import json
import logging
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from .environment import EPISODE_SCHEMA, VerifiedEpisodeManager


MAX_REQUEST_BYTES = 1024 * 1024
_BEARER_RE = re.compile(r"\ABearer ([A-Za-z0-9_-]{16,256})\Z")


class VerifiedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: VerifiedEpisodeManager,
    ) -> None:
        self.manager = manager
        super().__init__(server_address, VerifiedRequestHandler)


class VerifiedRequestHandler(BaseHTTPRequestHandler):
    server: VerifiedHTTPServer

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/":
                self.require_query(parsed.query, set())
                self.send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if parsed.path == "/metadata":
                self.require_query(parsed.query, set())
                self.send_json(HTTPStatus.OK, self.server.manager.metadata())
                return
            if parsed.path == "/observation":
                slot_id = self.authorized_query_slot(parsed.query)
                self.send_json(
                    HTTPStatus.OK,
                    {"observation": self.server.manager.observation(slot_id)},
                )
                return
            if parsed.path == "/prediction":
                slot_id = self.authorized_query_slot(parsed.query)
                self.send_json(
                    HTTPStatus.OK, self.server.manager.prediction(slot_id)
                )
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.handle_runtime_error(exc)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        try:
            self.require_query(parsed.query, set())
            if parsed.path == "/create":
                body = self.read_body({"arm", "run_id"})
                slot_id = self.server.manager.create(
                    arm=require_text(body, "arm"),
                    run_id=require_text(body, "run_id"),
                    run_capability=self.bearer_capability(),
                )
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "id": slot_id,
                        "capability": self.server.manager.capability(slot_id),
                        "observation": (
                            "Verified environment created; reset with data_idx."
                        ),
                        "reward": 0.0,
                        "done": False,
                        "info": {"schema": EPISODE_SCHEMA},
                    },
                )
                return
            if parsed.path == "/reset":
                body = self.read_body({"id", "data_idx"})
                slot_id = self.authorize_body(body)
                result = self.server.manager.reset(
                    slot_id,
                    require_integer(body, "data_idx"),
                )
                self.send_json(HTTPStatus.OK, result.as_dict())
                return
            if parsed.path == "/step":
                body = self.read_body({"id", "action"})
                slot_id = self.authorize_body(body)
                result = self.server.manager.step(
                    slot_id,
                    require_text(body, "action", allow_empty=True),
                )
                self.send_json(HTTPStatus.OK, result.as_dict())
                return
            if parsed.path == "/horizon":
                body = self.read_body({"id"})
                slot_id = self.authorize_body(body)
                result = self.server.manager.finalize_horizon(slot_id)
                self.send_json(HTTPStatus.OK, result.as_dict())
                return
            if parsed.path == "/no-submission":
                body = self.read_body({"id"})
                slot_id = self.authorize_body(body)
                result = self.server.manager.record_no_submission(slot_id)
                self.send_json(HTTPStatus.OK, result)
                return
            if parsed.path == "/predictions/assemble":
                body = self.read_body(
                    {"id", "arm", "run_id"}
                )
                arm = require_text(body, "arm")
                run_id = require_text(body, "run_id")
                self.authorize_body(body, arm=arm, run_id=run_id)
                result = self.server.manager.assemble_predictions(
                    arm=arm,
                    run_id=run_id,
                )
                self.send_json(HTTPStatus.OK, result)
                return
            if parsed.path == "/close":
                body = self.read_body({"id"})
                slot_id = self.authorize_body(body)
                result = self.server.manager.close(slot_id)
                self.send_json(HTTPStatus.OK, result)
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self.handle_runtime_error(exc)

    def read_body(self, expected_fields: set[str]) -> Mapping[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("request body is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be UTF-8 JSON") from exc
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise ValueError("request body has unexpected or missing fields")
        return value

    def authorized_query_slot(self, query: str) -> int:
        values = self.require_query(query, {"id"})
        raw = values["id"][0]
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("query id must be an integer") from exc
        self.server.manager.authorize(value, self.bearer_capability())
        return value

    def authorize_body(
        self,
        body: Mapping[str, Any],
        *,
        arm: str | None = None,
        run_id: str | None = None,
    ) -> int:
        slot_id = require_integer(body, "id")
        self.server.manager.authorize(
            slot_id,
            self.bearer_capability(),
            arm=arm,
            run_id=run_id,
        )
        return slot_id

    def bearer_capability(self) -> str:
        values = self.headers.get_all("Authorization", failobj=[])
        if len(values) != 1:
            raise PermissionError("bearer authorization failed")
        matched = _BEARER_RE.fullmatch(values[0])
        if matched is None:
            raise PermissionError("bearer authorization failed")
        return matched.group(1)

    @staticmethod
    def require_query(query: str, expected_fields: set[str]) -> dict[str, list[str]]:
        values = (
            parse_qs(query, keep_blank_values=True, strict_parsing=True)
            if query
            else {}
        )
        if set(values) != expected_fields or any(
            len(items) != 1 for items in values.values()
        ):
            raise ValueError("query has unexpected or missing fields")
        return values

    def handle_runtime_error(self, exc: Exception) -> None:
        if isinstance(exc, PermissionError):
            status = HTTPStatus.FORBIDDEN
        elif isinstance(exc, (ValueError, TypeError)):
            status = HTTPStatus.BAD_REQUEST
        elif isinstance(exc, KeyError):
            status = HTTPStatus.NOT_FOUND
        elif isinstance(exc, RuntimeError):
            status = HTTPStatus.CONFLICT
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        logging.exception("Verified HTTP request failed")
        self.send_error_json(status, "request failed closed; inspect server logs")

    def send_error_json(self, status: HTTPStatus, detail: str) -> None:
        self.send_json(status, {"detail": detail})

    def send_json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        payload = json.dumps(
            dict(value), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return None


def create_http_server(
    manager: VerifiedEpisodeManager,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> VerifiedHTTPServer:
    return VerifiedHTTPServer((host, port), manager)


def require_integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{key} must be an integer")
    return item


def require_text(
    value: Mapping[str, Any], key: str, *, allow_empty: bool = False
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item):
        raise TypeError(f"{key} must be text")
    return item
