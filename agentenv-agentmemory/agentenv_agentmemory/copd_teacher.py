"""Optional, episode-fixed text teacher behind an ordinary sandbox shell CLI.

Only this host module reads authentication. The mounted CLI uses local FIFOs;
policy namespaces retain their existing no-network and privilege boundaries.
There are no reward, policy-sampling, or optimizer dependencies here.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

MAX_QUESTION_BYTES = 32 * 1024
MAX_REPLY_BYTES = 12 * 1024
MAX_PROVIDER_BODY_BYTES = 256 * 1024
LEDGER_LOCK_TIMEOUT_SECONDS = 0.75
BROKER_JOIN_TIMEOUT_SECONDS = 2.0
REPLY_OPEN_TIMEOUT_SECONDS = 1.0
UNAVAILABLE = "[teacher_unavailable]"
SYSTEM_PROMPT = (
    "You advise an agent that alone acts in an environment. You cannot use tools "
    "or execute actions. Reply with concise guidance based only on the agent's "
    "question and supplied evidence. State uncertainty and suggest checks when "
    "evidence is missing. Never claim to have observed files or results that "
    "the agent did not supply."
)
POLICY_NOTICE = """
You may consult a text advisor with an ordinary shell_command. Use the executable
`/run/copd/copd_ask 'your question and relevant evidence'`; or pipe your question
through stdin with `/run/copd/copd_ask --stdin`. Its stdout is the advisor's reply,
so normal shell redirection can save it in your workspace. Decide whether and
when help is useful and supply the context you want the advisor to see. The
advisor has no tools or hidden task answers and may be wrong; verify its advice
with task evidence. Availability is fixed for the episode. If it returns
[teacher_unavailable], continue solving independently. A consultation uses the
same shell action budget as any other command and does not change task reward.
Use the existing maximum shell timeout for a consultation. The advisor only sees
what you explicitly send; later calls do not automatically include past calls.
""".strip()



def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def enabled() -> bool:
    value = os.environ.get("COPD_ENABLED", "0")
    if value not in {"0", "1"}:
        raise ValueError("COPD_ENABLED must be 0 or 1")
    return value == "1"


def _config() -> dict:
    probability = float(os.environ["COPD_TEACHER_ON_PROB"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("COPD teacher probability must be in [0,1]")
    api_base = os.environ["COPD_TEACHER_API_BASE"].rstrip("/")
    if api_base != "https://modelservice.jdcloud.com/v1":
        raise ValueError("COPD V1 requires the approved modelservice HTTPS endpoint")
    model = os.environ.get("COPD_TEACHER_MODEL", "Kimi-K2.6")
    if model != "Kimi-K2.6":
        raise ValueError("COPD V1 teacher model drift")
    proxy = os.environ.get("COPD_TEACHER_HTTPS_PROXY", "")
    if proxy not in {"", "http://bamboo-proxy.jd.com:80"}:
        raise ValueError("COPD requires the approved existing company proxy")
    max_tokens = int(os.environ.get("COPD_TEACHER_MAX_TOKENS", "256"))
    timeout = float(os.environ.get("COPD_TEACHER_TIMEOUT", "15"))
    if not 1 <= max_tokens <= 4096 or not 0 < timeout <= 120:
        raise ValueError("invalid teacher output/timeout bounds")
    key_path = Path(os.environ["COPD_TEACHER_KEY_FILE"])
    key_stat = key_path.lstat()
    if not stat.S_ISREG(key_stat.st_mode) or key_stat.st_mode & 0o077:
        raise ValueError("teacher key must be a private regular file")
    cli = Path(os.environ["COPD_CLI_BINARY"])
    cli_stat = cli.lstat()
    if not stat.S_ISREG(cli_stat.st_mode) or cli_stat.st_mode & 0o022:
        raise ValueError("teacher CLI must be a protected regular file")
    if hashlib.sha256(cli.read_bytes()).hexdigest() != os.environ["COPD_CLI_SHA256"]:
        raise ValueError("teacher CLI hash drift")
    ledger = Path(os.environ["COPD_LEDGER_DIR"])
    ledger.mkdir(parents=True, exist_ok=True, mode=0o700)
    if ledger.is_symlink() or ledger.stat().st_mode & 0o077:
        raise ValueError("teacher ledger must be private")
    return dict(probability=probability, api_base=api_base, model=model,
                max_tokens=max_tokens, timeout=timeout, key_path=key_path,
                ledger=ledger, run_id=os.environ["COPD_RUN_ID"],
                seed=os.environ["COPD_AVAILABILITY_SEED"], cli=cli, proxy=proxy)


def _write_event(path: Path, event: dict, *, exclusive: bool = False) -> None:
    """Write one immutable JSONL record without an unbounded lock wait."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    if exclusive:
        flags |= os.O_EXCL
    else:
        flags |= os.O_APPEND
    fd = os.open(path, flags, 0o600)
    try:
        payload = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode()
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short ledger write")
            view = view[written:]
    finally:
        os.close(fd)


def _fallback_event_path(path: Path) -> Path:
    """Choose a unique sibling record when the shared ledger lock is busy."""
    for _ in range(8):
        candidate = path.with_name(path.stem + ".event-" + uuid.uuid4().hex + ".jsonl")
        try:
            # Reserve the name before returning it.  The caller writes through
            # the already-created descriptor so another process cannot collide.
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            return candidate
        except FileExistsError:
            continue
    raise OSError("unable to allocate fallback ledger record")


def _append(path: Path, event: dict, *, lock_timeout: float = LEDGER_LOCK_TIMEOUT_SECONDS) -> dict:
    """Append a record with a bounded flock and an auditable fallback.

    The previous implementation could block the broker forever on an NFS
    ``flock``.  A busy lock now produces a separate immutable JSONL record,
    preserving the event while allowing the command broker to shut down.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    record = dict(event)
    fd = None
    acquired = False
    lock_error = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                lock_error = type(exc).__name__
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                if time.monotonic() - started >= max(0.0, lock_timeout):
                    break
                time.sleep(0.01)
        wait_ms = round((time.monotonic() - started) * 1000, 3)
        if acquired:
            record.update(ledger_append_status="appended", ledger_lock_wait_ms=wait_ms)
            payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short ledger write")
                view = view[written:]
            return {"status": "appended", "path": str(path), "wait_ms": wait_ms}
    except (OSError, ValueError) as exc:
        lock_error = lock_error or type(exc).__name__
    finally:
        if fd is not None:
            if acquired:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)

    # The target record may be locked or temporarily unavailable.  Do not
    # silently drop it and do not make the policy's shell command fail.
    fallback = _fallback_event_path(path)
    wait_ms = round((time.monotonic() - started) * 1000, 3)
    record.update(ledger_append_status="lock_timeout_fallback", ledger_lock_wait_ms=wait_ms)
    if lock_error:
        record["ledger_lock_error"] = lock_error
    _write_event(fallback, record)
    return {"status": "lock_timeout_fallback", "path": str(fallback), "wait_ms": wait_ms}


def episode_receipt(workspace: Path | str) -> dict | None:
    if not enabled():
        return None
    config = _config()
    # The workspace root is created at reset and remains fixed until teardown.
    # Hash-derived randomization is independent of queries and stays fixed even
    # when the public runner runs in a new host subprocess on every action.
    episode_key = digest(config["run_id"] + "\0" + str(workspace))
    coin = int(digest(config["seed"] + "\0" + episode_key)[:16], 16) / 2**64
    return {"schema": "copd_episode_availability_v1", "run_id": config["run_id"],
            "episode_key": episode_key, "teacher_available": coin < config["probability"],
            "teacher_on_probability": config["probability"],
            "availability_seed": config["seed"], "availability_fixed_for_episode": True}


def record_episode(workspace: Path | str) -> dict | None:
    receipt = episode_receipt(workspace)
    if receipt is not None:
        _append(_config()["ledger"] / (receipt["episode_key"] + ".jsonl"), receipt)
    return receipt


def policy_notice() -> str:
    return "\n\n" + POLICY_NOTICE if enabled() else ""


def _terminate_provider(proc: subprocess.Popen, *, grace: float = 0.5) -> dict:
    """Terminate only the exact provider process group and reap it."""
    result = {"provider_pid": proc.pid, "provider_process_group": None,
              "provider_termination": "already_exited"}
    try:
        pgid = os.getpgid(proc.pid)
        result["provider_process_group"] = pgid
    except OSError:
        pgid = None
    if proc.poll() is not None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=0)
        return result
    try:
        # ``start_new_session=True`` below makes pgid == pid.  If a platform
        # ever declines that setup, fall back to terminating just the child.
        if pgid is not None and pgid == proc.pid:
            os.killpg(pgid, signal.SIGTERM)
            result["provider_termination"] = "sigterm_process_group"
        else:
            proc.terminate()
            result["provider_termination"] = "sigterm_process"
    except OSError:
        with contextlib.suppress(OSError):
            proc.terminate()
        result["provider_termination"] = "terminate_error"
    try:
        proc.wait(timeout=max(0.0, grace))
    except subprocess.TimeoutExpired:
        try:
            if pgid is not None and pgid == proc.pid:
                os.killpg(pgid, signal.SIGKILL)
                result["provider_termination"] = "sigkill_process_group"
            else:
                proc.kill()
                result["provider_termination"] = "sigkill_process"
        except OSError:
            result["provider_termination"] = "kill_error"
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            proc.wait(timeout=max(0.0, grace))
    result["provider_returncode"] = proc.returncode
    return result


def _ask_bounded(question: str, *, config: dict, timeout: float,
                 cancel_event: threading.Event | None = None) -> tuple[str, dict]:
    """Run the host provider with a cancellable, process-group-bounded deadline."""
    started = time.monotonic()
    environment = dict(os.environ)
    if config.get("proxy"):
        environment.update(HTTPS_PROXY=config["proxy"], https_proxy=config["proxy"])
    detail = {"status": "transport_or_response_error", "upstream_attempts": 1,
              "usage": None, "cost": None, "cost_status": "provider_not_reported"}
    proc = None
    output_file = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            detail.update(status="cancelled", upstream_attempts=0,
                          provider_cleanup_status="not_started")
            return "[teacher_error:cancelled]", {**detail, "latency_seconds": 0.0}
        # A regular temporary file prevents a forked descendant from holding a
        # stdout pipe open after the exact provider child has exited.
        output_file = tempfile.TemporaryFile(prefix="copd-provider-", dir="/tmp")
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--host-ask"],
            stdin=subprocess.PIPE, stdout=output_file, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True, env=environment,
        )
        detail.update(provider_pid=proc.pid, provider_process_group=proc.pid)
        payload = json.dumps({"question": question, "timeout": timeout}).encode()
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            detail.update(status="transport_or_response_error",
                          error_class=type(exc).__name__)
            _terminate_provider(proc)
        deadline = time.monotonic() + max(0.01, timeout)
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                detail.update(status="cancelled")
                detail.update(_terminate_provider(proc))
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail.update(status="timeout")
                detail.update(_terminate_provider(proc))
                break
            try:
                proc.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            # A defensive final reap; normal provider processes are already
            # gone after the bounded group termination above.
            detail.update(_terminate_provider(proc, grace=0.2))
        else:
            detail.setdefault("provider_returncode", proc.returncode)
            detail.setdefault("provider_cleanup_status", "reaped")
        output_file.seek(0)
        raw = output_file.read(MAX_PROVIDER_BODY_BYTES + 1)
        if len(raw) > MAX_PROVIDER_BODY_BYTES:
            detail.update(status="transport_or_response_error",
                          error_class="oversized_provider_body")
            return "[teacher_error:oversized_provider_body]", {
                **detail, "latency_seconds": time.monotonic() - started}
        if detail["status"] in {"timeout", "cancelled"}:
            return "[teacher_error:" + detail["status"] + "]", {
                **detail, "latency_seconds": time.monotonic() - started}
        if proc.returncode != 0:
            detail.update(status="transport_or_response_error",
                          error_class="provider_exit_" + str(proc.returncode))
            return "[teacher_error:transport_or_response_error]", {
                **detail, "latency_seconds": time.monotonic() - started}
        try:
            reply, provider_detail = json.loads(raw.decode())
            if not isinstance(reply, str) or not isinstance(provider_detail, dict):
                raise ValueError("invalid provider response")
            detail.update(provider_detail)
            detail.setdefault("status", "ok")
            return reply, {**detail, "latency_seconds": time.monotonic() - started}
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            detail.update(status="transport_or_response_error", error_class=type(exc).__name__)
    except (OSError, ValueError, TypeError) as exc:
        detail.update(status="transport_or_response_error", error_class=type(exc).__name__)
        if proc is not None and proc.poll() is None:
            detail.update(_terminate_provider(proc))
    finally:
        if proc is not None:
            if proc.poll() is None:
                detail.update(_terminate_provider(proc, grace=0.2))
            with contextlib.suppress(OSError, ValueError):
                if proc.stdin is not None:
                    proc.stdin.close()
        if output_file is not None:
            with contextlib.suppress(OSError):
                output_file.close()
    return "[teacher_error:" + detail["status"] + "]", {
        **detail, "latency_seconds": time.monotonic() - started}


def _ask(question: str, *, config: dict, timeout: float) -> tuple[str, dict]:
    started = time.monotonic()
    telemetry = {"teacher_model": config["model"], "upstream_attempts": 1,
                 "usage": None, "cost": None, "cost_status": "provider_not_reported"}
    payload = {"model": config["model"], "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}], "temperature": 0,
        "max_tokens": config["max_tokens"], "thinking": {"type": "disabled"}}
    key = config["key_path"].read_text().strip()
    if not key:
        raise ValueError("empty teacher key")
    request = urllib.request.Request(config["api_base"] + "/chat/completions",
        data=json.dumps(payload).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + key,
            "User-Agent": "copd-v1/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(256 * 1024 + 1)
        if len(raw) > 256 * 1024:
            raise ValueError("oversized provider body")
        body = json.loads(raw)
        message = body["choices"][0]["message"]
        refusal = message.get("refusal")
        content = message.get("content")
        telemetry["status"] = "refusal" if refusal else "ok"
        if refusal:
            content = "[teacher_refusal] " + str(refusal)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty teacher reply")
        reply = content.strip().encode()[:MAX_REPLY_BYTES].decode("utf-8", errors="ignore")
        telemetry["reply_truncated"] = len(content.strip().encode()) > MAX_REPLY_BYTES
        usage = body.get("usage")
        if isinstance(usage, dict):
            telemetry["usage"] = {k: usage[k] for k in (
                "prompt_tokens", "completion_tokens", "total_tokens",
                "prompt_tokens_details", "completion_tokens_details") if k in usage}
        telemetry["provider_request_id"] = str(body.get("id", ""))[:200]
        telemetry["finish_reason"] = body["choices"][0].get("finish_reason")
    except urllib.error.HTTPError as exc:
        reply = "[teacher_error:http_" + str(exc.code) + "]"
        telemetry["status"] = "http_error"
        telemetry["http_status"] = exc.code
    except (TimeoutError, socket.timeout):
        reply = "[teacher_error:timeout]"
        telemetry["status"] = "timeout"
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        reply = "[teacher_error:" + type(exc).__name__ + "]"
        telemetry["status"] = "transport_or_response_error"
        telemetry["error_class"] = type(exc).__name__
    telemetry["latency_seconds"] = time.monotonic() - started
    return reply, telemetry


@contextlib.contextmanager
def command_mount(workspace, *, model_uid, command, timeout_ms, permitted=True):
    """Mountable per-command FIFO directory; broker never executes policy code."""
    if not permitted or not enabled():
        yield ""
        return
    config = _config()
    receipt = episode_receipt(workspace)
    command_id = uuid.uuid4().hex
    stop = threading.Event()
    deadline = time.monotonic() + timeout_ms / 1000
    thread_errors = []
    broker_state = {"provider_pid": None, "provider_process_group": None,
                    "provider_termination": None}
    with tempfile.TemporaryDirectory(prefix="copd-cli-", dir="/tmp") as raw:
        root = Path(raw)
        root.chmod(0o755)
        for name in ("request", "reply"):
            os.mkfifo(root / name, 0o600)
            if os.geteuid() == 0:
                os.chown(root / name, model_uid, model_uid)
        (root / "lock").write_text("")
        (root / "lock").chmod(0o444)
        shutil.copyfile(config["cli"], root / "copd_ask")
        (root / "copd_ask").chmod(0o555)
        request_fd = os.open(root / "request", os.O_RDWR | os.O_NONBLOCK)

        def serve():
            pending = bytearray()
            oversized = False
            try:
                while not stop.is_set():
                    try:
                        readable, _, _ = select.select([request_fd], [], [], 0.05)
                    except (OSError, ValueError) as exc:
                        if stop.is_set():
                            break
                        raise exc
                    if not readable:
                        continue
                    try:
                        chunk = os.read(request_fd, MAX_QUESTION_BYTES + 1)
                    except OSError as exc:
                        if stop.is_set() and exc.errno in (errno.EBADF, errno.EINTR):
                            break
                        raise
                    if not chunk:
                        # The broker keeps one O_RDWR descriptor open, but a
                        # peer can still transiently produce EOF while a mount
                        # is being torn down.
                        continue
                    if oversized:
                        if b"\0" not in chunk:
                            continue
                        chunk = chunk[chunk.index(b"\0"):]
                    pending.extend(chunk)
                    if b"\0" not in pending:
                        if len(pending) > MAX_QUESTION_BYTES:
                            oversized = True
                            pending.clear()
                        continue
                    raw_question, _, tail = pending.partition(b"\0")
                    pending[:] = tail
                    try:
                        question = raw_question.decode("utf-8")
                    except UnicodeDecodeError:
                        question = ""
                    invalid = oversized or len(raw_question) > MAX_QUESTION_BYTES or not question.strip()
                    oversized = False
                    if invalid:
                        reply, detail = "[teacher_error:invalid_question]", {"status": "invalid_question", "upstream_attempts": 0}
                    elif not receipt["teacher_available"]:
                        reply, detail = UNAVAILABLE, {"status": "unavailable", "upstream_attempts": 0, "usage": {"total_tokens": 0}, "cost": 0}
                    elif deadline - time.monotonic() <= 0.5:
                        reply, detail = "[teacher_error:timeout]", {"status": "timeout_before_request", "upstream_attempts": 0}
                    else:
                        reply, detail = _ask_bounded(
                            question, config=config,
                            timeout=min(config["timeout"], deadline-time.monotonic()-0.25),
                            cancel_event=stop,
                        )
                        for key in ("provider_pid", "provider_process_group", "provider_termination"):
                            if key in detail:
                                broker_state[key] = detail[key]
                    event = {**receipt, **detail, "schema": "copd_teacher_call_v1",
                        "call_id": uuid.uuid4().hex, "command_id": command_id,
                        "command_sha256": digest(command), "timestamp_ns": time.time_ns(),
                        "question": question, "question_sha256": digest(question),
                        "reply": reply, "reply_sha256": digest(reply), "delivered": False,
                        "delivery_status": "not_attempted"}
                    output = (reply + "\n").encode()
                    fd = None
                    delivery_deadline = min(
                        deadline + 0.25,
                        time.monotonic() + REPLY_OPEN_TIMEOUT_SECONDS,
                    )
                    while not stop.is_set() and time.monotonic() < delivery_deadline:
                        try:
                            fd = os.open(root / "reply", os.O_WRONLY | os.O_NONBLOCK)
                            break
                        except OSError as exc:
                            if exc.errno not in (errno.ENXIO, errno.ENOENT, errno.EINTR):
                                event["delivery_status"] = "reply_open_error_" + type(exc).__name__
                                break
                            stop.wait(0.01)
                    if fd is not None:
                        try:
                            sent = 0
                            while sent < len(output) and not stop.is_set() and time.monotonic() < delivery_deadline:
                                _, writable, _ = select.select([], [fd], [], 0.05)
                                if writable:
                                    try:
                                        sent += os.write(fd, output[sent:])
                                    except (BrokenPipeError, BlockingIOError):
                                        event["delivery_status"] = "client_gone"
                                        break
                            event["delivered"] = sent == len(output)
                            if event["delivered"]:
                                event["delivery_status"] = "delivered"
                            elif event["delivery_status"] == "not_attempted":
                                event["delivery_status"] = "reply_write_timeout"
                        except (BrokenPipeError, BlockingIOError):
                            event["delivery_status"] = "client_gone"
                        finally:
                            with contextlib.suppress(OSError):
                                os.close(fd)
                    elif event["delivery_status"] == "not_attempted":
                        event["delivery_status"] = "client_gone_or_reply_not_open"
                    event["broker_stop_seen"] = stop.is_set()
                    _append(config["ledger"] / (receipt["episode_key"] + ".jsonl"), event)
            except BaseException as exc:
                thread_errors.append(exc)

        thread = threading.Thread(target=serve, name="copd-command-broker", daemon=True)
        thread.start()
        try:
            yield str(root)
        finally:
            stop.set()
            # _ask_bounded polls this event while the provider process is
            # alive, so normal teardown does not wait for the full API timeout.
            thread.join(BROKER_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                # Closing the broker's private descriptor is a second, exact
                # wake-up path for a select() interrupted during teardown.
                with contextlib.suppress(OSError):
                    os.close(request_fd)
                thread.join(1.0)
            else:
                with contextlib.suppress(OSError):
                    os.close(request_fd)
            cleanup_errors = []
            if thread.is_alive():
                cleanup_errors.append("broker_thread_alive")
            if thread_errors:
                cleanup_errors.extend(type(exc).__name__ for exc in thread_errors)
            cleanup_event = {
                **receipt,
                "schema": "copd_teacher_cleanup_v1",
                "command_id": command_id,
                "timestamp_ns": time.time_ns(),
                "cleanup_status": "clean" if not cleanup_errors else "degraded",
                "cleanup_errors": cleanup_errors,
                "broker_thread_alive": thread.is_alive(),
                "provider": broker_state,
                "fifo_root": str(root),
                "fifo_paths": [str(root / name) for name in ("request", "reply")],
            }
            # Cleanup is evidence, never a reason to turn a successfully
            # executed environment step into HTTP 500.  _append itself has a
            # bounded lock and a fallback record.
            # Keep lifecycle records in a child directory so legacy readers
            # that glob the episode's call JSONL continue to see the call as
            # their final record.  The cleanup directory is still under the
            # same private, recursively auditable ledger root.
            cleanup_dir = config["ledger"] / "cleanup"
            with contextlib.suppress(BaseException):
                _append(cleanup_dir / (receipt["episode_key"] + "." + command_id + ".jsonl"), cleanup_event)


if __name__ == "__main__":
    if sys.argv[1:] != ["--host-ask"]:
        raise SystemExit("host-only entrypoint")
    request = json.loads(sys.stdin.read(MAX_QUESTION_BYTES * 6 + 1024))
    print(json.dumps(_ask(request["question"], config=_config(), timeout=request["timeout"])))
