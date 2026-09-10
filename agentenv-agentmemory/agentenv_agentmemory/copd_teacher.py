"""Optional, episode-fixed text teacher behind an ordinary sandbox shell CLI.

Only this host module reads authentication. The mounted CLI uses local FIFOs;
policy namespaces retain their existing no-network and privilege boundaries.
There are no reward, policy-sampling, or optimizer dependencies here.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
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


def _append(path: Path, event: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


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


def _ask_bounded(question: str, *, config: dict, timeout: float) -> tuple[str, dict]:
    """An absolute process deadline also bounds DNS and slow response bodies."""
    started = time.monotonic()
    environment = dict(os.environ)
    if config.get("proxy"):
        environment.update(HTTPS_PROXY=config["proxy"], https_proxy=config["proxy"])
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--host-ask"],
            input=json.dumps({"question": question, "timeout": timeout}),
            text=True, capture_output=True, timeout=timeout, check=True,
            env=environment,
        )
        reply, detail = json.loads(result.stdout)
        return reply, detail
    except subprocess.TimeoutExpired:
        status = "timeout"
    except (subprocess.CalledProcessError, ValueError):
        status = "transport_or_response_error"
    return "[teacher_error:" + status + "]", {
        "status": status, "upstream_attempts": 1, "usage": None,
        "cost": None, "cost_status": "provider_not_reported",
        "latency_seconds": time.monotonic() - started,
    }


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
                    readable, _, _ = select.select([request_fd], [], [], 0.05)
                    if not readable:
                        continue
                    chunk = os.read(request_fd, MAX_QUESTION_BYTES + 1)
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
                        reply, detail = _ask_bounded(question, config=config, timeout=min(config["timeout"], deadline-time.monotonic()-0.25))
                    event = {**receipt, **detail, "schema": "copd_teacher_call_v1",
                        "call_id": uuid.uuid4().hex, "command_id": command_id,
                        "command_sha256": digest(command), "timestamp_ns": time.time_ns(),
                        "question": question, "question_sha256": digest(question),
                        "reply": reply, "reply_sha256": digest(reply), "delivered": False}
                    output = (reply + "\n").encode()
                    fd = None
                    while not stop.is_set() and time.monotonic() < deadline + 0.25:
                        try:
                            fd = os.open(root / "reply", os.O_WRONLY | os.O_NONBLOCK)
                            break
                        except OSError:
                            stop.wait(0.01)
                    if fd is not None:
                        try:
                            sent = 0
                            while sent < len(output) and not stop.is_set():
                                _, writable, _ = select.select([], [fd], [], 0.05)
                                if writable:
                                    sent += os.write(fd, output[sent:])
                            event["delivered"] = sent == len(output)
                        except (BrokenPipeError, BlockingIOError):
                            pass
                        finally:
                            os.close(fd)
                    _append(config["ledger"] / (receipt["episode_key"] + ".jsonl"), event)
            except BaseException as exc:
                thread_errors.append(exc)

        thread = threading.Thread(target=serve, name="copd-command-broker")
        thread.start()
        try:
            yield str(root)
        finally:
            stop.set()
            thread.join(config["timeout"] + 1)
            os.close(request_fd)
            if thread.is_alive():
                raise RuntimeError("teacher broker failed to stop")
            if thread_errors:
                raise RuntimeError("teacher broker failed") from thread_errors[0]


if __name__ == "__main__":
    if sys.argv[1:] != ["--host-ask"]:
        raise SystemExit("host-only entrypoint")
    request = json.loads(sys.stdin.read(MAX_QUESTION_BYTES * 6 + 1024))
    print(json.dumps(_ask(request["question"], config=_config(), timeout=request["timeout"])))
