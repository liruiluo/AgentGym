import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "copd_teacher", Path(__file__).parents[1] / "agentenv_agentmemory/copd_teacher.py"
)
teacher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher)


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        key = self.root / "key"
        key.write_text("fixture-not-a-real-secret")
        key.chmod(0o600)
        cli = self.root / "cli"
        cli.write_bytes(b"fixture")
        cli.chmod(0o555)
        self.env = patch.dict(os.environ, {
            "COPD_ENABLED": "1", "COPD_TEACHER_ON_PROB": "0.5",
            "COPD_TEACHER_API_BASE": "https://modelservice.jdcloud.com/v1",
            "COPD_TEACHER_KEY_FILE": str(key),
            "COPD_LEDGER_DIR": str(self.root / "ledger"),
            "COPD_RUN_ID": "fixture", "COPD_AVAILABILITY_SEED": "20260910",
            "COPD_CLI_BINARY": str(cli), "COPD_CLI_SHA256": teacher.digest("fixture"),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def call(self, raw=b"help with this evidence\0"):
        with teacher.command_mount(self.root / "episode", model_uid=os.getuid(),
                                   command="ordinary command", timeout_ms=5000) as mount:
            root = Path(mount)
            self.assertEqual(set(p.name for p in root.iterdir()),
                             {"copd_ask", "request", "reply", "lock"})
            with (root / "request").open("wb", buffering=0) as request:
                request.write(raw)
            with (root / "reply").open("rb", buffering=0) as reply:
                result = reply.read().decode()
        self.assertFalse(root.exists())
        event = json.loads(next((self.root / "ledger").glob("*.jsonl")).read_text().splitlines()[-1])
        return result, event

    def test_availability_fixed_and_both_arms_present(self):
        states = [teacher.episode_receipt(self.root / str(n)) for n in range(100)]
        self.assertEqual(states[4], teacher.episode_receipt(self.root / "4"))
        count = sum(x["teacher_available"] for x in states)
        self.assertGreater(count, 25)
        self.assertLess(count, 75)

    def test_off_has_no_upstream_and_is_accounted(self):
        with patch.dict(os.environ, {"COPD_TEACHER_ON_PROB": "0"}), patch.object(teacher, "_ask_bounded") as ask:
            result, event = self.call()
        self.assertEqual(result.strip(), teacher.UNAVAILABLE)
        ask.assert_not_called()
        self.assertTrue(event["delivered"])
        self.assertEqual(event["upstream_attempts"], 0)
        self.assertEqual(event["cost"], 0)
        self.assertNotIn("reward", event)

    def test_on_delivers_only_text_and_ledger(self):
        with patch.dict(os.environ, {"COPD_TEACHER_ON_PROB": "1"}), patch.object(
                teacher, "_ask_bounded", return_value=("Check the constraint.", {"status": "ok", "usage": {"total_tokens": 27}})) as ask:
            result, event = self.call()
        self.assertEqual(result, "Check the constraint.\n")
        self.assertEqual(ask.call_args.args[0], "help with this evidence")
        self.assertEqual(event["usage"]["total_tokens"], 27)
        self.assertTrue(event["delivered"])
        self.assertNotIn("teacher_logprobs", event)

    def test_bad_utf8_and_oversize_are_observations(self):
        for raw in [b"\xff\0", b"x" * (teacher.MAX_QUESTION_BYTES + 128) + b"\0"]:
            with self.subTest(raw_size=len(raw)), patch.object(teacher, "_ask_bounded") as ask:
                result, event = self.call(raw)
                self.assertIn("invalid_question", result)
                self.assertEqual(event["status"], "invalid_question")
                ask.assert_not_called()

    def test_private_command_excludes_mount(self):
        with patch.object(teacher, "_config", side_effect=AssertionError("private caller used teacher")):
            with teacher.command_mount("private", model_uid=0, command="grade", timeout_ms=50, permitted=False) as mount:
                self.assertEqual(mount, "")

    def test_mock_provider_usage_and_refusal(self):
        for message, status in [({"content": "Try a check."}, "ok"), ({"content": None, "refusal": "Cannot advise."}, "refusal")]:
            body = {"id": "provider-fixture", "choices": [{"message": message, "finish_reason": "stop"}], "usage": {"prompt_tokens": 14, "completion_tokens": 8, "total_tokens": 22}}
            with patch.object(teacher.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
                reply, detail = teacher._ask("question", config=teacher._config(), timeout=1)
            self.assertTrue(reply)
            self.assertEqual(detail["status"], status)
            self.assertEqual(detail["usage"]["total_tokens"], 22)

    def test_provider_timeout(self):
        with patch.object(teacher.urllib.request, "urlopen", side_effect=socket.timeout()):
            reply, detail = teacher._ask("question", config=teacher._config(), timeout=1)
        self.assertEqual(reply, "[teacher_error:timeout]")
        self.assertEqual(detail["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
