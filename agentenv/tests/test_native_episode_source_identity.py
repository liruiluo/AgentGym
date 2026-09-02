from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agentenv.envs.agentmemory import AgentMemoryEnvClient
from agentenv.envs.literesearcher import LiteResearcherEnvClient
from agentenv.envs.openmle_fast import OpenMLEFastEnvClient
from agentenv.envs.swesmith import (
    SWE_MEMORY_CONTRACT,
    SwesmithEnvClient,
)


SCHEMA = "camg_native_episode_source_identity_v1"


class NativeEpisodeSourceIdentityTests(unittest.TestCase):
    def test_shop_reset_exposes_explicit_orbit_identity(self) -> None:
        client = object.__new__(AgentMemoryEnvClient)
        client.env_id = 7
        client.metadata = {}
        client.is_procedural = True
        client.last_action_submission = None
        client.post = Mock(
            return_value={
                "observation": "shop",
                "reward": 0.0,
                "done": False,
                "info": {
                    "data_idx": 12,
                    "scenario_id": "baking",
                    "orbit_index": 6,
                },
            }
        )

        client.reset(12)

        self.assertEqual(
            client.episode_source_identity,
            {
                "schema": SCHEMA,
                "route_id": "webshop",
                "data_idx": 12,
                "scenario_id": "baking",
                "orbit_index": 6,
            },
        )

    def test_shop_reset_fails_closed_without_explicit_orbit_identity(self) -> None:
        client = object.__new__(AgentMemoryEnvClient)
        client.env_id = 7
        client.metadata = {}
        client.is_procedural = True
        client.episode_source_identity = {"stale": True}
        client.last_action_submission = None
        client.post = Mock(
            return_value={
                "observation": "shop",
                "reward": 0.0,
                "done": False,
                "info": {"data_idx": 12, "scenario_id": "baking"},
            }
        )

        with self.assertRaisesRegex(RuntimeError, "orbit_index"):
            client.reset(12)
        self.assertIsNone(client.episode_source_identity)

    def test_swesmith_private_detail_token_is_pinned_and_identity_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "detail.token"
            token_path.write_text("private-token\n", encoding="utf-8")
            token_path.chmod(0o600)
            token_sha = hashlib.sha256(token_path.read_bytes()).hexdigest()
            with patch.object(
                SwesmithEnvClient,
                "_request",
                side_effect=[
                    {"memory_contract": SWE_MEMORY_CONTRACT, "task_count": 4},
                    {"id": 9, "observation": "created"},
                ],
            ):
                client = SwesmithEnvClient(
                    "http://127.0.0.1:65124",
                    detail_token_path=str(token_path),
                    detail_token_sha256=token_sha,
                )
            client._request = Mock(
                side_effect=[
                    {"observation": "reset", "reward": 0.0, "done": False, "info": {}},
                    {
                        "data_idx": 3,
                        "instance_id": "pallets.flask.issue-3",
                        "base_repository": "pallets/flask",
                    },
                ]
            )

            client.reset(3)

            self.assertEqual(
                client.episode_source_identity,
                {
                    "schema": SCHEMA,
                    "route_id": "swesmith",
                    "data_idx": 3,
                    "instance_id": "pallets.flask.issue-3",
                    "base_repository": "pallets/flask",
                },
            )
            self.assertEqual(
                client._request.call_args_list[1].kwargs["headers"],
                {"X-SWESMITH-Detail-Token": "private-token"},
            )

    def test_swesmith_rejects_non_private_token_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "detail.token"
            token_path.write_text("private-token\n", encoding="utf-8")
            token_path.chmod(0o644)
            token_sha = hashlib.sha256(token_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "0600"):
                SwesmithEnvClient(
                    "http://127.0.0.1:65124",
                    detail_token_path=str(token_path),
                    detail_token_sha256=token_sha,
                )

    def test_literesearcher_reset_exposes_frozen_row_identity(self) -> None:
        client = object.__new__(LiteResearcherEnvClient)
        client.env_id = 11
        client.episode_source_identity = None
        client._request = Mock(
            return_value={
                "observation": "research",
                "reward": 0.0,
                "done": False,
                "info": {
                    "data_idx": 5,
                    "row_identity": "a" * 64,
                    "source_pool_index": 77,
                },
            }
        )

        client.reset(5)

        self.assertEqual(
            client.episode_source_identity,
            {
                "schema": SCHEMA,
                "route_id": "literesearcher",
                "data_idx": 5,
                "row_identity": "a" * 64,
                "source_pool_index": 77,
            },
        )

    def test_openmle_reset_exposes_manifest_bound_identity(self) -> None:
        client = object.__new__(OpenMLEFastEnvClient)
        client.env_id = 13
        client.data_len = 10
        client.metadata = {"manifest_sha256": "b" * 64, "role": "heldout"}
        client.info = {}
        client._episode_identity = None
        client.episode_source_identity = None
        response = {
            "observation": "automl",
            "info": {"truncated": False},
        }
        client._request = Mock(return_value=response)
        env_info = {
            "truncated": False,
            "counters": {"action_count": 0},
            "data_idx": 2,
            "task_id": "competition@2",
            "source_family": "KAGGLE_DATASET:a/b",
            "manifest_role": "heldout",
            "manifest_sha256": "b" * 64,
        }
        with patch(
            "agentenv.envs.openmle_fast._validate_step_response",
            return_value=("automl", 0.0, False, env_info),
        ):
            client.reset(2)

        self.assertEqual(
            client.episode_source_identity,
            {
                "schema": SCHEMA,
                "route_id": "openmle_fast",
                "data_idx": 2,
                "task_id": "competition@2",
                "source_family": "KAGGLE_DATASET:a/b",
                "manifest_role": "heldout",
                "manifest_sha256": "b" * 64,
            },
        )


if __name__ == "__main__":
    unittest.main()
