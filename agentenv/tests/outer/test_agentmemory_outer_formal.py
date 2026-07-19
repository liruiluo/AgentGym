from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_ROOT = PROJECT_ROOT.parents[1]
SUPPORT_ROOT = Path(
    os.environ.get(
        "AGENTENV_SUPPORT_ROOT",
        "/root/jd-coding/continual-reasoning-gym-workspace/code/AgentGym-RL/AgentGym/agentenv",
    )
)
for root in (PROJECT_ROOT, SUPPORT_ROOT):
    if (root / "agentenv" / "controller").exists():
        sys.path.insert(0, str(root))
        break
else:
    raise RuntimeError("agentenv controller support root is unavailable")

MODULE_PATH = PROJECT_ROOT / "agentenv" / "envs" / "agentmemory.py"
SPEC = importlib.util.spec_from_file_location("agentmemory_outer_candidate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
agentmemory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agentmemory)

sys.path.insert(0, str(STAGING_ROOT / "AgentGym-RL"))
SMOKE_PATH = (
    STAGING_ROOT
    / "AgentGym-RL/docs/agentmemorygym/scripts/smoke_formal_ppo_runtime_evidence.py"
)
SMOKE_SPEC = importlib.util.spec_from_file_location("agentmemory_runtime_smoke", SMOKE_PATH)
assert SMOKE_SPEC is not None and SMOKE_SPEC.loader is not None
runtime_smoke = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(runtime_smoke)


VALID_PAYLOADS = {
    "ADD": {"key": "selected_product", "value": "policy-authored memory"},
    "UPDATE": {"memory_id": "mem_0000", "value": "updated memory"},
    "DELETE": {"memory_id": "mem_0000"},
    "RETRIEVE": {"query": "prior selected product", "top_k": 3},
    "SUMMARY": {"text": "visible context summary", "source_ids": ["S0"]},
    "FILTER": {"keep_ids": ["C0"], "scope": "active"},
    "SEARCH": {"query": "gluten free cake mix"},
    "PAGE": {"cursor": "cur_0123456789abcdef01234567"},
    "BUY": {"product_id": "B000000015"},
    "ANSWER": {"text": "bundle complete"},
}


class OuterFormalContractTests(unittest.TestCase):
    @staticmethod
    def _fake_client():
        client = object.__new__(agentmemory.AgentMemoryEnvClient)
        client.adapter_cls = agentmemory.AgentMemoryAdapter
        client.action_format = agentmemory.ActionFormat.REACT
        client.metadata = {"task_count": 1}
        client.task_round = 0
        client.info = {
            "observation": "current observation",
            "reward": 0.0,
            "done": False,
            "env_info": {
                "current_subtask_index": 0,
                "session_trace": ["current session"],
                "tool_ops": [],
                "memory_ops": [],
                "reward_components": [
                    {"name": "environment_base_reward", "value": 0.0, "op": "SEARCH", "step": 0}
                ],
            },
            "metadata": client.metadata,
        }
        calls = []

        def post(path, data):
            calls.append((path, dict(data)))
            if path == "reset":
                return {
                    "observation": "reset observation",
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "current_subtask_index": 0,
                        "session_trace": [],
                        "tool_ops": [],
                        "memory_ops": [],
                        "reward_components": [],
                    },
                }
            parsed_op = str(data["action"]).split(None, 1)[0]
            advanced = parsed_op == "BUY"
            return {
                "observation": f"after {parsed_op}",
                "reward": 1.0 if advanced else 0.0,
                "done": False,
                "info": {
                    "current_subtask_index": 1 if advanced else 0,
                    "session_trace": [] if advanced else [parsed_op],
                    "episode_success": False,
                    "tool_ops": [
                        {
                            "op": parsed_op,
                            "step": len(calls),
                            **(
                                {"result_count": 1, "result_product_ids": ["B000000015"]}
                                if parsed_op == "SEARCH"
                                else {
                                    "committed": True,
                                    "purchase_correct": True,
                                    "session_advanced": True,
                                    "terminal": False,
                                }
                            ),
                        }
                    ],
                    "memory_ops": [],
                    "reward_components": [
                        {
                            "name": "environment_base_reward",
                            "value": 1.0 if advanced else 0.0,
                            "op": parsed_op,
                            "step": len(calls),
                        }
                    ],
                },
            }

        client.post = post
        return client, calls

    def test_action_and_function_exact_sets(self) -> None:
        expected_actions = tuple(VALID_PAYLOADS)
        self.assertEqual(agentmemory.accepted_action_names(), expected_actions)
        self.assertEqual(
            tuple(agentmemory.function_to_action().values()),
            expected_actions,
        )
        self.assertEqual(
            tuple(item["name"] for item in agentmemory.agentmemory_function_descriptions()),
            tuple(action.lower() for action in expected_actions),
        )

    def test_search_page_and_retrieve_function_schemas(self) -> None:
        schemas = {
            item["name"]: item["parameters"]
            for item in agentmemory.agentmemory_function_descriptions()
        }
        self.assertEqual(set(schemas["search"]["properties"]), {"query"})
        self.assertEqual(schemas["search"]["required"], ["query"])
        self.assertFalse(schemas["search"]["additionalProperties"])
        self.assertEqual(set(schemas["page"]["properties"]), {"cursor"})
        self.assertEqual(schemas["page"]["required"], ["cursor"])
        self.assertEqual(schemas["retrieve"]["required"], ["query", "top_k"])
        self.assertEqual(schemas["retrieve"]["properties"]["top_k"]["enum"], [3])
        self.assertIn(
            "every required purchased product_id",
            schemas["answer"]["properties"]["text"]["description"],
        )

    def test_three_decoded_prompts_share_neutral_objective_contract(self) -> None:
        prompts = {
            action_format.value: messages[0]["value"]
            for action_format, messages in agentmemory.AgentMemoryAdapter.conversation_start_dict.items()
        }
        self.assertEqual(set(prompts), {"react", "function_calling", "code_as_action"})
        required = (
            "Valid actions are exactly ADD, UPDATE, DELETE, RETRIEVE, SUMMARY, FILTER, SEARCH, PAGE, BUY, and ANSWER",
            "RETRIEVE requires query and top_k=3",
            "runtime does not retrieve memory automatically",
            "SEARCH accepts only a non-empty product-keyword query",
            "product name or title",
            "at most 10 ordered rows from a backend pool of at most 50",
            "PAGE accepts only an opaque current-session next_cursor",
            "ASIN displayed by SEARCH or PAGE",
            "incorrect BUY terminates the episode with reward -0.5 and no retry",
            "ANSWER accepts text only after the bundle is complete",
            "text must contain every required purchased product_id",
        )
        forbidden = (
            "CHOOSE",
            "Recommended workflow",
            "required workflow",
            "try a different",
            "Do not use",
            "Choose the visible",
            'SEARCH {"query": "...", "top_k": 3}',
        )
        for name, prompt in prompts.items():
            for fragment in required:
                self.assertIn(fragment, prompt, f"{name} missing {fragment!r}")
            for fragment in forbidden:
                self.assertNotIn(fragment, prompt, f"{name} contains {fragment!r}")
            self.assertIsNone(re.search(r"AGENTMEMORY_[A-Z0-9_]+", prompt))

        code_prompt = prompts["code_as_action"]
        signatures = code_prompt.split("```python\n", 1)[1].split("\n```", 1)[0]
        compile(signatures, "<decoded-code-action-signatures>", "exec")

    def test_react_parser_accepts_every_formal_action(self) -> None:
        for action_name, payload in VALID_PAYLOADS.items():
            raw = f"Thought:\nstate\n\nAction:\n{action_name} {json.dumps(payload)}"
            parsed = agentmemory.AgentMemoryAdapter.parse_react(raw)
            parsed_name, parsed_payload = agentmemory.parse_env_action(parsed.action)
            self.assertEqual((parsed_name, parsed_payload), (action_name, payload))

        bare = agentmemory.AgentMemoryAdapter.parse_react(
            'SEARCH {"query":"cake mix"}'
        )
        self.assertEqual(
            agentmemory.parse_env_action(bare.action),
            ("SEARCH", {"query": "cake mix"}),
        )

    def test_function_calling_parser_covers_search_page_and_retrieve(self) -> None:
        for function_name in ("search", "page", "retrieve"):
            payload = VALID_PAYLOADS[function_name.upper()]
            raw = json.dumps(
                {
                    "thought": "state",
                    "function_name": function_name,
                    "arguments": payload,
                }
            )
            parsed = agentmemory.AgentMemoryAdapter.parse_function_calling(raw)
            self.assertEqual(
                agentmemory.parse_env_action(parsed.action),
                (function_name.upper(), payload),
            )

    def test_code_as_action_parser_covers_search_page_and_retrieve(self) -> None:
        samples = {
            "SEARCH": '```python\n# state\nsearch(query="cake mix")\n```',
            "PAGE": '```python\n# state\npage(cursor="cur_0123456789abcdef01234567")\n```',
            "RETRIEVE": '```python\n# state\nretrieve(query="prior product", top_k=3)\n```',
        }
        for action_name, raw in samples.items():
            parsed = agentmemory.AgentMemoryAdapter.parse_code_as_action(raw)
            parsed_name, _ = agentmemory.parse_env_action(parsed.action)
            self.assertEqual(parsed_name, action_name)

    def test_all_three_parsers_reject_removed_or_malformed_actions(self) -> None:
        invalid_env_actions = (
            'CHOOSE {"choice":"A"}',
            'RETIREVE {"query":"memory","top_k":3}',
            'SEARCH {"query":"cake","top_k":3}',
            'PAGE {"cursor":"cur_x","query":"cake"}',
            'RETRIEVE {"query":"memory"}',
            'RETRIEVE {"query":"memory","top_k":5}',
            'FILTER {"keep_ids":["C0"],"drop_ids":["S0"],"scope":"all"}',
        )
        for raw in invalid_env_actions:
            with self.assertRaises(ValueError, msg=raw):
                agentmemory.parse_env_action(raw)

        with self.assertRaises(ValueError):
            agentmemory.AgentMemoryAdapter.parse_function_calling(
                json.dumps(
                    {
                        "thought": "state",
                        "function_name": "choose",
                        "arguments": {"choice": "A"},
                    }
                )
            )
        with self.assertRaises((NameError, ValueError)):
            agentmemory.AgentMemoryAdapter.parse_code_as_action(
                '```python\nchoose(choice="A")\n```'
            )
        with self.assertRaises(ValueError):
            agentmemory.AgentMemoryAdapter.parse_code_as_action(
                '```python\nsearch(query="cake", top_k=3)\n```'
            )

    def test_outer_invalid_action_ledger_is_current_and_formally_valid(self) -> None:
        client, calls = self._fake_client()
        client.step('SEARCH {"query":"cake"}')
        self.assertEqual(client.info["env_info"]["tool_ops"][0]["step"], 1)
        first_invalid = client.step('RETIEVE {"query":"memory","top_k":3}')
        self.assertEqual(len(calls), 1)
        first_component = client.info["env_info"]["reward_components"][0]
        self.assertEqual((first_component["op"], first_component["step"]), ("RETIEVE", 2))
        self.assertEqual(client.info["env_info"]["tool_ops"], [])
        self.assertEqual(client.info["env_info"]["memory_ops"], [])
        second_invalid = client.step('RETIREVE {"query":"memory","top_k":3}')
        self.assertEqual(len(calls), 1)
        second_component = client.info["env_info"]["reward_components"][0]
        self.assertEqual((second_component["op"], second_component["step"]), ("RETIREVE", 3))
        self.assertNotEqual(first_component["raw_action"], second_component["raw_action"])
        self.assertNotIn(first_component, client.info["env_info"]["reward_components"])
        client.step('BUY {"product_id":"B000000015"}')
        self.assertEqual(client.info["env_info"]["tool_ops"][0]["step"], 4)
        self.assertEqual(client.info["env_info"]["reward_components"][0]["step"], 4)
        self.assertIn("Invalid Action", first_invalid.state)
        self.assertIn("Invalid Action", second_invalid.state)

        validator_client, validator_calls = self._fake_client()
        invalid = validator_client.step('RETIEVE {"query":"memory","top_k":3}')
        self.assertEqual(validator_calls, [])
        rows = runtime_smoke.runtime_rows()
        immediate = [-0.1, 1.0, 2.0]
        suffix = [2.9, 3.0, 2.0]
        trajectory_return = 2.9
        rows["immediate_rewards"] = immediate
        rows["suffix_returns"] = suffix
        rows["trajectory_returns"] = [trajectory_return] * 3
        rows["action_texts"][0] = 'RETIEVE {"query":"memory","top_k":3}'
        for index, raw_record in enumerate(rows["step_record_jsons"]):
            record = json.loads(raw_record)
            record["immediate_reward"] = immediate[index]
            record["suffix_return"] = suffix[index]
            record["trajectory_return"] = trajectory_return
            if index == 0:
                record["action"] = rows["action_texts"][0]
                record["env_result"] = invalid.state
                record["env_info_after"] = validator_client.info["env_info"]
            rows["step_record_jsons"][index] = json.dumps(record)
        summary = runtime_smoke.validate_formal_runtime_evidence_rows(
            **rows,
            expected_suffix_credit=True,
        )
        self.assertEqual(summary["valid_rows"], 3)
        self.assertEqual(summary["trajectory_count"], 1)

    def test_active_outer_source_has_zero_deprecated_controls_or_coaching(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        forbidden = (
            "CHOOSE",
            "LEGACY_R4",
            "DIRECT_BUY",
            "ADD_ONLY",
            "BUY_ONLY",
            "ADDBUY",
            "DELAYED",
            "Recommended workflow",
            "required workflow",
            "try a different",
            "Do not use",
            "Choose the visible candidate",
            'SEARCH {"query": "...", "top_k": 3}',
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, source)
        self.assertNotIn("os.environ", source)
        self.assertNotIn("_truthy_env", source)


if __name__ == "__main__":
    unittest.main()
