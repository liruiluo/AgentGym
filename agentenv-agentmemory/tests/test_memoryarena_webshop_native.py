from __future__ import annotations

import unittest

from agentenv_agentmemory.memoryarena_dataset import (
    MemoryArenaBundle,
    MemoryArenaBundleProvenance,
    MemoryArenaSession,
)
from agentenv_agentmemory.memoryarena_webshop_env import (
    InvalidNativeAction,
    MemoryArenaWebShopEnv,
    parse_mixed_action,
)
from agentenv_agentmemory.native_webshop_backend import NativePage, NativePurchase
from agentenv_agentmemory.reward_hierarchy import (
    EXACT_REPEAT_ACTION_PENALTY,
    FIRST_VALID_ADD_BONUS,
    FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
    INVALID_ACTION_PENALTY,
    WRONG_BUY_TERMINAL_FAILURE,
)


TARGETS = tuple(f"B0000000{index:02d}" for index in range(1, 7))
WRONG = "B999999999"


class FakeNativeBackend:
    surface = "memoryarena_webshop_native_v1"

    def __init__(self, prices: dict[str, int] | None = None) -> None:
        self.prices = {asin: 100 for asin in TARGETS}
        self.prices[WRONG] = 100
        self.prices.update(prices or {})
        self.sessions: dict[str, dict[str, object]] = {}
        self.closed_tokens: list[str] = []

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        if session_token in self.sessions:
            raise ValueError("duplicate token")
        self.sessions[session_token] = {"instruction": instruction, "asin": None}
        return NativePage(
            observation=f"WebShop [SEP] Instruction: [SEP] {instruction} [SEP] Search",
            url=f"http://native/{session_token}",
            has_search_bar=True,
            clickables=(),
        )

    def step(self, session_token: str, action: str) -> NativePage:
        session = self.sessions[session_token]
        instruction = str(session["instruction"])
        if action.startswith("search["):
            return NativePage(
                observation=f"{instruction} [SEP] Results [SEP] " + " [SEP] ".join(TARGETS + (WRONG,)),
                url=f"http://native/search/{session_token}",
                has_search_bar=True,
                clickables=TARGETS + (WRONG,),
            )
        if action.startswith("click[") and action.lower() != "click[buy now]":
            asin = action[6:-1].upper()
            session["asin"] = asin
            return NativePage(
                observation=f"{instruction} [SEP] asin [SEP] {asin} [SEP] Buy Now",
                url=f"http://native/item/{session_token}/{asin}",
                has_search_bar=False,
                clickables=("Buy Now",),
            )
        if action.lower() == "click[buy now]":
            asin = str(session["asin"] or "")
            if not asin:
                return NativePage(
                    observation=f"{instruction} [SEP] Search",
                    url=f"http://native/{session_token}",
                    has_search_bar=True,
                    clickables=(),
                )
            return NativePage(
                observation=(
                    f"{instruction} [SEP] Thank you [SEP] asin [SEP] {asin} "
                    "[SEP] SYNTHETIC_TARGET_SECRET [SEP] reward=1"
                ),
                url=f"http://native/done/{session_token}/{asin}",
                has_search_bar=False,
                clickables=(),
                purchase=NativePurchase(
                    asin=asin,
                    price_cents=self.prices[asin],
                    selected_options={},
                ),
            )
        raise ValueError(action)

    def close_session(self, session_token: str) -> None:
        self.sessions.pop(session_token, None)
        self.closed_tokens.append(session_token)

    def has_product(self, asin: str) -> bool:
        return asin.upper() in self.prices

    def metadata(self):
        return {"surface": self.surface}


def make_bundle(*, budget_cents: int = 10_000) -> MemoryArenaBundle:
    questions = tuple(f"Question {index}; total budget ${budget_cents / 100:.2f}." for index in range(1, 7))
    sessions = tuple(
        MemoryArenaSession(
            session_index=index,
            question=questions[index],
            instruction=f"Instruction {index + 1}",
            candidate_context="Available options",
            candidate_options=("Option A", "Option B"),
            raw_target_asin=TARGETS[index],
            target_asin=TARGETS[index],
            answer_attributes=(),
        )
        for index in range(6)
    )
    provenance = MemoryArenaBundleProvenance(
        raw_dataset_path="fixture.jsonl",
        raw_dataset_sha256="0" * 64,
        memoryarena_commit="1" * 40,
        domain_data_sha256="2" * 64,
        split_strategy="fixture",
        split_manifest_sha256="3" * 64,
        source_position=0,
        source_line_number=1,
        target_asin_membership_verified=True,
    )
    return MemoryArenaBundle(
        task_id="fixture_chain_0",
        questions=questions,
        target_asins=TARGETS,
        budget_cents=budget_cents,
        split="train",
        source_row_id=0,
        provenance=provenance,
        sessions=sessions,
        category="fixture",
        answer_attributes=((),) * 6,
    )


def purchase(env: MemoryArenaWebShopEnv, asin: str):
    env.step("search[item]")
    env.step(f"click[{asin}]")
    return env.step("click[Buy Now]")


class MixedActionParserTests(unittest.TestCase):
    def test_accepts_native_and_memory_actions(self) -> None:
        native = parse_mixed_action("search[red shoes]")
        self.assertEqual(native.raw_action, "search[red shoes]")
        self.assertEqual(native.op, "SEARCH")
        parsed = parse_mixed_action('ADD {"key":"k","value":"v"}')
        self.assertEqual(parsed.op, "ADD")
        self.assertEqual(parsed.payload, {"key": "k", "value": "v"})

    def test_rejects_surrogate_and_multiple_actions(self) -> None:
        for action in [
            'SEARCH {"query":"x"}',
            'BUY {"product_id":"x"}',
            'GROUND {"candidate_id":"x"}',
            "search[x]\nclick[y]",
            "click[x] trailing",
        ]:
            with self.subTest(action=action), self.assertRaises(InvalidNativeAction):
                parse_mixed_action(action)


class MemoryArenaWebShopEnvTests(unittest.TestCase):
    def make_env(self, *, backend: FakeNativeBackend | None = None, budget_cents: int = 10_000):
        backend = backend or FakeNativeBackend()
        env = MemoryArenaWebShopEnv(bundles=[make_bundle(budget_cents=budget_cents)], backend=backend, env_uid="test")
        env.reset()
        return env, backend

    def assert_reward_ledger(
        self,
        *,
        reward: float,
        info: dict[str, object],
        op: str,
        names: list[str],
        step: int,
    ) -> None:
        components = info["reward_components"]
        self.assertIsInstance(components, list)
        self.assertTrue(components)
        self.assertEqual([item["name"] for item in components], names)
        self.assertEqual({item["op"] for item in components}, {op})
        self.assertEqual({item["step"] for item in components}, {step})
        self.assertAlmostEqual(sum(float(item["value"]) for item in components), reward)

    def test_native_search_and_nonpurchase_click_have_exact_zero_ledgers(self) -> None:
        env, _ = self.make_env()

        _, reward, done, _, info = env.step("search[item]")

        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["tool_ops"][0]["op"], "SEARCH")
        self.assertEqual(info["tool_ops"][0]["result_count"], 7)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="SEARCH",
            names=["search_transition"],
            step=1,
        )

        _, reward, done, _, info = env.step(f"click[{TARGETS[0]}]")

        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["tool_ops"][0]["op"], "CLICK")
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="CLICK",
            names=["click_transition"],
            step=2,
        )

    def test_first_add_has_micro_bonus_and_first_session_retrieve_stays_zero(self) -> None:
        env, _ = self.make_env()

        _, reward, done, _, info = env.step('ADD {"key":"screen","value":"LED television"}')

        self.assertEqual(reward, FIRST_VALID_ADD_BONUS)
        self.assertFalse(done)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="ADD",
            names=["add_transition", "memory_add_first_valid_this_session"],
            step=1,
        )

        observation, reward, done, _, info = env.step('RETRIEVE {"query":"LED","top_k":3}')

        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertIn("LED television", observation)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="RETRIEVE",
            names=["retrieve_transition"],
            step=2,
        )

    def test_memory_micro_bonuses_are_once_per_session_and_reset_on_advance(self) -> None:
        env, _ = self.make_env()
        add_action = 'ADD {"key":"screen","value":"LED television"}'

        _, reward, _, _, info = env.step(add_action)
        self.assertEqual(reward, FIRST_VALID_ADD_BONUS)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="ADD",
            names=["add_transition", "memory_add_first_valid_this_session"],
            step=1,
        )

        _, reward, _, _, info = env.step('ADD {"key":"size","value":"32 inch"}')
        self.assertEqual(reward, 0.0)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="ADD",
            names=["add_transition"],
            step=2,
        )

        _, reward, _, _, info = env.step(add_action)
        self.assertEqual(reward, EXACT_REPEAT_ACTION_PENALTY)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="ADD",
            names=["add_transition", "exact_repeated_valid_zero_reward_action"],
            step=3,
        )

        purchase(env, TARGETS[0])
        _, reward, _, _, info = env.step(add_action)
        self.assertEqual(reward, FIRST_VALID_ADD_BONUS)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="ADD",
            names=["add_transition", "memory_add_first_valid_this_session"],
            step=7,
        )

    def test_later_session_first_retrieve_bonus_is_once_and_resets(self) -> None:
        env, _ = self.make_env()
        env.step('ADD {"key":"screen","value":"LED television"}')
        retrieve_action = 'RETRIEVE {"query":"LED","top_k":3}'

        _, reward, _, _, _ = env.step(retrieve_action)
        self.assertEqual(reward, 0.0)

        purchase(env, TARGETS[0])
        _, reward, _, _, info = env.step(retrieve_action)
        self.assertEqual(reward, FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="RETRIEVE",
            names=["retrieve_transition", "memory_retrieve_first_valid_later_session"],
            step=6,
        )

        _, reward, _, _, info = env.step('RETRIEVE {"query":"television","top_k":3}')
        self.assertEqual(reward, 0.0)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="RETRIEVE",
            names=["retrieve_transition"],
            step=7,
        )

        _, reward, _, _, info = env.step(retrieve_action)
        self.assertEqual(reward, EXACT_REPEAT_ACTION_PENALTY)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="RETRIEVE",
            names=["retrieve_transition", "exact_repeated_valid_zero_reward_action"],
            step=8,
        )

        purchase(env, TARGETS[1])
        _, reward, _, _, info = env.step(retrieve_action)
        self.assertEqual(reward, FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="RETRIEVE",
            names=["retrieve_transition", "memory_retrieve_first_valid_later_session"],
            step=12,
        )

    def test_exact_repeated_valid_action_penalty_resets_on_session_advance(self) -> None:
        env, _ = self.make_env()

        _, first_reward, _, _, _ = env.step("search[item]")
        _, repeat_reward, _, _, repeat_info = env.step("  search[item]  ")
        _, different_reward, _, _, _ = env.step("search[other item]")

        self.assertEqual(first_reward, 0.0)
        self.assertEqual(repeat_reward, EXACT_REPEAT_ACTION_PENALTY)
        self.assertEqual(different_reward, 0.0)
        self.assert_reward_ledger(
            reward=repeat_reward,
            info=repeat_info,
            op="SEARCH",
            names=["search_transition", "exact_repeated_valid_zero_reward_action"],
            step=2,
        )

        purchase(env, TARGETS[0])
        _, reward, _, _, info = env.step("search[item]")
        self.assertEqual(reward, 0.0)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="SEARCH",
            names=["search_transition"],
            step=7,
        )

    def test_correct_purchase_advances_without_add(self) -> None:
        env, backend = self.make_env()
        first_token = env.native_session_token

        observation, reward, done, _, info = purchase(env, TARGETS[0])

        self.assertEqual(reward, 1.0)
        self.assertFalse(done)
        self.assertEqual(info["current_subtask_index"], 1)
        self.assertTrue(info["tool_ops"][0]["purchase_correct"])
        self.assertTrue(info["tool_ops"][0]["session_advanced"])
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="BUY",
            names=["buy_committed_correct"],
            step=3,
        )
        self.assertIn(first_token, backend.closed_tokens)
        self.assertNotEqual(first_token, env.native_session_token)
        self.assertIn("Question 2", observation)
        self.assertNotIn("fixture_chain_0", observation)

    def test_wrong_purchase_is_fail_fast_and_feedback_has_no_target(self) -> None:
        env, _ = self.make_env()

        observation, reward, done, _, info = purchase(env, WRONG)

        self.assertEqual(reward, WRONG_BUY_TERMINAL_FAILURE)
        self.assertTrue(done)
        self.assertEqual(info["current_subtask_index"], 0)
        event = info["tool_ops"][0]
        self.assertTrue(event["committed"])
        self.assertFalse(event["purchase_correct"])
        self.assertFalse(event["session_advanced"])
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="BUY",
            names=["buy_committed_incorrect"],
            step=3,
        )
        self.assertNotIn("incorrect", observation.lower())
        self.assertNotIn("expected", observation.lower())
        self.assertNotIn("target asin", observation.lower())
        self.assertNotIn("SYNTHETIC_TARGET_SECRET", observation)

    def test_budget_uses_structured_cents_and_overflow_terminates(self) -> None:
        backend = FakeNativeBackend(prices={TARGETS[0]: 101})
        env, _ = self.make_env(backend=backend, budget_cents=100)

        _, reward, done, _, info = purchase(env, TARGETS[0])

        self.assertEqual(reward, WRONG_BUY_TERMINAL_FAILURE)
        self.assertTrue(done)
        self.assertFalse(info["tool_ops"][0]["budget_ok"])
        self.assertEqual(info["tool_ops"][0]["actual_price_cents"], 101)

    def test_long_term_memory_is_exact_and_hidden_across_session_boundary(self) -> None:
        env, _ = self.make_env()
        authored = "Remember the exact prior choice: Alpha 7."
        env.step(f'ADD {{"key":"prior","value":"{authored}"}}')

        observation, _, _, _, _ = purchase(env, TARGETS[0])

        self.assertNotIn(authored, observation)
        observation, reward, done, _, info = env.step('RETRIEVE {"query":"Alpha","top_k":3}')
        self.assertEqual(reward, FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS)
        self.assertFalse(done)
        self.assertIn(authored, observation)
        self.assertEqual(info["memory_ops"][0]["retrieved_count"], 1)

    def test_full_six_purchase_chain_succeeds(self) -> None:
        env, backend = self.make_env()
        rewards = []
        for asin in TARGETS:
            observation, reward, done, _, info = purchase(env, asin)
            rewards.append(reward)

        self.assertEqual(rewards, [1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
        self.assertTrue(done)
        self.assertTrue(info["episode_success"])
        self.assertEqual(info["current_subtask_index"], 6)
        self.assertEqual([item["actual_asin"] for item in info["purchase_history"]], list(TARGETS))
        self.assertEqual(len(backend.closed_tokens), 6)
        self.assert_reward_ledger(
            reward=rewards[-1],
            info=info,
            op="BUY",
            names=["buy_committed_correct", "bundle_complete_bonus"],
            step=18,
        )
        self.assertNotIn("SYNTHETIC_TARGET_SECRET", observation)

    def test_invalid_actions_have_exact_bound_negative_ledgers(self) -> None:
        env, _ = self.make_env()
        for step, raw_action in enumerate(
            ('BUY {"product_id":"not-native"}', "ADD {bad}"),
            start=1,
        ):
            with self.subTest(raw_action=raw_action):
                _, reward, done, _, info = env.step(raw_action)

                self.assertEqual(reward, INVALID_ACTION_PENALTY)
                self.assertFalse(done)
                self.assertEqual(info["tool_ops"], [])
                self.assert_reward_ledger(
                    reward=reward,
                    info=info,
                    op="INVALID",
                    names=["invalid_action"],
                    step=step,
                )
                component = info["reward_components"][0]
                self.assertEqual(component["raw_action"], raw_action)
                self.assertTrue(component["error"])

        raw_action = 'ADD {"key":"","value":"missing key"}'
        _, reward, done, _, info = env.step(raw_action)

        self.assertEqual(reward, INVALID_ACTION_PENALTY)
        self.assertFalse(done)
        self.assert_reward_ledger(
            reward=reward,
            info=info,
            op="ADD",
            names=["invalid_action"],
            step=3,
        )
        self.assertEqual(info["reward_components"][0]["raw_action"], raw_action)

    def test_initial_observation_hides_stable_dataset_row_identity(self) -> None:
        env, _ = self.make_env()

        observation, _ = env.reset()

        self.assertNotIn("fixture_chain_0", observation)

    def test_interleaved_envs_use_isolated_native_sessions_and_memory(self) -> None:
        backend = FakeNativeBackend()
        env_a = MemoryArenaWebShopEnv(bundles=[make_bundle()], backend=backend, env_uid="a")
        env_b = MemoryArenaWebShopEnv(bundles=[make_bundle()], backend=backend, env_uid="b")
        env_a.reset()
        env_b.reset()
        token_a = env_a.native_session_token
        token_b = env_b.native_session_token

        env_a.step('ADD {"key":"owner","value":"only-a"}')
        observation_b, _, _, _, _ = env_b.step('RETRIEVE {"query":"owner"}')

        self.assertNotEqual(token_a, token_b)
        self.assertIn(token_a, backend.sessions)
        self.assertIn(token_b, backend.sessions)
        self.assertNotIn("only-a", observation_b)

    def test_info_never_exposes_expected_label_fields(self) -> None:
        env, _ = self.make_env()
        _, info = env.reset()

        serialized_keys = " ".join(info.keys()).lower()
        self.assertNotIn("target", serialized_keys)
        self.assertNotIn("answer", serialized_keys)
        self.assertNotIn("expected", serialized_keys)


if __name__ == "__main__":
    unittest.main()
