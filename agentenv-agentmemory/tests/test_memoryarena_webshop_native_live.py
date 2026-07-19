from __future__ import annotations

import os
import unittest
from concurrent.futures import ThreadPoolExecutor

from agentenv_agentmemory.memoryarena_dataset import load_memoryarena_dataset
from agentenv_agentmemory.memoryarena_webshop_env import MemoryArenaWebShopEnv
from agentenv_agentmemory.native_webshop_backend import MemoryArenaNativeWebShopBackend


LIVE = os.environ.get("AGENTMEMORY_RUN_NATIVE_LIVE_TESTS") == "1"


@unittest.skipUnless(LIVE, "set AGENTMEMORY_RUN_NATIVE_LIVE_TESTS=1 on the 9N native runtime")
class NativeMemoryArenaLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = [
            "MEMORYARENA_ROOT",
            "AGENTMEMORY_MEMORYARENA_RAW_PATH",
            "MEMORYARENA_WEBSHOP_ITEMS_FILE",
            "MEMORYARENA_WEBSHOP_ATTR_FILE",
            "MEMORYARENA_WEBSHOP_SEARCH_ROOT",
            "MEMORYARENA_WEBSHOP_JAVA_HOME",
        ]
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise RuntimeError(f"Missing live native environment variables: {missing}")
        cls.backend = MemoryArenaNativeWebShopBackend(
            memoryarena_root=os.environ["MEMORYARENA_ROOT"],
            items_file=os.environ["MEMORYARENA_WEBSHOP_ITEMS_FILE"],
            attributes_file=os.environ["MEMORYARENA_WEBSHOP_ATTR_FILE"],
            search_root=os.environ["MEMORYARENA_WEBSHOP_SEARCH_ROOT"],
            java_home=os.environ["MEMORYARENA_WEBSHOP_JAVA_HOME"],
            price_seed=int(os.environ.get("AGENTMEMORY_WEBSHOP_PRICE_SEED", "233")),
        )
        cls.dataset = load_memoryarena_dataset(
            os.environ["AGENTMEMORY_MEMORYARENA_RAW_PATH"],
            frozen_product_asins=cls.backend.product_asins(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()

    def test_page_price_matches_structured_purchase_ledger(self) -> None:
        bundle = self.dataset.get("baking_item_0")
        asin = bundle.target_asins[0]
        token = "live_price_parity"
        page = self.backend.open_session(token, bundle.questions[0])
        try:
            page = self.backend.step(token, f"search[{_argument(self.backend.product_title(asin))}]")
            page = _open_asin(self.backend, token, page, asin)
            product_page = page.observation
            done_page = self.backend.step(token, "click[Buy Now]")
            self.assertIsNotNone(done_page.purchase)
            expected_price = f"${done_page.purchase.price_cents / 100:.2f}"
            self.assertIn(expected_price, product_page)
        finally:
            self.backend.close_session(token)

    def test_complete_six_purchase_oracle(self) -> None:
        bundle = self.dataset.get("baking_item_0")
        env = MemoryArenaWebShopEnv(
            bundles=[bundle],
            backend=self.backend,
            env_uid="live_full_chain",
        )
        try:
            env.reset()
            for index, asin in enumerate(bundle.target_asins):
                env.step(f"search[{_argument(self.backend.product_title(asin))}]")
                _open_env_asin(env, asin)
                observation, reward, done, _, info = env.step("click[Buy Now]")
                self.assertTrue(info["tool_ops"][0]["purchase_correct"], info)
                self.assertEqual(2.0 if index == 5 else 1.0, reward)
            self.assertTrue(done)
            self.assertTrue(info["episode_success"])
            self.assertNotIn("reward=", observation)
        finally:
            env.close()

    def test_48_session_threaded_isolation_and_lucene_search(self) -> None:
        bundle = self.dataset.get("baking_item_0")
        query = _argument(self.backend.product_title(bundle.target_asins[0]))

        def open_and_search(index: int):
            token = f"live_concurrency_{index}"
            page = self.backend.open_session(token, f"private-instruction-{index}")
            page = self.backend.step(token, f"search[{query}]")
            return token, page

        tokens: list[str] = []
        try:
            with ThreadPoolExecutor(max_workers=48) as executor:
                results = list(executor.map(open_and_search, range(48)))
            tokens = [token for token, _ in results]
            self.assertEqual(48, len(set(tokens)))
            for index, (token, page) in enumerate(results):
                self.assertIn(f"private-instruction-{index}", page.observation)
                self.assertNotIn(
                    f"private-instruction-{(index + 1) % 48}",
                    page.observation,
                )
                self.assertTrue(page.clickables)
                self.assertIn(token, self.backend._envs)
        finally:
            for token in tokens:
                self.backend.close_session(token)


def _open_asin(backend, token: str, page, asin: str):
    target = asin.lower()
    for _ in range(5):
        if target in {value.lower() for value in page.clickables}:
            return backend.step(token, f"click[{asin}]")
        if "next >" not in {value.lower() for value in page.clickables}:
            break
        page = backend.step(token, "click[Next >]")
    raise AssertionError(f"ASIN {asin} missing from exact-title Lucene top-50")


def _open_env_asin(env: MemoryArenaWebShopEnv, asin: str) -> None:
    target = asin.lower()
    for _ in range(5):
        page = env.native_page
        if page is None:
            raise AssertionError("native page missing")
        if target in {value.lower() for value in page.clickables}:
            env.step(f"click[{asin}]")
            return
        if "next >" not in {value.lower() for value in page.clickables}:
            break
        env.step("click[Next >]")
    raise AssertionError(f"ASIN {asin} missing from exact-title Lucene top-50")


def _argument(value: str) -> str:
    text = " ".join(value.split())
    if any(char in text for char in "[]\r\n"):
        raise ValueError(text)
    return text


if __name__ == "__main__":
    unittest.main()
