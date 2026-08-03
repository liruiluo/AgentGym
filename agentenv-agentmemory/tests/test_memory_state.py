from __future__ import annotations

import unittest

from agentenv_agentmemory.memory_state import MemoryEntry, rank_memory_entries_bm25


class MemoryStateTests(unittest.TestCase):
    def test_bm25_score_is_independent_of_query_term_order(self) -> None:
        entry = MemoryEntry(
            memory_id="mem_0000",
            key="standing constraints",
            value="customer alpha never accepts red or gray color",
            created_step=1,
            updated_step=1,
        )
        forward = rank_memory_entries_bm25(
            "customer alpha standing constraints gray red",
            [entry],
            top_k=1,
        )
        reverse = rank_memory_entries_bm25(
            "red gray constraints standing alpha customer",
            [entry],
            top_k=1,
        )
        self.assertEqual(forward[0][1], reverse[0][1])

    def test_bm25_ties_use_memory_id_order(self) -> None:
        entries = [
            MemoryEntry("mem_0001", "profile", "alpha", 1, 1),
            MemoryEntry("mem_0000", "profile", "alpha", 1, 1),
        ]
        ranked = rank_memory_entries_bm25("profile alpha", entries, top_k=2)
        self.assertEqual([item.memory_id for item, _ in ranked], ["mem_0000", "mem_0001"])


if __name__ == "__main__":
    unittest.main()
