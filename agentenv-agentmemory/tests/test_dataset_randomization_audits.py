#!/usr/bin/env python3
"""Regression tests for the MemoryArena dataset-randomization audit tools."""

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parent.parent
    / "scripts"
    / "audits"
    / "dataset_randomization"
    / "audit_full_catalog_pools.py"
)
if not SCRIPT_PATH.exists():
    SCRIPT_PATH = Path(__file__).with_name("audit_full_catalog_pools.py")
SPEC = importlib.util.spec_from_file_location("audit_full_catalog_pools", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CatalogCategoryMatchingTest(unittest.TestCase):
    def test_exact_category_matches(self) -> None:
        category = "Beauty & Personal Care › Skin Care"
        self.assertTrue(MODULE.category_matches(category, category))

    def test_descendant_category_matches(self) -> None:
        canonical = "Beauty & Personal Care › Skin Care"
        descendant = canonical + " › Face › Treatments"
        self.assertTrue(MODULE.category_matches(canonical, descendant))

    def test_text_prefix_without_category_boundary_does_not_match(self) -> None:
        canonical = "Beauty & Personal Care › Skin Care"
        similar = "Beauty & Personal Care › Skin Careful Products"
        self.assertFalse(MODULE.category_matches(canonical, similar))


class PriceParsingTest(unittest.TestCase):
    def test_missing_price_uses_runtime_default_bucket(self) -> None:
        self.assertEqual(MODULE.parse_price_kind(""), ("missing_default_100", []))

    def test_single_and_range_prices_are_distinguished(self) -> None:
        self.assertEqual(MODULE.parse_price_kind("$12.50"), ("single", [12.5]))
        self.assertEqual(MODULE.parse_price_kind("$10.00 - $20.00"), ("range", [10.0, 20.0]))


if __name__ == "__main__":
    unittest.main()
