from __future__ import annotations

import unittest

from agentenv_agentmemory.memoryarena_dataset import (
    MemoryArenaBundle,
    MemoryArenaBundleProvenance,
    MemoryArenaSession,
)
from agentenv_agentmemory.memoryarena_webshop_env import MemoryArenaWebShopEnv
from agentenv_agentmemory.native_webshop_backend import NativePage
from agentenv_agentmemory.presentation_randomization import (
    PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1,
    PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2,
    PRESENTATION_RANDOMIZATION_NONE,
    PresentationRandomizationError,
    build_presentation_variant,
    reorder_candidate_options,
    split_candidate_block,
    unique_bundle_permutation_ranks,
)


OPTIONS = (
    "Alpha product with exact title",
    "Beta product with exact title",
    "Gamma product with exact title",
)
FIVE_OPTIONS = OPTIONS + (
    "Delta product with exact title",
    "Epsilon product with exact title",
)
TARGETS = tuple(f"B0000000{index:02d}" for index in range(1, 7))


def make_question(step_index: int, options: tuple[str, ...] = OPTIONS) -> str:
    return (
        "You are an intelligent Shopping Agent operating in a webshop.\n\n"
        "*** GLOBAL RULES ***\n"
        "1. Keep every non-option byte unchanged.\n\n"
        + "-" * 64
        + "\n"
        + f"Product {step_index}:\n"
        + f"**Goal:** Select product {step_index}.\n"
        + "**Available Options:**\n"
        + "".join(f"- {option}\n" for option in options)
    )


def make_bundle_with_options(
    options_by_session: tuple[tuple[str, ...], ...],
) -> MemoryArenaBundle:
    questions = tuple(
        make_question(index, options_by_session[index - 1])
        for index in range(1, 7)
    )
    sessions = tuple(
        MemoryArenaSession(
            session_index=index,
            question=questions[index],
            instruction=f"Instruction {index + 1}",
            candidate_context=(
                "**Available Options:**\n"
                + "".join(
                    f"- {option}\n" for option in options_by_session[index]
                )
            ),
            candidate_options=options_by_session[index],
            raw_target_asin=TARGETS[index],
            target_asin=TARGETS[index],
            answer_attributes=(f"attribute-{index}",),
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
        source_position=7,
        source_line_number=8,
        target_asin_membership_verified=True,
    )
    return MemoryArenaBundle(
        task_id="fixture_chain_7",
        questions=questions,
        target_asins=TARGETS,
        budget_cents=10_000,
        split="train",
        source_row_id=7,
        provenance=provenance,
        sessions=sessions,
        category="fixture",
        answer_attributes=tuple(session.answer_attributes for session in sessions),
    )


def make_bundle() -> MemoryArenaBundle:
    return make_bundle_with_options((OPTIONS,) * 6)


def make_unique_bundle() -> MemoryArenaBundle:
    return make_bundle_with_options((OPTIONS,) + (FIVE_OPTIONS,) * 5)


class RecordingBackend:
    surface = "memoryarena_webshop_native_v1"

    def __init__(self) -> None:
        self.instructions: list[str] = []

    def open_session(self, session_token: str, instruction: str) -> NativePage:
        self.instructions.append(instruction)
        return NativePage(
            observation=f"Instruction: {instruction}",
            url=f"http://native/{session_token}",
            has_search_bar=True,
            clickables=(),
        )

    def close_session(self, session_token: str) -> None:
        del session_token

    def step(self, session_token: str, action: str) -> NativePage:
        raise AssertionError((session_token, action))

    def has_product(self, asin: str) -> bool:
        return asin in TARGETS

    def metadata(self):
        return {"surface": self.surface}


class PresentationRandomizationTests(unittest.TestCase):
    def test_reorders_only_complete_candidate_lines(self) -> None:
        question = make_question(1).rstrip("\n")
        source = split_candidate_block(question)

        rendered = reorder_candidate_options(question, (2, 0, 1))
        transformed = split_candidate_block(rendered)

        self.assertEqual(transformed.prefix, source.prefix)
        self.assertEqual(transformed.suffix, source.suffix)
        self.assertEqual(transformed.option_endings, source.option_endings)
        self.assertEqual(transformed.option_titles, (OPTIONS[2], OPTIONS[0], OPTIONS[1]))
        self.assertCountEqual(transformed.option_lines, source.option_lines)

    def test_candidate_order_variant_is_replayable_and_keeps_labels_frozen(self) -> None:
        bundle = make_bundle()

        first = build_presentation_variant(
            bundle,
            mode=PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1,
            base_seed=20260728,
            env_uid="env11",
            episode_counter=3,
        )
        replay = build_presentation_variant(
            bundle,
            mode=PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1,
            base_seed=20260728,
            env_uid="env11",
            episode_counter=3,
        )

        self.assertEqual(first, replay)
        self.assertEqual(bundle.target_asins, TARGETS)
        self.assertEqual(len(first.questions), 6)
        self.assertEqual(len(first.candidate_permutations), 6)
        self.assertTrue(
            any(
                question != source
                for question, source in zip(first.questions, bundle.questions)
            )
        )
        for source, rendered in zip(bundle.questions, first.questions):
            source_block = split_candidate_block(source)
            rendered_block = split_candidate_block(rendered)
            self.assertEqual(rendered_block.prefix, source_block.prefix)
            self.assertEqual(rendered_block.suffix, source_block.suffix)
            self.assertEqual(rendered_block.option_endings, source_block.option_endings)
            self.assertCountEqual(rendered_block.option_lines, source_block.option_lines)
        self.assertEqual(
            first.as_info()["label_contract"],
            "frozen_upstream_target_asins_unchanged",
        )

    def test_unique_v2_keeps_eight_full_variants_distinct(self) -> None:
        bundle = make_unique_bundle()
        variants = [
            build_presentation_variant(
                bundle,
                mode=PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2,
                base_seed=20260728,
                env_uid=f"env{index}",
                episode_counter=1,
                variant_index=index,
            )
            for index in range(8)
        ]

        full_variants = {
            variant.candidate_permutations for variant in variants
        }
        self.assertEqual(len(full_variants), 8)
        self.assertEqual(
            len({variant.candidate_permutations[0] for variant in variants}),
            6,
        )
        for session_index in range(1, 6):
            self.assertEqual(
                len(
                    {
                        variant.candidate_permutations[session_index]
                        for variant in variants
                    }
                ),
                8,
            )
        self.assertEqual(bundle.target_asins, TARGETS)
        self.assertEqual(
            variants[7].as_info()["variant_index"],
            7,
        )

    def test_unique_v2_first_120_variants_cover_later_sessions_without_replacement(self) -> None:
        rank_tuples = [
            unique_bundle_permutation_ranks(
                option_counts=(3, 5, 5, 5, 5, 5),
                variant_index=index,
            )
            for index in range(120)
        ]

        self.assertEqual(len(set(rank_tuples)), 120)
        self.assertEqual(len({ranks[0] for ranks in rank_tuples}), 6)
        for session_index in range(1, 6):
            self.assertEqual(
                len({ranks[session_index] for ranks in rank_tuples}),
                120,
            )

    def test_unique_v2_rejects_missing_index_or_wrong_option_pattern(self) -> None:
        with self.assertRaisesRegex(PresentationRandomizationError, "variant_index"):
            build_presentation_variant(
                make_unique_bundle(),
                mode=PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2,
                base_seed=1,
                env_uid="env0",
                episode_counter=1,
            )
        with self.assertRaisesRegex(PresentationRandomizationError, "option pattern"):
            build_presentation_variant(
                make_bundle(),
                mode=PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2,
                base_seed=1,
                env_uid="env0",
                episode_counter=1,
                variant_index=0,
            )

    def test_episode_identity_changes_and_original_mode_is_byte_exact(self) -> None:
        bundle = make_bundle()
        original = build_presentation_variant(
            bundle,
            mode=PRESENTATION_RANDOMIZATION_NONE,
            base_seed=9,
            env_uid="env0",
            episode_counter=1,
        )
        later = build_presentation_variant(
            bundle,
            mode=PRESENTATION_RANDOMIZATION_NONE,
            base_seed=9,
            env_uid="env0",
            episode_counter=2,
        )

        self.assertEqual(original.questions, bundle.questions)
        self.assertEqual(original.content_sha256, later.content_sha256)
        self.assertNotEqual(original.instance_sha256, later.instance_sha256)
        self.assertEqual(original.candidate_permutations, ((0, 1, 2),) * 6)

    def test_environment_records_randomized_variant_without_exposing_a_label(self) -> None:
        bundle = make_bundle()
        backend = RecordingBackend()
        env = MemoryArenaWebShopEnv(
            bundles=[bundle],
            backend=backend,
            env_uid="env5",
            presentation_randomization_mode=(
                PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1
            ),
            presentation_seed=20260728,
        )

        _, info = env.reset()

        self.assertEqual(backend.instructions[-1], env.presentation_variant.questions[0])
        self.assertEqual(info["source"], "memoryarena_derived_presentation")
        self.assertEqual(
            info["presentation_variant"]["mode"],
            PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1,
        )
        self.assertFalse(any("target" in key for key in info["presentation_variant"]))
        self.assertNotIn(TARGETS[0], str(info["presentation_variant"]))

    def test_environment_passes_explicit_unique_variant_index(self) -> None:
        bundle = make_unique_bundle()
        backend = RecordingBackend()
        env = MemoryArenaWebShopEnv(
            bundles=[bundle],
            backend=backend,
            env_uid="env7",
            presentation_randomization_mode=(
                PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2
            ),
            presentation_seed=20260728,
            presentation_variant_index=7,
        )

        _, info = env.reset()

        self.assertEqual(info["presentation_variant"]["variant_index"], 7)
        self.assertEqual(
            info["presentation_variant"]["mode"],
            PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2,
        )

    def test_rejects_non_permutation_or_non_terminal_candidate_rows(self) -> None:
        with self.assertRaisesRegex(PresentationRandomizationError, "each index"):
            reorder_candidate_options(make_question(1), (0, 0, 1))
        malformed = make_question(1) + "Non-option footer\n"
        with self.assertRaisesRegex(PresentationRandomizationError, "terminal"):
            split_candidate_block(malformed)


if __name__ == "__main__":
    unittest.main()
