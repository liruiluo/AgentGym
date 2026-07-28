from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .memoryarena_dataset import MemoryArenaBundle


PRESENTATION_RANDOMIZATION_NONE = "none"
PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1 = "candidate_order_v1"
PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2 = (
    "candidate_order_unique_v2"
)
PRESENTATION_RANDOMIZATION_MODES = (
    PRESENTATION_RANDOMIZATION_NONE,
    PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1,
    PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2,
)
PRESENTATION_VARIANT_SCHEMA = "memoryarena_presentation_variant_v1"
PRESENTATION_LABEL_CONTRACT = "frozen_upstream_target_asins_unchanged"
UNIQUE_V2_OPTION_COUNTS = (3, 5, 5, 5, 5, 5)
UNIQUE_V2_PERMUTATION_RADICES = tuple(
    math.factorial(option_count) for option_count in UNIQUE_V2_OPTION_COUNTS
)

_CANDIDATE_MARKER = "**Available Options:**"
_OPTION_PATTERN = re.compile(r"^\s*-\s+(?P<title>\S(?:.*\S)?)\s*$")


class PresentationRandomizationError(ValueError):
    """Raised when a presentation-only transform cannot be proven safe."""


@dataclass(frozen=True)
class CandidateBlock:
    prefix: str
    option_lines: tuple[str, ...]
    option_endings: tuple[str, ...]
    option_titles: tuple[str, ...]
    suffix: str


@dataclass(frozen=True)
class PresentationVariant:
    mode: str
    base_seed: int
    env_uid: str
    episode_counter: int
    variant_index: int | None
    task_id: str
    questions: tuple[str, ...]
    candidate_permutations: tuple[tuple[int, ...], ...]
    source_question_sha256s: tuple[str, ...]
    rendered_question_sha256s: tuple[str, ...]
    content_sha256: str
    instance_sha256: str

    def as_info(self) -> dict[str, Any]:
        return {
            "schema": PRESENTATION_VARIANT_SCHEMA,
            "mode": self.mode,
            "base_seed": self.base_seed,
            "env_uid": self.env_uid,
            "episode_counter": self.episode_counter,
            "variant_index": self.variant_index,
            "task_id": self.task_id,
            "candidate_permutations": [
                list(permutation) for permutation in self.candidate_permutations
            ],
            "source_question_sha256s": list(self.source_question_sha256s),
            "rendered_question_sha256s": list(self.rendered_question_sha256s),
            "content_sha256": self.content_sha256,
            "instance_sha256": self.instance_sha256,
            "label_contract": PRESENTATION_LABEL_CONTRACT,
        }


def build_presentation_variant(
    bundle: MemoryArenaBundle,
    *,
    mode: str,
    base_seed: int,
    env_uid: str,
    episode_counter: int,
    variant_index: int | None = None,
) -> PresentationVariant:
    if mode not in PRESENTATION_RANDOMIZATION_MODES:
        raise PresentationRandomizationError(
            f"Unsupported presentation randomization mode {mode!r}; "
            f"expected one of {PRESENTATION_RANDOMIZATION_MODES}."
        )
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise PresentationRandomizationError("base_seed must be an integer.")
    if not isinstance(env_uid, str) or not env_uid:
        raise PresentationRandomizationError("env_uid must be a non-empty string.")
    if isinstance(episode_counter, bool) or not isinstance(episode_counter, int):
        raise PresentationRandomizationError("episode_counter must be an integer.")
    if episode_counter < 1:
        raise PresentationRandomizationError("episode_counter must be positive.")
    if variant_index is not None and (
        isinstance(variant_index, bool)
        or not isinstance(variant_index, int)
        or variant_index < 0
    ):
        raise PresentationRandomizationError(
            "variant_index must be a non-negative integer or None."
        )
    if (
        mode == PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2
        and variant_index is None
    ):
        raise PresentationRandomizationError(
            "candidate_order_unique_v2 requires an explicit variant_index."
        )
    if len(bundle.questions) != len(bundle.sessions):
        raise PresentationRandomizationError(
            f"Bundle {bundle.task_id!r} has misaligned questions and sessions."
        )

    option_counts = tuple(
        len(session.candidate_options) for session in bundle.sessions
    )
    unique_ranks: tuple[int, ...] | None = None
    if mode == PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2:
        unique_ranks = unique_bundle_permutation_ranks(
            option_counts=option_counts,
            variant_index=variant_index,
        )

    questions: list[str] = []
    permutations: list[tuple[int, ...]] = []
    for session_index, (question, session) in enumerate(
        zip(bundle.questions, bundle.sessions)
    ):
        option_count = len(session.candidate_options)
        if option_count < 1:
            raise PresentationRandomizationError(
                f"Bundle {bundle.task_id!r} session {session_index + 1} has no options."
            )
        if mode == PRESENTATION_RANDOMIZATION_NONE:
            permutation = tuple(range(option_count))
            rendered = question
        else:
            block = split_candidate_block(question)
            if block.option_titles != session.candidate_options:
                raise PresentationRandomizationError(
                    f"Bundle {bundle.task_id!r} session {session_index + 1} "
                    "question options disagree with the parsed frozen session."
                )
            if mode == PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1:
                permutation = stable_candidate_permutation(
                    base_seed=base_seed,
                    env_uid=env_uid,
                    episode_counter=episode_counter,
                    task_id=bundle.task_id,
                    source_row_id=bundle.source_row_id,
                    session_index=session_index,
                    option_titles=block.option_titles,
                )
            else:
                if unique_ranks is None:
                    raise AssertionError("unique permutation ranks are unavailable")
                permutation = stable_unique_candidate_permutation(
                    base_seed=base_seed,
                    task_id=bundle.task_id,
                    source_row_id=bundle.source_row_id,
                    session_index=session_index,
                    option_titles=block.option_titles,
                    permutation_rank=unique_ranks[session_index],
                )
            rendered = reorder_candidate_options(question, permutation)
            rendered_block = split_candidate_block(rendered)
            if sorted(rendered_block.option_lines) != sorted(block.option_lines):
                raise PresentationRandomizationError(
                    f"Bundle {bundle.task_id!r} session {session_index + 1} "
                    "changed candidate content while reordering."
                )
            if rendered_block.prefix != block.prefix or rendered_block.suffix != block.suffix:
                raise PresentationRandomizationError(
                    f"Bundle {bundle.task_id!r} session {session_index + 1} "
                    "changed non-candidate text while reordering."
                )
            if rendered_block.option_endings != block.option_endings:
                raise PresentationRandomizationError(
                    f"Bundle {bundle.task_id!r} session {session_index + 1} "
                    "changed candidate row boundaries while reordering."
                )
        questions.append(rendered)
        permutations.append(permutation)

    source_hashes = tuple(_sha256_text(question) for question in bundle.questions)
    rendered_hashes = tuple(_sha256_text(question) for question in questions)
    content_payload = {
        "schema": PRESENTATION_VARIANT_SCHEMA,
        "mode": mode,
        "task_id": bundle.task_id,
        "source_row_id": bundle.source_row_id,
        "raw_dataset_sha256": bundle.provenance.raw_dataset_sha256,
        "candidate_permutations": [list(value) for value in permutations],
        "variant_index": variant_index,
        "source_question_sha256s": list(source_hashes),
        "rendered_question_sha256s": list(rendered_hashes),
        "label_contract": PRESENTATION_LABEL_CONTRACT,
    }
    content_sha256 = _sha256_json(content_payload)
    instance_sha256 = _sha256_json(
        {
            "content_sha256": content_sha256,
            "base_seed": base_seed,
            "env_uid": env_uid,
            "episode_counter": episode_counter,
            "variant_index": variant_index,
        }
    )
    return PresentationVariant(
        mode=mode,
        base_seed=base_seed,
        env_uid=env_uid,
        episode_counter=episode_counter,
        variant_index=variant_index,
        task_id=bundle.task_id,
        questions=tuple(questions),
        candidate_permutations=tuple(permutations),
        source_question_sha256s=source_hashes,
        rendered_question_sha256s=rendered_hashes,
        content_sha256=content_sha256,
        instance_sha256=instance_sha256,
    )


def stable_candidate_permutation(
    *,
    base_seed: int,
    env_uid: str,
    episode_counter: int,
    task_id: str,
    source_row_id: int,
    session_index: int,
    option_titles: Sequence[str],
) -> tuple[int, ...]:
    """Return a cross-process deterministic permutation without global RNG state."""

    scores: list[tuple[bytes, int]] = []
    for option_index, option_title in enumerate(option_titles):
        payload = {
            "schema": PRESENTATION_VARIANT_SCHEMA,
            "base_seed": base_seed,
            "env_uid": env_uid,
            "episode_counter": episode_counter,
            "task_id": task_id,
            "source_row_id": source_row_id,
            "session_index": session_index,
            "option_index": option_index,
            "option_title": option_title,
        }
        score = hashlib.sha256(_canonical_json(payload)).digest()
        scores.append((score, option_index))
    return tuple(index for _, index in sorted(scores))


def unique_bundle_permutation_ranks(
    *,
    option_counts: Sequence[int],
    variant_index: int | None,
) -> tuple[int, ...]:
    """Map a variant index to a unique six-session permutation tuple.

    MemoryArena has one 3-option session followed by five 5-option sessions.
    The triangular mixed-radix mapping is bijective over the complete bundle
    presentation space. For the first 120 variants, session 1 covers its six
    orders evenly while every later session uses a distinct order.
    """

    normalized_counts = tuple(option_counts)
    if normalized_counts != UNIQUE_V2_OPTION_COUNTS:
        raise PresentationRandomizationError(
            "candidate_order_unique_v2 requires the audited MemoryArena option "
            f"pattern {UNIQUE_V2_OPTION_COUNTS}; observed {normalized_counts}."
        )
    if (
        isinstance(variant_index, bool)
        or not isinstance(variant_index, int)
        or variant_index < 0
    ):
        raise PresentationRandomizationError(
            "variant_index must be a non-negative integer."
        )

    space_size = math.prod(UNIQUE_V2_PERMUTATION_RADICES)
    if variant_index >= space_size:
        raise PresentationRandomizationError(
            f"variant_index {variant_index} exceeds the unique bundle space "
            f"of {space_size}."
        )

    remaining = variant_index
    raw_ranks: list[int] = []
    for radix in UNIQUE_V2_PERMUTATION_RADICES:
        raw_ranks.append(remaining % radix)
        remaining //= radix
    if remaining:
        raise AssertionError("mixed-radix decomposition left a remainder")

    first_rank = raw_ranks[0]
    second_rank = (
        raw_ranks[1]
        + (UNIQUE_V2_PERMUTATION_RADICES[1] // UNIQUE_V2_PERMUTATION_RADICES[0])
        * first_rank
    ) % UNIQUE_V2_PERMUTATION_RADICES[1]
    ranks = [first_rank, second_rank]
    for raw_rank, multiplier, radix in zip(
        raw_ranks[2:],
        (1, 7, 11, 13),
        UNIQUE_V2_PERMUTATION_RADICES[2:],
    ):
        ranks.append((raw_rank + multiplier * second_rank) % radix)
    return tuple(ranks)


def stable_unique_candidate_permutation(
    *,
    base_seed: int,
    task_id: str,
    source_row_id: int,
    session_index: int,
    option_titles: Sequence[str],
    permutation_rank: int,
) -> tuple[int, ...]:
    """Select one seeded permutation by rank without replacement."""

    permutations = tuple(itertools.permutations(range(len(option_titles))))
    if (
        isinstance(permutation_rank, bool)
        or not isinstance(permutation_rank, int)
        or permutation_rank < 0
        or permutation_rank >= len(permutations)
    ):
        raise PresentationRandomizationError(
            f"permutation_rank must be in [0, {len(permutations)}); "
            f"observed {permutation_rank!r}."
        )
    scored: list[tuple[bytes, tuple[int, ...]]] = []
    for permutation in permutations:
        payload = {
            "schema": "memoryarena_candidate_order_unique_v2",
            "base_seed": base_seed,
            "task_id": task_id,
            "source_row_id": source_row_id,
            "session_index": session_index,
            "option_titles": list(option_titles),
            "permutation": list(permutation),
        }
        score = hashlib.sha256(_canonical_json(payload)).digest()
        scored.append((score, permutation))
    return sorted(scored)[permutation_rank][1]


def reorder_candidate_options(question: str, permutation: Sequence[int]) -> str:
    block = split_candidate_block(question)
    normalized = tuple(permutation)
    expected = tuple(range(len(block.option_lines)))
    if sorted(normalized) != list(expected):
        raise PresentationRandomizationError(
            f"Candidate permutation must contain each index exactly once; "
            f"expected {expected}, observed {normalized}."
        )
    option_text = "".join(
        block.option_lines[source_index] + block.option_endings[target_index]
        for target_index, source_index in enumerate(normalized)
    )
    return block.prefix + option_text + block.suffix


def split_candidate_block(question: str) -> CandidateBlock:
    if not isinstance(question, str) or not question:
        raise PresentationRandomizationError("question must be a non-empty string.")
    marker_count = question.count(_CANDIDATE_MARKER)
    if marker_count != 1:
        raise PresentationRandomizationError(
            f"question must contain exactly one {_CANDIDATE_MARKER!r}; "
            f"observed {marker_count}."
        )
    marker_end = question.index(_CANDIDATE_MARKER) + len(_CANDIDATE_MARKER)
    prefix = question[:marker_end]
    tail = question[marker_end:]
    leading: list[str] = []
    option_lines: list[str] = []
    option_endings: list[str] = []
    option_titles: list[str] = []
    trailing: list[str] = []
    for line in tail.splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        line_ending = line[len(line_body) :]
        match = _OPTION_PATTERN.fullmatch(line_body)
        if match and not trailing:
            option_lines.append(line_body)
            option_endings.append(line_ending)
            option_titles.append(match.group("title"))
        elif not option_lines:
            if line.strip():
                raise PresentationRandomizationError(
                    "candidate marker must be followed only by blank lines and option rows."
                )
            leading.append(line)
        else:
            if line.strip():
                raise PresentationRandomizationError(
                    "candidate option rows must be contiguous and terminal."
                )
            trailing.append(line)
    if not option_lines:
        raise PresentationRandomizationError("question has no candidate option rows.")
    if len(option_titles) != len(set(option_titles)):
        raise PresentationRandomizationError("question has duplicate candidate option text.")
    return CandidateBlock(
        prefix=prefix + "".join(leading),
        option_lines=tuple(option_lines),
        option_endings=tuple(option_endings),
        option_titles=tuple(option_titles),
        suffix="".join(trailing),
    )


def presentation_config_manifest(*, mode: str, base_seed: int) -> dict[str, Any]:
    if mode not in PRESENTATION_RANDOMIZATION_MODES:
        raise PresentationRandomizationError(
            f"Unsupported presentation randomization mode {mode!r}."
        )
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise PresentationRandomizationError("base_seed must be an integer.")
    return {
        "schema": PRESENTATION_VARIANT_SCHEMA,
        "mode": mode,
        "base_seed": base_seed,
        "label_contract": PRESENTATION_LABEL_CONTRACT,
        "changes_target_asins": False,
        "changes_prices": False,
        "changes_reward": False,
        "changes_candidate_text": False,
        "variant_schedule": (
            "without_replacement_bundle_v2"
            if mode == PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2
            else "independent_hash_v1"
            if mode == PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1
            else "identity"
        ),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


__all__ = [
    "PRESENTATION_LABEL_CONTRACT",
    "PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_V1",
    "PRESENTATION_RANDOMIZATION_CANDIDATE_ORDER_UNIQUE_V2",
    "PRESENTATION_RANDOMIZATION_MODES",
    "PRESENTATION_RANDOMIZATION_NONE",
    "PRESENTATION_VARIANT_SCHEMA",
    "UNIQUE_V2_OPTION_COUNTS",
    "UNIQUE_V2_PERMUTATION_RADICES",
    "CandidateBlock",
    "PresentationRandomizationError",
    "PresentationVariant",
    "build_presentation_variant",
    "presentation_config_manifest",
    "reorder_candidate_options",
    "split_candidate_block",
    "stable_candidate_permutation",
    "stable_unique_candidate_permutation",
    "unique_bundle_permutation_ranks",
]
