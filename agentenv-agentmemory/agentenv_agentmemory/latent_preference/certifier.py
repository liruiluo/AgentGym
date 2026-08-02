from __future__ import annotations

import collections
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..native_webshop_backend import NativeWebShopBackend
from .schema import (
    SPLITS,
    CertifiedPreferenceProduct,
    LatentPreferenceDataError,
    PreferenceProductPool,
    PreferenceRecipe,
    canonical_sha256,
    normalize_native_title,
    preference_classification_payload,
    require_sha256,
)


CANDIDATE_SCHEMA = "agentmemory_latent_preference_rule_candidate_v2"
CERTIFIER_VERSION = "native_latent_preference_rules_v4"
CERTIFICATION_AUDIT_SCHEMA = "agentmemory_latent_preference_pool_certification_v4"

_ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
_QUERY_WORD_RE = re.compile(r"\b[\w][\w'&+./-]*\b", flags=re.UNICODE)
_QUERY_SEPARATOR_RE = re.compile(r"\s+(?:[-\u2013\u2014|])\s+|[,;:]")
_QUERY_EDGE_CHARS = " \t,.;:|/\\-\u2013\u2014(){}"
_QUERY_UNSAFE_CHARS = "[]\r\n"

_ROW_KEYS = {
    "schema",
    "asin",
    "axis",
    "attribute_value",
    "category_id",
    "classification_sha256",
    "normalized_title",
    "product_category",
    "title",
    "title_evidence",
}
_SOURCE_CLASSIFICATION_KEYS = (
    "category_id",
    "axis",
    "attribute_value",
    "asin",
    "title",
    "product_category",
    "title_evidence",
)


PREFERENCE_RECIPES = (
    PreferenceRecipe(
        recipe_id="color.black_gray",
        axis="color",
        axis_display_name="color",
        values=("black", "gray"),
        value_display_names=("black", "gray"),
        categories=("phone_case", "pillowcase", "watch_band", "window_curtain"),
        category_display_names=(
            "phone case",
            "pillowcase",
            "watch band",
            "window curtain",
        ),
    ),
    PreferenceRecipe(
        recipe_id="dietary_profile.gluten_free_organic",
        axis="dietary_profile",
        axis_display_name="dietary preference",
        values=("gluten_free", "organic"),
        value_display_names=("gluten-free", "organic"),
        categories=("cookies", "drink_mix", "granola", "nutrition_bar"),
        category_display_names=(
            "cookies",
            "drink mix",
            "granola",
            "nutrition bar",
        ),
    ),
    PreferenceRecipe(
        recipe_id="flavor.chocolate_vanilla",
        axis="flavor",
        axis_display_name="flavor",
        values=("chocolate", "vanilla"),
        value_display_names=("chocolate", "vanilla"),
        categories=("cake_mix", "cookies", "coffee_creamer", "nutrition_bar"),
        category_display_names=(
            "cake mix",
            "cookies",
            "coffee creamer",
            "nutrition bar",
        ),
    ),
    PreferenceRecipe(
        recipe_id="pattern.floral_geometric",
        axis="pattern",
        axis_display_name="pattern",
        values=("floral", "geometric"),
        value_display_names=("floral", "geometric"),
        categories=("area_rug", "comforter", "pillowcase", "window_curtain"),
        category_display_names=(
            "area rug",
            "comforter",
            "pillowcase",
            "window curtain",
        ),
    ),
)


_ATTRIBUTE_GUARD_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "color": {
        "black": (r"\bblack\b",),
        "white": (r"\bwhite\b",),
        "gray": (r"\bgr[ae]y\b",),
        "red": (r"\bred\b",),
        "green": (r"\bgreen\b",),
        "pink": (r"\bpink\b",),
        "purple": (r"\bpurple\b",),
        "beige": (r"\bbeige\b",),
        "blue": (r"\bblue\b",),
        "navy": (r"\bnavy\b",),
        "teal": (r"\bteal\b",),
        "aqua": (r"\baqua\b",),
        "turquoise": (r"\bturquoise\b",),
        "yellow": (r"\byellow\b",),
        "gold": (r"\bgold(?:en)?\b",),
        "silver": (r"\bsilver\b",),
        "brown": (r"\bbrown\b",),
        "orange": (r"\borange\b",),
        "ivory": (r"\bivory\b",),
        "cream": (r"\bcream\b",),
        "tan": (r"\btan\b",),
        "khaki": (r"\bkhaki\b",),
        "burgundy": (r"\bburgundy\b",),
        "maroon": (r"\bmaroon\b",),
        "coral": (r"\bcoral\b",),
        "taupe": (r"\btaupe\b",),
        "lime": (r"\blime\b",),
        "lavender": (r"\blavender\b",),
        "olive": (r"\bolive\b",),
        "mint": (r"\bmint\b",),
        "cyan": (r"\bcyan\b",),
        "magenta": (r"\bmagenta\b",),
        "peach": (r"\bpeach\b",),
        "rose": (r"\brose\b",),
        "wine": (r"\bwine\b",),
        "charcoal": (r"\bcharcoal\b",),
        "bronze": (r"\bbronze\b",),
        "copper": (r"\bcopper\b",),
        "clear": (r"\bclear\b",),
        "transparent": (r"\btransparent\b",),
        "violet": (r"\bviolet\b",),
        "lilac": (r"\blilac\b",),
        "mauve": (r"\bmauve\b",),
        "fuchsia": (r"\bfuchsia\b",),
        "multicolor": (
            r"\bmulti[ -]?colou?rs?\b",
            r"\brainbow\b",
            r"\bassorted colou?rs?\b",
            r"\bcolor ?block(?:ed)?\b",
            r"\bcolou?rful\b",
        ),
    },
    "dietary_profile": {
        "gluten_free": (r"\bgluten[ -]?free\b",),
        "organic": (r"\borganic\b",),
        "sugar_free": (r"\bsugar[ -]?free\b",),
        "vegan": (r"\bvegan\b",),
        "keto": (r"\bketo(?:genic)?\b",),
        "dairy_free": (r"\bdairy[ -]?free\b",),
        "non_gmo": (r"\bnon[ -]?gmo\b",),
        "kosher": (r"\bkosher\b",),
        "paleo": (r"\bpaleo\b",),
        "low_carb": (r"\blow[ -]?carb\b",),
    },
    "flavor": {
        "chocolate": (r"\bchocolate\b", r"\bcocoa\b"),
        "vanilla": (r"\bvanilla\b",),
        "strawberry": (r"\bstrawberry\b",),
        "peanut_butter": (r"\bpeanut butter\b",),
        "caramel": (r"\bcaramel\b",),
        "coffee": (r"\bcoffee\b",),
        "mocha": (r"\bmocha\b",),
        "cinnamon": (r"\bcinnamon\b",),
        "lemon": (r"\blemon\b",),
        "lime": (r"\blime\b",),
        "orange": (r"\borange\b",),
        "blueberry": (r"\bblueberr(?:y|ies)\b",),
        "raspberry": (r"\braspberr(?:y|ies)\b",),
        "coconut": (r"\bcoconut\b",),
        "mint": (r"\bmint\b",),
        "banana": (r"\bbanana\b",),
        "honey": (r"\bhoney\b",),
        "maple": (r"\bmaple\b",),
        "unflavored": (r"\bunflavou?red\b",),
        "variety": (r"\bvariety(?: pack)?\b", r"\bassorted flavors?\b"),
    },
    "pattern": {
        "solid": (r"\bsolid(?: colou?r)?\b",),
        "striped": (r"\bstrip(?:e|ed|es)\b",),
        "floral": (r"\bfloral\b", r"\bflower print\b"),
        "plaid": (r"\bplaid\b",),
        "geometric": (r"\bgeometric\b",),
        "checkered": (r"\bcheck(?:er|ered)\b", r"\bgingham\b"),
        "polka_dot": (r"\bpolka dots?\b",),
        "paisley": (r"\bpaisley\b",),
        "abstract": (r"\babstract\b",),
        "animal_print": (r"\b(?:leopard|zebra|animal) print\b",),
        "chevron": (r"\bchevron\b",),
        "damask": (r"\bdamask\b",),
        "patchwork": (r"\bpatchwork\b",),
        "tie_dye": (r"\btie[ -]?dye\b",),
        "camouflage": (r"\b(?:camo|camouflage)\b",),
        "ombre": (r"\bombr[e\u00e9]\b",),
    },
}

# Most preference axes are exclusive labels: a black/lime title is not a
# clean black example. Dietary labels are different: gluten-free, vegan and
# non-GMO routinely coexist, so only the two values contrasted by the recipe
# are mutually disqualifying.
_ATTRIBUTE_GUARD_SCOPES: dict[str, tuple[str, ...]] = {
    "color": tuple(_ATTRIBUTE_GUARD_PATTERNS["color"]),
    "dietary_profile": ("gluten_free", "organic"),
    "flavor": tuple(_ATTRIBUTE_GUARD_PATTERNS["flavor"]),
    "pattern": tuple(_ATTRIBUTE_GUARD_PATTERNS["pattern"]),
}

_CATEGORY_TITLE_GUARD_PATTERNS: dict[str, tuple[str, ...]] = {
    "area_rug": (r"\brugs?\b",),
    "cake_mix": (
        r"\b(?:cakes?|cupcakes?)\b.*\bmix(?:es)?\b",
        r"\bmix(?:es)?\b.*\b(?:cakes?|cupcakes?)\b",
    ),
    "comforter": (r"\bcomforters?\b",),
    "coffee_creamer": (r"\bcreamers?\b",),
    "cookies": (r"\b(?:cookies?|wafers?|biscuits?)\b",),
    "drink_mix": (
        r"\b(?:powder(?:ed)?|mix(?:es)?)\b",
        r"\binstant\b.*\bdrink\b",
    ),
    "granola": (r"\bgranola\b",),
    "nutrition_bar": (
        r"\b(?:protein|energy|nutrition|meal|snack|fiber|granola)\b.*\bbars?\b",
        r"\bbars?\b.*\b(?:protein|energy|nutrition|meal|snack|fiber|granola)\b",
    ),
    "phone_case": (r"\b(?:case|cover)\b",),
    "pillowcase": (r"\bpillow(?:cases?| covers?| shams?)\b",),
    "watch_band": (
        r"\b(?:watch|smartwatch)\b.*\b(?:band|strap|wristband)\b",
        r"\b(?:band|strap|wristband)\b.*\b(?:watch|smartwatch)\b",
    ),
    "window_curtain": (
        r"\b(?:curtains?|drapes?|panels?|valance)\b",
    ),
}

_CATEGORY_TITLE_EXCLUSION_PATTERNS: dict[str, tuple[str, ...]] = {
    "cookies": (
        r"\bbars?\b",
        r"\b(?:baking|cookie|dough)\s+mix(?:es)?\b",
        r"\bmix(?:es)?\b.*\b(?:cookies?|dough)\b",
        r"\bcookie\s+dough\b",
        r"\bbrittle\b",
    ),
    "drink_mix": (
        r"\bready[ -]?to[ -]?drink\b",
        r"\bsyrups?\b",
        r"\bfl\.?\s*oz\b",
        r"\bbottles?\b",
        r"\bcans?\b",
        r"\bliquids?\b",
        r"\bpulp\b",
        r"\bliquors?\b",
        r"\bspirits?\b",
    ),
    "granola": (
        r"\bbars?\b",
        r"\bbites?\b",
        r"\bminis?\b",
        r"\bbutter\b",
        r"\balternatives?\b",
        r"\bnot\s+granola\b",
    ),
}

_ATTRIBUTE_CONTEXT_RULES: dict[str, dict[str, object]] = {
    "dietary_profile.organic": {
        "attribute_pattern": r"\borganic\b",
        "direction": "attribute_before_category",
        "max_intervening_words": 5,
        "category_patterns": {
            "cookies": r"\b(?:cookies?|biscuits?|wafers?)\b",
            "drink_mix": r"\b(?:powder(?:ed)?|mix(?:es)?)\b",
            "granola": r"\bgranola\b",
            "nutrition_bar": r"\bbars?\b",
        },
    },
}


PREFERENCE_RULES_SHA256 = canonical_sha256(
    {
        "certifier_version": CERTIFIER_VERSION,
        "recipes": [recipe.as_dict() for recipe in PREFERENCE_RECIPES],
        "attribute_guard_patterns": _ATTRIBUTE_GUARD_PATTERNS,
        "attribute_guard_scopes": _ATTRIBUTE_GUARD_SCOPES,
        "category_title_guard_patterns": _CATEGORY_TITLE_GUARD_PATTERNS,
        "category_title_exclusion_patterns": (
            _CATEGORY_TITLE_EXCLUSION_PATTERNS
        ),
        "attribute_context_rules": _ATTRIBUTE_CONTEXT_RULES,
        "title_normalization": "unicode_nfkc_whitespace_casefold_v1",
    }
)


class NativePreferencePoolCertificationError(LatentPreferenceDataError):
    def __init__(self, message: str, *, audit: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class NativePreferenceCertificationConfig:
    pool_id: str = "memoryarena_latent_preference_mainline_v4"
    recipe_ids: tuple[str, ...] = tuple(
        recipe.recipe_id for recipe in PREFERENCE_RECIPES
    )
    products_per_cell: int = 2
    candidate_cap_per_cell: int = 72
    max_search_rank: int = 10
    min_title_chars: int = 8
    max_title_chars: int = 240
    min_search_query_chars: int = 8
    max_search_query_chars: int = 160

    def __post_init__(self) -> None:
        known_order = tuple(
            recipe.recipe_id
            for recipe in PREFERENCE_RECIPES
            if recipe.recipe_id in set(self.recipe_ids)
        )
        if not self.recipe_ids or self.recipe_ids != known_order:
            raise LatentPreferenceDataError(
                "recipe_ids must be a non-empty canonical ordered subset."
            )
        if self.products_per_cell < 2:
            raise LatentPreferenceDataError("products_per_cell must be at least two.")
        required = self.products_per_cell * len(SPLITS)
        if self.candidate_cap_per_cell < required:
            raise LatentPreferenceDataError(
                "candidate_cap_per_cell cannot be smaller than the split-balanced need."
            )
        if not 1 <= self.max_search_rank <= 10:
            raise LatentPreferenceDataError(
                "max_search_rank must stay in the native first-page window."
            )
        if not 1 <= self.min_title_chars <= self.max_title_chars:
            raise LatentPreferenceDataError("invalid native title length bounds.")
        if not 1 <= self.min_search_query_chars <= self.max_search_query_chars:
            raise LatentPreferenceDataError("invalid search-query length bounds.")
        if self.max_search_query_chars > self.max_title_chars:
            raise LatentPreferenceDataError(
                "max_search_query_chars cannot exceed max_title_chars."
            )

    @property
    def recipes(self) -> tuple[PreferenceRecipe, ...]:
        selected = set(self.recipe_ids)
        return tuple(
            recipe for recipe in PREFERENCE_RECIPES if recipe.recipe_id in selected
        )


@dataclass(frozen=True, order=True)
class _Slot:
    axis: str
    category_id: str
    attribute_value: str
    split: str
    ordinal: int

    @property
    def base_cell(self) -> tuple[str, str, str]:
        return (self.axis, self.category_id, self.attribute_value)

    @property
    def name(self) -> str:
        return "/".join(
            (
                self.axis,
                self.category_id,
                self.attribute_value,
                self.split,
                str(self.ordinal),
            )
        )


@dataclass(frozen=True)
class _Candidate:
    asin: str
    title: str
    normalized_title: str
    product_category: str
    category_title_evidence: tuple[str, ...]
    category_id: str
    axis: str
    attribute_value: str
    title_evidence: tuple[str, ...]
    guard_matches: tuple[str, ...]
    source_candidate_sha256: str
    classification_sha256: str
    selection_sha256: str
    catalog_title_match_count: int | None = None

    @property
    def base_cell(self) -> tuple[str, str, str]:
        return (self.axis, self.category_id, self.attribute_value)


@dataclass(frozen=True)
class _NativeEvidence:
    price_cents: int
    search_query: str
    search_rank: int
    catalog_record_sha256: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def certify_native_preference_product_pool(
    backend: NativeWebShopBackend,
    *,
    candidate_artifact: str | Path,
    expected_candidate_artifact_sha256: str,
    catalog_sha256: str,
    attributes_sha256: str,
    lucene_index_sha256: str,
    config: NativePreferenceCertificationConfig | None = None,
) -> tuple[PreferenceProductPool, dict[str, Any]]:
    """Build a split-balanced real-product pool without human or LLM review."""

    config = config or NativePreferenceCertificationConfig()
    require_sha256(catalog_sha256, field="catalog_sha256")
    require_sha256(attributes_sha256, field="attributes_sha256")
    require_sha256(lucene_index_sha256, field="lucene_index_sha256")
    require_sha256(
        expected_candidate_artifact_sha256,
        field="expected_candidate_artifact_sha256",
    )
    candidate_path = Path(candidate_artifact).expanduser().resolve()
    if not candidate_path.is_file():
        raise LatentPreferenceDataError(
            f"candidate artifact is not a file: {candidate_path}"
        )
    observed_candidate_sha256 = file_sha256(candidate_path)
    if observed_candidate_sha256 != expected_candidate_artifact_sha256:
        raise LatentPreferenceDataError(
            "candidate artifact SHA256 mismatch: "
            f"expected {expected_candidate_artifact_sha256}, "
            f"observed {observed_candidate_sha256}."
        )

    metadata = backend.metadata()
    price_table_sha256 = metadata.get("price_table_sha256")
    require_sha256(price_table_sha256, field="native price_table_sha256")
    candidates_by_cell, parse_counts = _load_candidates(
        candidate_path,
        config=config,
    )
    candidates_by_cell, catalog_counts = _certify_catalog_identity(
        backend,
        candidates_by_cell=candidates_by_cell,
        config=config,
    )
    slots = _expected_slots(config)
    source_manifest = {
        "certifier_version": CERTIFIER_VERSION,
        "rules_sha256": PREFERENCE_RULES_SHA256,
        "config": _config_payload(config),
        "candidate_artifact_sha256": observed_candidate_sha256,
        "catalog_sha256": catalog_sha256,
        "attributes_sha256": attributes_sha256,
        "price_table_sha256": price_table_sha256,
        "lucene_index_sha256": lucene_index_sha256,
        "backend": _stable_backend_metadata(metadata),
    }
    source_manifest_sha256 = canonical_sha256(source_manifest)

    blocked_edges: set[tuple[str, tuple[str, str, str]]] = set()
    accepted_edges: dict[
        tuple[str, tuple[str, str, str]], _NativeEvidence
    ] = {}
    probe_details: list[dict[str, Any]] = []
    rejection_counts: collections.Counter[str] = collections.Counter()
    matching_rebuilds = 0
    final_matching: dict[_Slot, _Candidate] | None = None

    while True:
        matching_rebuilds += 1
        matching = _deterministic_unique_asin_matching(
            slots,
            candidates_by_cell,
            blocked_edges=blocked_edges,
        )
        if matching is None:
            audit = _build_audit(
                status="failed",
                pool=None,
                config=config,
                metadata=metadata,
                source_manifest_sha256=source_manifest_sha256,
                candidate_artifact_sha256=observed_candidate_sha256,
                catalog_sha256=catalog_sha256,
                attributes_sha256=attributes_sha256,
                price_table_sha256=price_table_sha256,
                lucene_index_sha256=lucene_index_sha256,
                candidates_by_cell=candidates_by_cell,
                parse_counts=parse_counts,
                catalog_counts=catalog_counts,
                matching_rebuilds=matching_rebuilds,
                blocked_edges=blocked_edges,
                rejection_counts=rejection_counts,
                probe_details=probe_details,
                final_matching=None,
            )
            raise NativePreferencePoolCertificationError(
                "no global ASIN-unique matching remains after fail-closed "
                "candidate certification.",
                audit=audit,
            )

        pending = next(
            (
                (slot, candidate)
                for slot, candidate in sorted(matching.items())
                if (candidate.asin, candidate.base_cell) not in accepted_edges
            ),
            None,
        )
        if pending is None:
            final_matching = matching
            break

        slot, candidate = pending
        edge = (candidate.asin, candidate.base_cell)
        evidence, detail = _audit_native_candidate(
            backend,
            candidate=candidate,
            probe_index=len(probe_details),
            config=config,
        )
        detail["assigned_slot"] = slot.name
        probe_details.append(detail)
        if evidence is None:
            blocked_edges.add(edge)
            rejection_counts[str(detail["rejection_reason"])] += 1
            continue
        accepted_edges[edge] = evidence

    assert final_matching is not None
    recipe_by_axis = {recipe.axis: recipe for recipe in config.recipes}
    products: list[CertifiedPreferenceProduct] = []
    for slot, candidate in sorted(final_matching.items()):
        evidence = accepted_edges[(candidate.asin, candidate.base_cell)]
        recipe = recipe_by_axis[slot.axis]
        products.append(
            CertifiedPreferenceProduct(
                asin=candidate.asin,
                title=candidate.title,
                native_title_normalized=candidate.normalized_title,
                price_cents=evidence.price_cents,
                product_category=candidate.product_category,
                category_title_evidence=candidate.category_title_evidence,
                category_id=slot.category_id,
                category_display_name=recipe.category_display_name(slot.category_id),
                axis=slot.axis,
                attribute_value=slot.attribute_value,
                attribute_display_name=recipe.value_display_name(
                    slot.attribute_value
                ),
                split=slot.split,
                search_query=evidence.search_query,
                search_rank=evidence.search_rank,
                catalog_record_sha256=evidence.catalog_record_sha256,
                title_evidence=candidate.title_evidence,
                guard_matches=candidate.guard_matches,
                classification_sha256=candidate.classification_sha256,
                source_candidate_sha256=candidate.source_candidate_sha256,
                native_title_catalog_match_count=(
                    candidate.catalog_title_match_count or 0
                ),
                native_title_globally_unique=True,
                native_search_verified=True,
                native_open_verified=True,
                native_purchase_verified=True,
            )
        )
    products.sort(
        key=lambda product: (
            product.axis,
            product.category_id,
            product.attribute_value,
            product.split,
            product.asin,
        )
    )
    pool = PreferenceProductPool(
        pool_id=config.pool_id,
        certifier_version=CERTIFIER_VERSION,
        products_per_cell=config.products_per_cell,
        recipes=config.recipes,
        products=tuple(products),
        catalog_sha256=catalog_sha256,
        attributes_sha256=attributes_sha256,
        price_table_sha256=price_table_sha256,
        lucene_index_sha256=lucene_index_sha256,
        candidate_artifact_sha256=observed_candidate_sha256,
        rules_sha256=PREFERENCE_RULES_SHA256,
        source_manifest_sha256=source_manifest_sha256,
    )
    audit = _build_audit(
        status="certified",
        pool=pool,
        config=config,
        metadata=metadata,
        source_manifest_sha256=source_manifest_sha256,
        candidate_artifact_sha256=observed_candidate_sha256,
        catalog_sha256=catalog_sha256,
        attributes_sha256=attributes_sha256,
        price_table_sha256=price_table_sha256,
        lucene_index_sha256=lucene_index_sha256,
        candidates_by_cell=candidates_by_cell,
        parse_counts=parse_counts,
        catalog_counts=catalog_counts,
        matching_rebuilds=matching_rebuilds,
        blocked_edges=blocked_edges,
        rejection_counts=rejection_counts,
        probe_details=probe_details,
        final_matching=final_matching,
    )
    return pool, audit


def _load_candidates(
    path: Path,
    *,
    config: NativePreferenceCertificationConfig,
) -> tuple[
    dict[tuple[str, str, str], tuple[_Candidate, ...]],
    dict[str, int],
]:
    required_cells = set(_expected_base_cells(config))
    candidates_by_edge: dict[
        tuple[str, tuple[str, str, str]], _Candidate
    ] = {}
    counts: collections.Counter[str] = collections.Counter()
    seen_rows: set[str] = set()
    asin_axis_assignment: dict[tuple[str, str], tuple[str, str]] = {}
    asin_identity: dict[str, tuple[str, str, str]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise LatentPreferenceDataError(
                    f"candidate artifact contains blank line {line_number}."
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LatentPreferenceDataError(
                    f"candidate line {line_number} is invalid JSON."
                ) from exc
            if not isinstance(parsed, Mapping):
                raise LatentPreferenceDataError(
                    f"candidate line {line_number} is not an object."
                )
            row = dict(parsed)
            _validate_source_row(row, line_number=line_number)
            row_sha256 = canonical_sha256(row)
            if row_sha256 in seen_rows:
                raise LatentPreferenceDataError(
                    f"candidate artifact repeats line semantics at line {line_number}."
                )
            seen_rows.add(row_sha256)
            counts["candidate_rows"] += 1

            asin = str(row["asin"])
            identity = (
                str(row["title"]),
                str(row["normalized_title"]),
                str(row["product_category"]),
            )
            prior_identity = asin_identity.setdefault(asin, identity)
            if prior_identity != identity:
                raise LatentPreferenceDataError(
                    f"candidate ASIN {asin} has conflicting source identities."
                )
            title = str(row["title"])
            if not config.min_title_chars <= len(title) <= config.max_title_chars:
                counts["rejected_title_length"] += 1
                continue
            if any(char in title for char in "\r\n"):
                counts["rejected_multiline_title"] += 1
                continue
            if asin.casefold() in normalize_native_title(title):
                counts["rejected_title_contains_internal_asin"] += 1
                continue

            axis = str(row["axis"])
            value = str(row["attribute_value"])
            category = str(row["category_id"])
            prior_assignment = asin_axis_assignment.setdefault(
                (asin, axis), (category, value)
            )
            if prior_assignment != (category, value):
                raise LatentPreferenceDataError(
                    f"candidate ASIN {asin} has conflicting {axis} assignments: "
                    f"{prior_assignment} vs {(category, value)}."
                )
            base_cell = (axis, category, value)
            if base_cell not in required_cells:
                continue
            evidence = tuple(str(value) for value in row["title_evidence"])
            category_title_evidence = _category_title_evidence(
                category,
                title,
            )
            if not category_title_evidence:
                counts["rejected_missing_category_title_evidence"] += 1
                continue
            if _category_title_exclusions(category, title):
                counts["rejected_category_title_exclusion"] += 1
                continue
            guard_matches = _guard_matches(axis, str(row["title"]))
            if guard_matches != (value,):
                reason = (
                    "rejected_broad_guard_no_match"
                    if not guard_matches
                    else "rejected_broad_guard_multiple_or_conflicting_values"
                )
                counts[reason] += 1
                continue
            if not _attribute_context_is_valid(
                axis=axis,
                attribute_value=value,
                category_id=category,
                title=title,
            ):
                counts["rejected_attribute_context"] += 1
                continue
            classification_sha256 = canonical_sha256(
                preference_classification_payload(
                    asin=asin,
                    title=str(row["title"]),
                    product_category=str(row["product_category"]),
                    category_title_evidence=category_title_evidence,
                    category_id=category,
                    axis=axis,
                    attribute_value=value,
                    title_evidence=evidence,
                    guard_matches=guard_matches,
                    source_candidate_sha256=row_sha256,
                )
            )
            candidate = _Candidate(
                asin=asin,
                title=str(row["title"]),
                normalized_title=str(row["normalized_title"]),
                product_category=str(row["product_category"]),
                category_title_evidence=category_title_evidence,
                category_id=category,
                axis=axis,
                attribute_value=value,
                title_evidence=evidence,
                guard_matches=guard_matches,
                source_candidate_sha256=row_sha256,
                classification_sha256=classification_sha256,
                selection_sha256=canonical_sha256(
                    {
                        "certifier_version": CERTIFIER_VERSION,
                        "pool_id": config.pool_id,
                        "rules_sha256": PREFERENCE_RULES_SHA256,
                        "base_cell": list(base_cell),
                        "asin": asin,
                        "source_candidate_sha256": row_sha256,
                    }
                ),
            )
            edge = (asin, base_cell)
            prior = candidates_by_edge.get(edge)
            if prior is None or (
                candidate.selection_sha256,
                candidate.source_candidate_sha256,
            ) < (
                prior.selection_sha256,
                prior.source_candidate_sha256,
            ):
                candidates_by_edge[edge] = candidate
            counts["eligible_axis_edges"] += 1

    if not seen_rows:
        raise LatentPreferenceDataError("candidate artifact is empty.")

    grouped: dict[tuple[str, str, str], list[_Candidate]] = {
        cell: [] for cell in sorted(required_cells)
    }
    for candidate in candidates_by_edge.values():
        grouped[candidate.base_cell].append(candidate)
    result: dict[tuple[str, str, str], tuple[_Candidate, ...]] = {}
    for cell, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda candidate: (
                candidate.selection_sha256,
                candidate.asin,
                candidate.source_candidate_sha256,
            ),
        )
        counts["eligible_unique_asin_cell_edges"] += len(ordered)
        if len(ordered) > config.candidate_cap_per_cell:
            counts["candidate_cap_truncated_edges"] += (
                len(ordered) - config.candidate_cap_per_cell
            )
        result[cell] = tuple(ordered[: config.candidate_cap_per_cell])
    return result, dict(sorted(counts.items()))


def _validate_source_row(row: Mapping[str, Any], *, line_number: int) -> None:
    if set(row) != _ROW_KEYS:
        raise LatentPreferenceDataError(
            f"candidate line {line_number} fields mismatch: "
            f"missing={sorted(_ROW_KEYS - set(row))} "
            f"extra={sorted(set(row) - _ROW_KEYS)}."
        )
    if row.get("schema") != CANDIDATE_SCHEMA:
        raise LatentPreferenceDataError(
            f"candidate line {line_number} has unsupported schema."
        )
    for key in (
        "asin",
        "axis",
        "attribute_value",
        "category_id",
        "classification_sha256",
        "normalized_title",
        "product_category",
        "title",
    ):
        if not isinstance(row.get(key), str) or not row[key]:
            raise LatentPreferenceDataError(
                f"candidate line {line_number} has invalid {key!r}."
            )
    asin = str(row["asin"])
    if not _ASIN_RE.fullmatch(asin):
        raise LatentPreferenceDataError(
            f"candidate line {line_number} has invalid ASIN {asin!r}."
        )
    title = str(row["title"])
    if str(row["normalized_title"]) != normalize_native_title(title):
        raise LatentPreferenceDataError(
            f"candidate line {line_number} normalized title mismatch."
        )
    evidence = row["title_evidence"]
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(value, str) or not value.strip() for value in evidence)
    ):
        raise LatentPreferenceDataError(
            f"candidate line {line_number} has invalid title_evidence."
        )
    if any(
        normalize_native_title(value) not in normalize_native_title(title)
        for value in evidence
    ):
        raise LatentPreferenceDataError(
            f"candidate line {line_number} has title evidence absent from title."
        )
    declared = str(row["classification_sha256"])
    require_sha256(declared, field=f"candidate line {line_number} classification")
    expected = canonical_sha256(
        {key: row[key] for key in _SOURCE_CLASSIFICATION_KEYS}
    )
    if declared != expected:
        raise LatentPreferenceDataError(
            f"candidate line {line_number} classification hash mismatch."
        )


def _guard_matches(axis: str, title: str) -> tuple[str, ...]:
    try:
        rules = _ATTRIBUTE_GUARD_PATTERNS[axis]
        scope = _ATTRIBUTE_GUARD_SCOPES[axis]
    except KeyError:
        return ()
    normalized = normalize_native_title(title)
    return tuple(
        value_id
        for value_id in sorted(scope)
        for patterns in (rules[value_id],)
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)
    )


def _category_title_evidence(category_id: str, title: str) -> tuple[str, ...]:
    try:
        patterns = _CATEGORY_TITLE_GUARD_PATTERNS[category_id]
    except KeyError:
        return ()
    normalized = normalize_native_title(title)
    return tuple(
        sorted(
            {
                match.group(0)
                for pattern in patterns
                for match in re.finditer(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                )
            }
        )
    )


def _category_title_exclusions(category_id: str, title: str) -> tuple[str, ...]:
    patterns = _CATEGORY_TITLE_EXCLUSION_PATTERNS.get(category_id, ())
    normalized = normalize_native_title(title)
    return tuple(
        sorted(
            {
                match.group(0)
                for pattern in patterns
                for match in re.finditer(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                )
            }
        )
    )


def _attribute_context_is_valid(
    *,
    axis: str,
    attribute_value: str,
    category_id: str,
    title: str,
) -> bool:
    rule = _ATTRIBUTE_CONTEXT_RULES.get(f"{axis}.{attribute_value}")
    if rule is None:
        return True
    category_patterns = rule["category_patterns"]
    if not isinstance(category_patterns, Mapping):
        raise AssertionError("attribute context category patterns must be a mapping")
    category_pattern = category_patterns.get(category_id)
    if not isinstance(category_pattern, str):
        return False
    max_words = rule["max_intervening_words"]
    if isinstance(max_words, bool) or not isinstance(max_words, int):
        raise AssertionError("attribute context max words must be an integer")
    attribute_pattern = rule["attribute_pattern"]
    if not isinstance(attribute_pattern, str):
        raise AssertionError("attribute context pattern must be text")
    normalized = normalize_native_title(title)
    return bool(
        re.search(
            attribute_pattern
            + rf"(?:\W+\w+){{0,{max_words}}}\W+"
            + category_pattern,
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _certify_catalog_identity(
    backend: NativeWebShopBackend,
    *,
    candidates_by_cell: Mapping[
        tuple[str, str, str], tuple[_Candidate, ...]
    ],
    config: NativePreferenceCertificationConfig,
) -> tuple[
    dict[tuple[str, str, str], tuple[_Candidate, ...]],
    dict[str, int],
]:
    counts: collections.Counter[str] = collections.Counter()
    candidates = {
        (candidate.asin, candidate.base_cell): candidate
        for values in candidates_by_cell.values()
        for candidate in values
    }
    normalized_titles = {
        candidate.normalized_title for candidate in candidates.values()
    }
    title_matches: dict[str, set[str]] = {
        title: set() for title in normalized_titles
    }
    for raw_asin in backend.product_asins():
        counts["native_catalog_title_records_scanned"] += 1
        asin = str(raw_asin).upper()
        normalized = normalize_native_title(backend.product_title(asin))
        if normalized in title_matches:
            title_matches[normalized].add(asin)

    grouped: dict[tuple[str, str, str], list[_Candidate]] = {
        cell: [] for cell in _expected_base_cells(config)
    }
    for candidate in candidates.values():
        try:
            record = backend.product_record(candidate.asin)
        except Exception:
            counts["rejected_missing_native_asin"] += 1
            continue
        if str(record.get("Title") or "") != candidate.title:
            counts["rejected_native_title_mismatch"] += 1
            continue
        if str(record.get("product_category") or "") != candidate.product_category:
            counts["rejected_native_product_category_mismatch"] += 1
            continue
        matches = title_matches[candidate.normalized_title]
        if matches != {candidate.asin}:
            counts["rejected_nonunique_native_normalized_title"] += 1
            continue
        grouped[candidate.base_cell].append(
            _Candidate(
                **{
                    **candidate.__dict__,
                    "catalog_title_match_count": len(matches),
                }
            )
        )
        counts["catalog_identity_certified_edges"] += 1
    return (
        {
            cell: tuple(
                sorted(
                    values,
                    key=lambda candidate: (
                        candidate.selection_sha256,
                        candidate.asin,
                    ),
                )
            )
            for cell, values in sorted(grouped.items())
        },
        dict(sorted(counts.items())),
    )


def _deterministic_unique_asin_matching(
    slots: Sequence[_Slot],
    candidates_by_cell: Mapping[
        tuple[str, str, str], tuple[_Candidate, ...]
    ],
    *,
    blocked_edges: set[tuple[str, tuple[str, str, str]]],
) -> dict[_Slot, _Candidate] | None:
    available: dict[_Slot, tuple[_Candidate, ...]] = {
        slot: tuple(
            candidate
            for candidate in candidates_by_cell[slot.base_cell]
            if (candidate.asin, candidate.base_cell) not in blocked_edges
        )
        for slot in slots
    }
    ordered_slots = sorted(slots, key=lambda slot: (len(available[slot]), slot))
    owner_by_asin: dict[str, _Slot] = {}
    matched: dict[_Slot, _Candidate] = {}

    def augment(slot: _Slot, seen_asins: set[str]) -> bool:
        for candidate in available[slot]:
            if candidate.asin in seen_asins:
                continue
            seen_asins.add(candidate.asin)
            previous = owner_by_asin.get(candidate.asin)
            if previous is None or augment(previous, seen_asins):
                owner_by_asin[candidate.asin] = slot
                matched[slot] = candidate
                return True
        return False

    for slot in ordered_slots:
        if not augment(slot, set()):
            return None
    if len(matched) != len(slots) or len(owner_by_asin) != len(slots):
        return None
    return matched


def _audit_native_candidate(
    backend: NativeWebShopBackend,
    *,
    candidate: _Candidate,
    probe_index: int,
    config: NativePreferenceCertificationConfig,
) -> tuple[_NativeEvidence | None, dict[str, Any]]:
    token = f"amglp-cert-{probe_index}-{candidate.asin}"
    detail: dict[str, Any] = {
        "probe_index": probe_index,
        "base_cell": "/".join(candidate.base_cell),
        "asin": candidate.asin,
        "title": candidate.title,
        "selection_sha256": candidate.selection_sha256,
        "source_candidate_sha256": candidate.source_candidate_sha256,
        "search_attempts": [],
    }

    def reject(reason: str, **extra: Any) -> tuple[None, dict[str, Any]]:
        detail.update(status="rejected", rejection_reason=reason, **extra)
        return None, detail

    try:
        record = backend.product_record(candidate.asin)
        if str(record.get("Title") or "") != candidate.title:
            return reject("native_title_changed_during_probe")
        if str(record.get("product_category") or "") != candidate.product_category:
            return reject("native_category_changed_during_probe")
        if _guard_matches(candidate.axis, candidate.title) != candidate.guard_matches:
            return reject("broad_attribute_guard_changed_during_probe")
        if backend.product_title(candidate.asin) != candidate.title:
            return reject("native_title_api_mismatch")
        price_cents = backend.product_price_cents(candidate.asin)
        if price_cents <= 0:
            return reject("nonpositive_native_price")
        record_sha256 = backend.product_record_sha256(candidate.asin)
        require_sha256(record_sha256, field="native catalog_record_sha256")

        page = backend.open_session(
            token,
            "Native execution certification for a hidden user preference task.",
        )
        if not page.has_search_bar:
            return reject("native_search_bar_missing")
        selected_query: str | None = None
        selected_rank: int | None = None
        target_seen_outside_limit = False
        queries = _candidate_search_queries(
            candidate,
            min_chars=config.min_search_query_chars,
            max_chars=config.max_search_query_chars,
        )
        if not queries:
            return reject("no_safe_title_derived_search_query")
        for query in queries:
            page = backend.step(token, f"search[{query}]")
            result_asins = tuple(
                value.upper()
                for value in page.clickables
                if _ASIN_RE.fullmatch(value.upper())
            )
            try:
                rank = result_asins.index(candidate.asin) + 1
            except ValueError:
                rank = None
            within_limit = rank is not None and rank <= config.max_search_rank
            detail["search_attempts"].append(
                {
                    "query": query,
                    "result_asins": list(result_asins),
                    "target_rank": rank,
                    "within_limit": within_limit,
                }
            )
            if within_limit:
                selected_query = query
                selected_rank = rank
                break
            if rank is not None:
                target_seen_outside_limit = True
        if selected_query is None or selected_rank is None:
            reason = (
                "native_search_rank_exceeds_limit"
                if target_seen_outside_limit
                else "target_absent_from_all_title_derived_first_pages"
            )
            return reject(reason)

        page = backend.step(token, f"click[{candidate.asin}]")
        opened_url = page.url
        if candidate.asin.casefold() not in opened_url.casefold():
            return reject("native_open_url_asin_mismatch", observed_url=opened_url)
        buy_now = next(
            (value for value in page.clickables if value.casefold() == "buy now"),
            None,
        )
        if buy_now is None:
            return reject("native_buy_now_missing")
        page = backend.step(token, f"click[{buy_now}]")
        if page.purchase is None:
            return reject("native_purchase_receipt_missing")
        if page.purchase.asin.upper() != candidate.asin:
            return reject("native_purchase_asin_mismatch")
        if page.purchase.price_cents != price_cents:
            return reject("native_purchase_price_mismatch")
        detail.update(
            status="accepted",
            rejection_reason=None,
            selected_search_query=selected_query,
            selected_search_rank=selected_rank,
            price_cents=price_cents,
            catalog_record_sha256=record_sha256,
            opened_url=opened_url,
        )
        return (
            _NativeEvidence(
                price_cents=price_cents,
                search_query=selected_query,
                search_rank=selected_rank,
                catalog_record_sha256=record_sha256,
            ),
            detail,
        )
    except Exception as exc:
        return reject(
            f"exception_{type(exc).__name__}",
            exception_type=type(exc).__name__,
        )
    finally:
        try:
            backend.close_session(token)
        except Exception:
            pass


def _candidate_search_queries(
    candidate: _Candidate,
    *,
    min_chars: int,
    max_chars: int,
) -> tuple[str, ...]:
    title = candidate.title
    normalized_title = candidate.normalized_title
    normalized_evidence = tuple(
        normalize_native_title(value) for value in candidate.title_evidence
    )
    queries: list[str] = []
    seen: set[str] = set()

    def add(raw_query: str) -> None:
        query = raw_query.strip(_QUERY_EDGE_CHARS)
        if not min_chars <= len(query) <= max_chars:
            return
        if any(char in query for char in _QUERY_UNSAFE_CHARS):
            return
        if len(_QUERY_WORD_RE.findall(query)) < 3:
            return
        normalized_query = normalize_native_title(query)
        if not normalized_query or normalized_query not in normalized_title:
            return
        if not any(value in normalized_query for value in normalized_evidence):
            return
        if normalized_query in seen:
            return
        seen.add(normalized_query)
        queries.append(query)

    add(title)
    separators = tuple(_QUERY_SEPARATOR_RE.finditer(title))
    for separator in separators:
        add(title[: separator.start()])
    segment_starts = [0, *(match.end() for match in separators)]
    segment_ends = [*(match.start() for match in separators), len(title)]
    for segment_index in range(len(segment_starts)):
        for width in (1, 2, 3):
            end_index = segment_index + width - 1
            if end_index >= len(segment_ends):
                break
            add(title[segment_starts[segment_index] : segment_ends[end_index]])

    words = tuple(_QUERY_WORD_RE.finditer(title))
    folded_title = title.casefold()
    for evidence in candidate.title_evidence:
        start = folded_title.find(evidence.casefold())
        if start < 0:
            continue
        end = start + len(evidence)
        containing = [
            index
            for index, word in enumerate(words)
            if word.start() < end and word.end() > start
        ]
        if not containing:
            continue
        first_word = containing[0]
        last_word = containing[-1]
        for trailing_words in (2, 4, 8, 12):
            end_word = min(len(words) - 1, last_word + trailing_words)
            add(title[: words[end_word].end()])
        for context_words in (4, 8, 12):
            start_word = max(0, first_word - context_words)
            end_word = min(len(words) - 1, last_word + context_words)
            add(title[words[start_word].start() : words[end_word].end()])
    for prefix_words in (6, 8, 12, 16, 24, 32):
        if len(words) >= prefix_words:
            add(title[: words[prefix_words - 1].end()])
    return tuple(queries)


def _expected_base_cells(
    config: NativePreferenceCertificationConfig,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            {
                (recipe.axis, category, value)
                for recipe in config.recipes
                for category in recipe.categories
                for value in recipe.values
            }
        )
    )


def _expected_slots(
    config: NativePreferenceCertificationConfig,
) -> tuple[_Slot, ...]:
    return tuple(
        _Slot(*cell, split, ordinal)
        for cell in _expected_base_cells(config)
        for split in SPLITS
        for ordinal in range(config.products_per_cell)
    )


def _config_payload(
    config: NativePreferenceCertificationConfig,
) -> dict[str, Any]:
    return {
        "pool_id": config.pool_id,
        "recipe_ids": list(config.recipe_ids),
        "products_per_cell": config.products_per_cell,
        "candidate_cap_per_cell": config.candidate_cap_per_cell,
        "max_search_rank": config.max_search_rank,
        "title_length_chars": [config.min_title_chars, config.max_title_chars],
        "search_query_length_chars": [
            config.min_search_query_chars,
            config.max_search_query_chars,
        ],
        "split_assignment": (
            "deterministic global ASIN-unique bipartite matching before native probes"
        ),
    }


def _stable_backend_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in (
            "surface",
            "price_seed",
            "product_count",
            "price_table_sha256",
            "upstream_provenance",
        )
        if key in metadata
    }


def _build_audit(
    *,
    status: str,
    pool: PreferenceProductPool | None,
    config: NativePreferenceCertificationConfig,
    metadata: Mapping[str, Any],
    source_manifest_sha256: str,
    candidate_artifact_sha256: str,
    catalog_sha256: str,
    attributes_sha256: str,
    price_table_sha256: str,
    lucene_index_sha256: str,
    candidates_by_cell: Mapping[
        tuple[str, str, str], tuple[_Candidate, ...]
    ],
    parse_counts: Mapping[str, int],
    catalog_counts: Mapping[str, int],
    matching_rebuilds: int,
    blocked_edges: set[tuple[str, tuple[str, str, str]]],
    rejection_counts: collections.Counter[str],
    probe_details: Sequence[Mapping[str, Any]],
    final_matching: Mapping[_Slot, _Candidate] | None,
) -> dict[str, Any]:
    per_cell = {
        "/".join(cell): {
            "shortlisted": len(candidates_by_cell[cell]),
            "required": config.products_per_cell * len(SPLITS),
            "selected": (
                0
                if final_matching is None
                else sum(
                    candidate.base_cell == cell
                    for candidate in final_matching.values()
                )
            ),
        }
        for cell in _expected_base_cells(config)
    }
    selected_asins = (
        []
        if final_matching is None
        else sorted(candidate.asin for candidate in final_matching.values())
    )
    return {
        "schema": CERTIFICATION_AUDIT_SCHEMA,
        "status": status,
        "certifier_version": CERTIFIER_VERSION,
        "pool_id": config.pool_id,
        "product_pool_semantic_sha256": (
            None if pool is None else pool.semantic_sha256
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "contract": _config_payload(config),
        "provenance": {
            "candidate_artifact_sha256": candidate_artifact_sha256,
            "catalog_sha256": catalog_sha256,
            "attributes_sha256": attributes_sha256,
            "price_table_sha256": price_table_sha256,
            "lucene_index_sha256": lucene_index_sha256,
            "rules_sha256": PREFERENCE_RULES_SHA256,
            "backend": _stable_backend_metadata(metadata),
        },
        "counts": {
            **dict(parse_counts),
            **dict(catalog_counts),
            "expected_slots": len(_expected_slots(config)),
            "matching_rebuilds": matching_rebuilds,
            "native_probes": len(probe_details),
            "native_probe_rejections": dict(sorted(rejection_counts.items())),
            "blocked_asin_cell_edges": len(blocked_edges),
            "selected_unique_asins": len(set(selected_asins)),
            "per_base_cell": per_cell,
        },
        "selected_asins": selected_asins,
        "candidate_probes": [dict(detail) for detail in probe_details],
        "verification": {
            "real_frozen_webshop_records_only": True,
            "candidate_artifact_strict_jsonl_and_sha256": True,
            "candidate_classification_hash_recomputed": True,
            "independent_axis_candidate_rows": True,
            "broader_same_axis_attribute_guard": True,
            "category_identity_requires_native_title_evidence": True,
            "multi_value_titles_rejected": True,
            "unicode_nfkc_title_normalization": True,
            "full_catalog_normalized_title_uniqueness": True,
            "global_asin_uniqueness_across_axes_cells_splits": (
                final_matching is not None
                and len(selected_asins) == len(set(selected_asins))
            ),
            "native_search_first_page": status == "certified",
            "native_item_page_exact_asin": status == "certified",
            "native_purchase_receipt_exact_asin_and_price": status == "certified",
            "native_search_result_asin_handles_visible": True,
            "native_click_action_uses_asin_handle": True,
            "human_review_required": False,
            "llm_judge_required": False,
        },
    }
