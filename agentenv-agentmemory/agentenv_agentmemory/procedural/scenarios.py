from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCENARIO_DEFINITION_VERSION = "memoryarena_natural_order_chains_v2"


def _canonical_sha256(payload: Any) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class AttributeValueSpec:
    value_id: str
    display_name: str
    title_patterns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "display_name": self.display_name,
            "title_patterns": list(self.title_patterns),
        }


@dataclass(frozen=True)
class SlotSpec:
    scenario_id: str
    slot_id: str
    display_name: str
    attribute_name: str
    title_patterns: tuple[str, ...]
    title_exclusions: tuple[str, ...]
    product_category_patterns: tuple[str, ...]
    product_category_exclusions: tuple[str, ...]
    values: tuple[AttributeValueSpec, AttributeValueSpec]

    def as_dict(self, *, include_rules: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "slot_id": self.slot_id,
            "display_name": self.display_name,
            "attribute_name": self.attribute_name,
            "values": [value.as_dict() for value in self.values],
        }
        if include_rules:
            payload.update(
                {
                    "title_patterns": list(self.title_patterns),
                    "title_exclusions": list(self.title_exclusions),
                    "product_category_patterns": list(
                        self.product_category_patterns
                    ),
                    "product_category_exclusions": list(
                        self.product_category_exclusions
                    ),
                }
            )
        return payload

    @property
    def value_ids(self) -> tuple[str, str]:
        return tuple(value.value_id for value in self.values)  # type: ignore[return-value]

    def value(self, value_id: str) -> AttributeValueSpec:
        for value in self.values:
            if value.value_id == value_id:
                return value
        raise KeyError(
            f"Unknown attribute value {value_id!r} for {self.scenario_id}/{self.slot_id}."
        )


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    display_name: str
    order_context: str
    native_category: str
    slots: tuple[SlotSpec, ...]

    def as_dict(self, *, include_rules: bool = True) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "display_name": self.display_name,
            "order_context": self.order_context,
            "native_category": self.native_category,
            "slots": [slot.as_dict(include_rules=include_rules) for slot in self.slots],
        }

    def slot(self, slot_id: str) -> SlotSpec:
        for slot in self.slots:
            if slot.slot_id == slot_id:
                return slot
        raise KeyError(f"Unknown slot {slot_id!r} for scenario {self.scenario_id!r}.")


@dataclass(frozen=True)
class ProductClassification:
    scenario_id: str
    slot_id: str
    attribute_name: str
    attribute_value: str
    attribute_display_name: str
    native_category: str
    catalog_query: str
    product_category: str
    slot_title_evidence: tuple[str, ...]
    attribute_title_evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "slot_id": self.slot_id,
            "attribute_name": self.attribute_name,
            "attribute_value": self.attribute_value,
            "attribute_display_name": self.attribute_display_name,
            "native_category": self.native_category,
            "catalog_query": self.catalog_query,
            "product_category": self.product_category,
            "slot_title_evidence": list(self.slot_title_evidence),
            "attribute_title_evidence": list(self.attribute_title_evidence),
            "scenario_definition_version": SCENARIO_DEFINITION_VERSION,
            "scenario_definition_sha256": SCENARIO_DEFINITION_SHA256,
        }

    @property
    def semantic_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


def _value(value_id: str, display_name: str, *patterns: str) -> AttributeValueSpec:
    return AttributeValueSpec(value_id, display_name, tuple(patterns))


def _slot(
    scenario_id: str,
    slot_id: str,
    display_name: str,
    attribute_name: str,
    values: tuple[AttributeValueSpec, AttributeValueSpec],
    *,
    title_patterns: Sequence[str],
    product_category_patterns: Sequence[str],
    title_exclusions: Sequence[str] = (),
    product_category_exclusions: Sequence[str] = (),
) -> SlotSpec:
    return SlotSpec(
        scenario_id=scenario_id,
        slot_id=slot_id,
        display_name=display_name,
        attribute_name=attribute_name,
        title_patterns=tuple(title_patterns),
        title_exclusions=tuple(title_exclusions),
        product_category_patterns=tuple(product_category_patterns),
        product_category_exclusions=tuple(product_category_exclusions),
        values=values,
    )


BAKING = ScenarioSpec(
    scenario_id="baking",
    display_name="Celebration dessert table",
    order_context="a celebration dessert-table order",
    native_category="grocery",
    slots=(
        _slot(
            "baking",
            "cake_base",
            "cake base",
            "listed cake flavor",
            (
                _value("chocolate", "chocolate", r"\bchocolate\b"),
                _value("vanilla", "vanilla", r"\bvanilla\b"),
            ),
            title_patterns=(r"\b(?:cake|cupcake)\s+mix\b", r"\bcake base\b"),
            title_exclusions=(r"\bfrosting\b", r"\btopper\b"),
            product_category_patterns=(r"› Baking Mixes › Cakes$",),
        ),
        _slot(
            "baking",
            "frosting",
            "frosting",
            "frosting flavor",
            (
                _value("chocolate", "chocolate", r"\bchocolate\b", r"\bfudge\b"),
                _value("cream_cheese", "cream cheese", r"\bcream cheese\b"),
            ),
            title_patterns=(r"\bfrosting\b", r"\b(?:cake )?icing\b"),
            title_exclusions=(r"\bhair\b", r"\bmix\b", r"\bcake with\b"),
            product_category_patterns=(
                r"› Frosting, Icing & Decorations(?: › Icing Decorations)?$",
            ),
        ),
        _slot(
            "baking",
            "coloring",
            "food coloring",
            "color",
            (
                _value("red", "red", r"\bred\b"),
                _value("blue", "blue", r"\bblue\b"),
            ),
            title_patterns=(
                r"\bfood (?:color(?:ing)?|dye)\b",
                r"\bicing color\b",
                r"\bairbrush food color\b",
            ),
            title_exclusions=(r"\bhair\b",),
            product_category_patterns=(r"› Cooking & Baking › Food Coloring$",),
        ),
        _slot(
            "baking",
            "sprinkles",
            "decorating sprinkles",
            "listed accent color",
            (
                _value("gold", "gold", r"\bgold(?:en)?\b"),
                _value("silver", "silver", r"\bsilver\b"),
            ),
            title_patterns=(r"\bsprinkles?\b",),
            product_category_patterns=(
                r"› Frosting, Icing & Decorations › Sprinkles$",
            ),
        ),
        _slot(
            "baking",
            "topper",
            "cake topper",
            "listed occasion",
            (
                _value("birthday", "birthday", r"\bbirthday\b"),
                _value("wedding", "wedding", r"\bwedding\b"),
            ),
            title_patterns=(r"\b(?:cake|cupcake) toppers?\b",),
            title_exclusions=(r"\bsprinkles?\b",),
            product_category_patterns=(
                r"› Frosting, Icing & Decorations › (?:Cake|Cupcake) Toppers$",
            ),
        ),
        _slot(
            "baking",
            "cookies",
            "cookies",
            "cookie style",
            (
                _value(
                    "chocolate_chip",
                    "chocolate chip",
                    r"\bchocolate chip\b",
                ),
                _value("shortbread", "shortbread", r"\bshortbread\b"),
            ),
            title_patterns=(r"\bcookies?\b",),
            title_exclusions=(
                r"\bcookie (?:cutter|scoop|jar|sheet)s?\b",
                r"\bcookie (?:mix|dough)\b",
            ),
            product_category_patterns=(r"› Breads & Bakery › Cookies(?: › .+)?$",),
        ),
    ),
)


BEAUTY = ScenarioSpec(
    scenario_id="beauty",
    display_name="Skin-care routine",
    order_context="a personal skin-care routine",
    native_category="beauty",
    slots=(
        _slot(
            "beauty",
            "cleanser",
            "facial cleanser",
            "cleanser texture",
            (
                _value(
                    "gel",
                    "gel",
                    r"\bgel[ -]?(?:cleanser|face wash|facial wash)\b",
                    r"\b(?:cleansing|cleanser|face wash|facial wash) gel\b",
                ),
                _value("foam", "foam", r"\bfoam(?:ing)?\b"),
            ),
            title_patterns=(
                r"\b(?:face|facial)\b.{0,60}\b(?:cleanser|wash)\b",
                r"\b(?:cleanser|wash)\b.{0,60}\b(?:face|facial)\b",
            ),
            title_exclusions=(r"\bhair\b", r"\bshampoo\b"),
            product_category_patterns=(
                r"› Skin Care › Face › Cleansers(?: › (?:Gels|Washes))?$",
            ),
        ),
        _slot(
            "beauty",
            "toner",
            "facial toner",
            "toner ingredient",
            (
                _value("rose", "rose", r"\brose(?: water)?\b"),
                _value("witch_hazel", "witch hazel", r"\bwitch hazel\b"),
            ),
            title_patterns=(
                r"\b(?:face|facial|skin)\b.{0,70}\b(?:toner|astringent)\b",
                r"\b(?:toner|astringent)\b.{0,70}\b(?:face|facial|skin)\b",
                r"\b(?:rose water|witch hazel)\b.{0,50}\b(?:toner|astringent)\b",
                r"\b(?:toner|astringent)\b.{0,50}\b(?:rose water|witch hazel)\b",
            ),
            title_exclusions=(r"\bhair\b", r"\bprinter\b"),
            product_category_patterns=(
                r"› Skin Care › Face › Toners & Astringents$",
            ),
        ),
        _slot(
            "beauty",
            "active",
            "facial serum",
            "active ingredient",
            (
                _value("niacinamide", "niacinamide", r"\bniacinamide\b"),
                _value(
                    "hyaluronic_acid",
                    "hyaluronic acid",
                    r"\bhyaluronic acid\b",
                ),
            ),
            title_patterns=(r"\bserum\b", r"\bface treatment\b"),
            title_exclusions=(
                r"\bhair\b",
                r"\beye(?:lid)?\b",
                r"\blash\b",
                r"\beyebrow\b",
                r"\bmoisturizer\b",
                r"\bserum cream\b",
            ),
            product_category_patterns=(
                r"› Skin Care › Face › Treatments & Masks › Serums$",
            ),
        ),
        _slot(
            "beauty",
            "weekly_treatment",
            "weekly face treatment",
            "mask type",
            (
                _value("clay", "clay mask", r"\bclay\b"),
                _value("sheet", "sheet mask", r"\bsheet\b"),
            ),
            title_patterns=(
                r"\b(?:face|facial|skin)\b.{0,60}\b(?:mask|masque)\b",
                r"\b(?:mask|masque)\b.{0,60}\b(?:face|facial|skin)\b",
                r"\bclay mask\b",
                r"\bsheet mask\b",
            ),
            title_exclusions=(r"\bhair\b", r"\bfoot\b", r"\blip\b", r"\beye mask\b"),
            product_category_patterns=(
                r"› Skin Care › Face › Treatments & Masks › Masks$",
            ),
        ),
        _slot(
            "beauty",
            "moisturizer",
            "facial moisturizer",
            "moisturizer format",
            (
                _value("gel_cream", "gel cream", r"\bgel[ -]?cream\b"),
                _value("night_cream", "night cream", r"\bnight cream\b"),
            ),
            title_patterns=(
                r"\b(?:face|facial|skin)\b.{0,70}\b(?:cream|moisturizer|moisturiser)\b",
                r"\b(?:cream|moisturizer|moisturiser)\b.{0,70}\b(?:face|facial|skin)\b",
            ),
            title_exclusions=(r"\bhair\b", r"\bhand cream\b", r"\bfoot cream\b"),
            product_category_patterns=(
                r"› Skin Care › Face › Creams & Moisturizers › "
                r"(?:Face Moisturizers|Night Creams)$",
            ),
        ),
        _slot(
            "beauty",
            "eye_treatment",
            "eye treatment",
            "active ingredient",
            (
                _value("caffeine", "caffeine", r"\bcaffeine\b"),
                _value("peptide", "peptide", r"\bpeptides?\b"),
            ),
            title_patterns=(
                r"\beye\b.{0,45}\b(?:cream|treatment|moisturizer|serum)\b",
                r"\b(?:cream|treatment|moisturizer|serum)\b.{0,45}\beye\b",
            ),
            title_exclusions=(r"\bmakeup\b", r"\bconcealer\b", r"\bshadow\b"),
            product_category_patterns=(r"› Skin Care › Eyes › (?:Creams|Serums)$",),
        ),
    ),
)


ELECTRONICS = ScenarioSpec(
    scenario_id="electronics",
    display_name="Home media setup",
    order_context="a home media setup",
    native_category="electronics",
    slots=(
        _slot(
            "electronics",
            "display",
            "display",
            "display resolution",
            (
                _value(
                    "1080p",
                    "1080p",
                    r"\b1080p\b",
                    r"\bfull hd\b",
                    r"\b1920\s*[x×]\s*1080\b",
                ),
                _value("4k", "4K", r"\b4k\b", r"\bultra hd\b", r"\buhd\b"),
            ),
            title_patterns=(r"\bmonitor\b", r"\bdisplay\b", r"\btelevision\b", r"\btv\b"),
            title_exclusions=(
                r"\badapter\b",
                r"\bcharger\b",
                r"\bcompatible (?:with|for)\b",
                r"\bdvr\b",
                r"\bcamera\b",
                r"\bdash cam\b",
                r"\bmount\b",
                r"\bpower (?:cord|supply)\b",
                r"\breceiver\b",
                r"\breplacement\b",
                r"\bremote\b",
                r"\bstand\b",
            ),
            product_category_patterns=(
                r"› Computers & Accessories › Monitors$",
                r"› Television & Video › Televisions"
                r"(?: › (?:LED & LCD TVs|QLED TVs|OLED TVs))?$",
            ),
        ),
        _slot(
            "electronics",
            "audio",
            "audio system",
            "audio setup",
            (
                _value("soundbar", "soundbar", r"\bsound ?bar\b"),
                _value(
                    "bookshelf_speakers",
                    "bookshelf speakers",
                    r"\bbookshelf speakers?\b",
                ),
            ),
            title_patterns=(r"\bsound ?bar\b", r"\bbookshelf speakers?\b"),
            title_exclusions=(
                r"\badapter\b",
                r"\bcompatible (?:with|for)\b",
                r"\bmount\b",
                r"\bpower (?:cord|supply)\b",
                r"\breplacement\b",
                r"\bremote\b",
                r"\bstand\b",
            ),
            product_category_patterns=(
                r"› Home Audio › Speakers › (?:Sound Bars|Bookshelf Speakers)$",
            ),
        ),
        _slot(
            "electronics",
            "mount",
            "display mount",
            "mount movement",
            (
                _value("fixed", "fixed", r"\bfixed\b"),
                _value("full_motion", "full motion", r"\bfull[ -]?motion\b"),
            ),
            title_patterns=(
                r"\b(?:tv|television|monitor)\b.{0,70}\b(?:mount|bracket)\b",
                r"\b(?:mount|bracket)\b.{0,70}\b(?:tv|television|monitor)\b",
            ),
            title_exclusions=(r"\bsound ?bar\b", r"\bspeaker\b"),
            product_category_patterns=(
                r"› Television Accessories › TV Mounts, Stands & Turntables "
                r"› TV Ceiling & Wall Mounts$",
            ),
        ),
        _slot(
            "electronics",
            "source",
            "content source",
            "source type",
            (
                _value(
                    "streaming_player",
                    "streaming media player",
                    r"\bstreaming\b",
                    r"\broku\b",
                    r"\bchromecast\b",
                    r"\bfire tv\b",
                ),
                _value(
                    "blu_ray_player",
                    "Blu-ray player",
                    r"\bblu[ -]?ray(?: disc)? players?\b",
                ),
            ),
            title_patterns=(
                r"\bstreaming\b.{0,50}\b(?:player|stick|box)\b",
                r"\b(?:roku|chromecast|fire tv)\b",
                r"\bblu[ -]?ray(?: disc)? players?\b",
            ),
            title_exclusions=(
                r"\badapter\b",
                r"\bburner\b",
                r"\bcables?\b",
                r"\bcase\b",
                r"\bcompatible (?:with|for)\b",
                r"\bcover\b",
                r"\bdrive\b",
                r"\bexternal\b",
                r"\bmount\b",
                r"\bpower (?:cord|supply)\b",
                r"\breplacement\b",
                r"\bremote\b",
                r"\bskins?\b",
                r"\bwriter\b",
            ),
            product_category_patterns=(
                r"› Television & Video › Streaming Media Players$",
                r"› Television & Video › Blu-ray Players & Recorders "
                r"› Blu-ray Players$",
            ),
        ),
        _slot(
            "electronics",
            "cable",
            "signal cable",
            "connector type",
            (
                _value("hdmi", "HDMI", r"\bhdmi\b"),
                _value("displayport", "DisplayPort", r"\bdisplay ?port\b"),
            ),
            title_patterns=(r"\bcables?\b", r"\bcords?\b"),
            title_exclusions=(r"\brepeater\b", r"\bsignal booster\b"),
            product_category_patterns=(
                r"› Audio & Video Accessories › Cables & Interconnects "
                r"› Video Cables(?: › [^›]+)?$",
                r"› Computer Accessories & Peripherals › Cables & Accessories "
                r"› Cables & Interconnects(?: › [^›]+)?$",
            ),
        ),
        _slot(
            "electronics",
            "power",
            "power protection",
            "protection type",
            (
                _value(
                    "surge_protector",
                    "surge protector",
                    r"\bsurge protector\b",
                ),
                _value(
                    "ups",
                    "uninterruptible power supply",
                    r"\buninterruptible power(?: supply)?\b",
                    r"\bups\b.{0,35}\b(?:system|battery backup|power supply|\d{3,4}\s*va)\b",
                    r"\b(?:system|battery backup|power supply|\d{3,4}\s*va)\b.{0,35}\bups\b",
                    r"\bback-?ups\b",
                ),
            ),
            title_patterns=(
                r"\bsurge protector\b",
                r"\buninterruptible power\b",
                r"\bups\b",
            ),
            title_exclusions=(
                r"\b(?:holder|mount|shelf|bracket)\b",
                r"\bcover ups\b",
                r"\breplacement\b",
                r"\bswimwear\b",
            ),
            product_category_patterns=(
                r"› Power Strips & Surge Protectors › "
                r"(?:Power Strips|Surge Protectors)$",
                r"› Computer Accessories & Peripherals "
                r"› Uninterruptible Power Supply \(UPS\)$",
            ),
        ),
    ),
)


GROCERY = ScenarioSpec(
    scenario_id="grocery",
    display_name="Movie-night snack box",
    order_context="a movie-night snack and drink box",
    native_category="grocery",
    slots=(
        _slot(
            "grocery",
            "popcorn",
            "popcorn",
            "popcorn flavor",
            (
                _value("caramel", "caramel", r"\bcaramel\b"),
                _value("cheddar", "cheddar", r"\bcheddar(?:corn)?\b"),
            ),
            title_patterns=(r"\bpopcorn\b",),
            product_category_patterns=(r"› Snack Foods › Popcorn › Popped$",),
        ),
        _slot(
            "grocery",
            "pretzel",
            "pretzels",
            "pretzel shape",
            (
                _value("twists", "twists", r"\btwists?\b"),
                _value("sticks", "sticks", r"\bsticks?\b"),
            ),
            title_patterns=(r"\bpretzels?\b",),
            product_category_patterns=(r"› Snack Foods › Pretzels$",),
        ),
        _slot(
            "grocery",
            "potato_chips",
            "potato chips",
            "chip flavor",
            (
                _value(
                    "barbecue",
                    "barbecue",
                    r"\b(?:barbecue|barbeque|bbq)\b",
                ),
                _value("cheddar", "cheddar", r"\bcheddar\b"),
            ),
            title_patterns=(r"\bpotato chips?\b",),
            title_exclusions=(r"\bdip\b", r"\bseasoning\b"),
            product_category_patterns=(r"› Chips & Crisps › Potato$",),
        ),
        _slot(
            "grocery",
            "crackers",
            "crackers",
            "cracker type",
            (
                _value("graham", "graham", r"\bgraham\b"),
                _value("saltine", "saltine", r"\bsaltine\b"),
            ),
            title_patterns=(r"\bcrackers?\b",),
            title_exclusions=(r"\bcrust\b",),
            product_category_patterns=(
                r"› Snack Foods › Crackers(?: › (?:Graham Crackers|Saltines))?$",
            ),
        ),
        _slot(
            "grocery",
            "sparkling_water",
            "sparkling water",
            "listed drink flavor",
            (
                _value("lemon", "lemon", r"\blemon\b"),
                _value("lime", "lime", r"\blime\b"),
            ),
            title_patterns=(r"\bsparkling water\b", r"\bseltzer\b"),
            product_category_patterns=(
                r"› Water › (?:Seltzer Water|Sparkling Water)$",
            ),
        ),
        _slot(
            "grocery",
            "chocolate_bar",
            "chocolate bar",
            "listed chocolate type",
            (
                _value(
                    "dark",
                    "dark chocolate",
                    r"\bdark chocolate\b",
                    r"\bdark\b(?=.{0,18}\bchocolate\b)",
                ),
                _value("milk", "milk chocolate", r"\bmilk chocolate\b"),
            ),
            title_patterns=(r"\bchocolate bars?\b",),
            title_exclusions=(r"\bprotein bars?\b", r"\bgranola bars?\b"),
            product_category_patterns=(r"› Chocolate › Candy & Chocolate Bars$",),
        ),
    ),
)


HOME = ScenarioSpec(
    scenario_id="home",
    display_name="Living-room furnishing plan",
    order_context="a coordinated living-room furnishing plan",
    native_category="garden",
    slots=(
        _slot(
            "home",
            "seating",
            "primary seating",
            "upholstery material",
            (
                _value("leather", "leather", r"\b(?:faux |pu )?leather\b"),
                _value("velvet", "velvet", r"\bvelvet\b"),
            ),
            title_patterns=(r"\bsofa\b", r"\bcouch\b", r"\bloveseat\b", r"\bfuton\b"),
            title_exclusions=(r"\bcover\b", r"\bprotector\b", r"\btable\b"),
            product_category_patterns=(
                r"› Furniture › Living Room Furniture › Sofas & Couches$",
            ),
        ),
        _slot(
            "home",
            "footrest",
            "footrest",
            "upholstery material",
            (
                _value("leather", "leather", r"\b(?:faux |pu )?leather\b"),
                _value("velvet", "velvet", r"\bvelvet\b"),
            ),
            title_patterns=(r"\bott(?:o|o)man\b", r"\bfootstool\b", r"\bfootrest\b"),
            title_exclusions=(r"\bcover\b",),
            product_category_patterns=(
                r"› Furniture › Accent Furniture › Ottomans$",
            ),
        ),
        _slot(
            "home",
            "coffee_table",
            "coffee table",
            "tabletop material",
            (
                _value("wood", "wood", r"\bwood(?:en)?\b"),
                _value("glass", "glass", r"\bglass\b"),
            ),
            title_patterns=(r"\bcoffee table\b", r"\bcocktail table\b"),
            title_exclusions=(
                r"\b(?:bistro|conversation|furniture) set\b",
                r"\bchairs?\b",
                r"\bset of\b",
            ),
            product_category_patterns=(
                r"› Furniture › Living Room Furniture › Tables › Coffee Tables$",
            ),
        ),
        _slot(
            "home",
            "side_table",
            "side table",
            "tabletop material",
            (
                _value("wood", "wood", r"\bwood(?:en)?\b"),
                _value("glass", "glass", r"\bglass\b"),
            ),
            title_patterns=(r"\bend table\b", r"\bside table\b", r"\baccent table\b"),
            title_exclusions=(r"\bcoffee table\b", r"\bcocktail table\b"),
            product_category_patterns=(
                r"› Furniture › Living Room Furniture › Tables › End Tables$",
            ),
        ),
        _slot(
            "home",
            "rug",
            "area rug",
            "rug color",
            (
                _value("beige", "beige", r"\bbeige\b"),
                _value("gray", "gray", r"\bgr[ae]y\b"),
            ),
            title_patterns=(r"\barea rug\b", r"\brugs?\b"),
            title_exclusions=(r"\bpad\b", r"\bgripper\b"),
            product_category_patterns=(
                r"› Home Décor Products › Rugs, Pads & Protectors › Area Rugs$",
            ),
        ),
        _slot(
            "home",
            "curtains",
            "window curtains",
            "curtain material",
            (
                _value("linen", "linen", r"\blinen\b"),
                _value("silk", "silk", r"\bsilk\b"),
            ),
            title_patterns=(r"\bcurtains?\b", r"\bdrapes?\b"),
            title_exclusions=(r"\bshower curtain\b",),
            product_category_patterns=(
                r"› Home Décor Products › Window Treatments › Curtains & Drapes "
                r"› Panels$",
            ),
        ),
    ),
)


SCENARIOS: tuple[ScenarioSpec, ...] = (BAKING, BEAUTY, ELECTRONICS, GROCERY, HOME)
SCENARIOS_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}
SCENARIOS_BY_NATIVE_CATEGORY: dict[str, tuple[ScenarioSpec, ...]] = {}
for _scenario in SCENARIOS:
    SCENARIOS_BY_NATIVE_CATEGORY.setdefault(_scenario.native_category, ())
    SCENARIOS_BY_NATIVE_CATEGORY[_scenario.native_category] += (_scenario,)


def scenario_definition_manifest() -> dict[str, Any]:
    return {
        "version": SCENARIO_DEFINITION_VERSION,
        "scenarios": [scenario.as_dict(include_rules=True) for scenario in SCENARIOS],
    }


SCENARIO_DEFINITION_SHA256 = _canonical_sha256(scenario_definition_manifest())


def scenario_by_id(scenario_id: str) -> ScenarioSpec:
    try:
        return SCENARIOS_BY_ID[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown procedural shopping scenario {scenario_id!r}.") from exc


def classify_product_record(
    record: Mapping[str, Any],
    *,
    scenario_ids: Sequence[str] | None = None,
) -> tuple[ProductClassification, ...]:
    """Return every exact natural-attribute cell matched by a native record.

    Certification accepts only records with exactly one returned match. This
    makes category and attribute ambiguity a rejection rather than a judgment
    call.
    """

    title = _canonical_record_text(record.get("Title"))
    native_category = _canonical_record_text(record.get("category")).casefold()
    catalog_query = _canonical_record_text(record.get("query"))
    product_category = _canonical_record_text(record.get("product_category"))
    if not title or not native_category:
        return ()
    allowed = set(scenario_ids) if scenario_ids is not None else None
    scenarios = SCENARIOS_BY_NATIVE_CATEGORY.get(native_category, ())
    matches: list[ProductClassification] = []
    for scenario in scenarios:
        if allowed is not None and scenario.scenario_id not in allowed:
            continue
        for slot in scenario.slots:
            slot_evidence = _matched_text(title, slot.title_patterns)
            if not slot_evidence:
                continue
            if _matches_any(title, slot.title_exclusions):
                continue
            if not _matches_any(
                product_category,
                slot.product_category_patterns,
            ):
                continue
            if _matches_any(
                product_category,
                slot.product_category_exclusions,
            ):
                continue
            value_matches: list[tuple[AttributeValueSpec, tuple[str, ...]]] = []
            for value in slot.values:
                evidence = _matched_text(title, value.title_patterns)
                if evidence:
                    value_matches.append((value, evidence))
            if len(value_matches) != 1:
                continue
            value, attribute_evidence = value_matches[0]
            matches.append(
                ProductClassification(
                    scenario_id=scenario.scenario_id,
                    slot_id=slot.slot_id,
                    attribute_name=slot.attribute_name,
                    attribute_value=value.value_id,
                    attribute_display_name=value.display_name,
                    native_category=native_category,
                    catalog_query=catalog_query,
                    product_category=product_category,
                    slot_title_evidence=slot_evidence,
                    attribute_title_evidence=attribute_evidence,
                )
            )
    return tuple(matches)


def require_unique_product_classification(
    record: Mapping[str, Any],
    *,
    scenario_ids: Sequence[str] | None = None,
) -> ProductClassification:
    matches = classify_product_record(record, scenario_ids=scenario_ids)
    if len(matches) != 1:
        cells = [f"{item.scenario_id}/{item.slot_id}/{item.attribute_value}" for item in matches]
        raise ValueError(
            "native product must match exactly one scenario/slot/attribute cell; "
            f"observed {len(matches)}: {cells}"
        )
    return matches[0]


def _canonical_record_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _matched_text(text: str, patterns: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            value = " ".join(match.group(0).split())
            if value.casefold() not in {item.casefold() for item in values}:
                values.append(value)
    return tuple(values)


def validate_scenario_definitions() -> None:
    if len(SCENARIOS) != 5:
        raise ValueError("procedural shopping requires exactly five scenarios")
    scenario_ids = [scenario.scenario_id for scenario in SCENARIOS]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario IDs must be unique")
    for scenario in SCENARIOS:
        if len(scenario.slots) != 6:
            raise ValueError(f"scenario {scenario.scenario_id} must contain six slots")
        slot_ids = [slot.slot_id for slot in scenario.slots]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError(f"scenario {scenario.scenario_id} has duplicate slots")
        for slot in scenario.slots:
            if slot.scenario_id != scenario.scenario_id:
                raise ValueError("slot scenario ID mismatch")
            if len(slot.values) != 2:
                raise ValueError("every slot must define two natural attribute values")
            if len(set(slot.value_ids)) != 2:
                raise ValueError("slot attribute values must be distinct")
            if not slot.title_patterns:
                raise ValueError("every slot needs a fail-closed title classifier")
            if not slot.product_category_patterns:
                raise ValueError(
                    "every slot needs a fail-closed product-category classifier"
                )
            if any(not value.title_patterns for value in slot.values):
                raise ValueError("every attribute value needs title evidence")


validate_scenario_definitions()
