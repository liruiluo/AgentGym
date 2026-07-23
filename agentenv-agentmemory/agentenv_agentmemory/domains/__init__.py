"""Domain drivers for the AgentMemoryGym v3 runtime."""

from .browsecomp import BROWSECOMP_SURFACES, BrowseCompPlusFactory
from .formal_reasoning import (
    FORMAL_REASONING_PAPER_EVAL_SURFACES,
    FORMAL_REASONING_SURFACES,
    FORMAL_REASONING_SURFACES_BY_MODE,
    FormalReasoningFactory,
)
from .travel import TRAVEL_SURFACES, TravelPlannerFactory

V3_SURFACES = (
    TRAVEL_SURFACES["failfast"],
    TRAVEL_SURFACES["paper_eval"],
    FORMAL_REASONING_SURFACES["math"],
    FORMAL_REASONING_SURFACES["phys"],
    FORMAL_REASONING_PAPER_EVAL_SURFACES["math"],
    FORMAL_REASONING_PAPER_EVAL_SURFACES["phys"],
    BROWSECOMP_SURFACES["paper_eval"],
    BROWSECOMP_SURFACES["failfast"],
)

__all__ = [
    "BROWSECOMP_SURFACES",
    "BrowseCompPlusFactory",
    "FORMAL_REASONING_PAPER_EVAL_SURFACES",
    "FORMAL_REASONING_SURFACES",
    "FORMAL_REASONING_SURFACES_BY_MODE",
    "FormalReasoningFactory",
    "TRAVEL_SURFACES",
    "TravelPlannerFactory",
    "V3_SURFACES",
]
