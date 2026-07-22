"""Domain drivers for the AgentMemoryGym v3 runtime."""

from .browsecomp import BROWSECOMP_SURFACE, BrowseCompPlusFactory
from .formal_reasoning import (
    FORMAL_REASONING_SURFACES,
    FormalReasoningFactory,
)
from .travel import TRAVEL_SURFACE, TravelPlannerFactory

V3_SURFACES = (
    TRAVEL_SURFACE,
    FORMAL_REASONING_SURFACES["math"],
    FORMAL_REASONING_SURFACES["phys"],
    BROWSECOMP_SURFACE,
)

__all__ = [
    "BROWSECOMP_SURFACE",
    "BrowseCompPlusFactory",
    "FORMAL_REASONING_SURFACES",
    "FormalReasoningFactory",
    "TRAVEL_SURFACE",
    "TravelPlannerFactory",
    "V3_SURFACES",
]
