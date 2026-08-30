"""LiteResearcher Stage-1 RL intake substrate.

This package is deliberately independent from the shared policy rollout. It
provides frozen intake fixtures, the released upstream hybrid search/page
backend, and task-neutral receipts used by the AgentMemoryGym runner.
"""

from .backend import (
    FrozenLiteResearchBackend,
    LiteResearchBackendError,
    LiteResearchRequestError,
)
from .contracts import (
    LITERESEARCHER_DATA_REVISION,
    LITERESEARCHER_UPSTREAM_COMMIT,
    LiteResearcherCoverage,
    LiteResearcherTask,
    load_coverage_manifest,
)
from .wrapper import (
    LITERESEARCHER_FULLPOOL_SURFACE,
    LITERESEARCHER_SURFACE,
    LiteResearcherWrapper,
)
from .full_pool import (
    FullPoolLiteResearcherTask,
    FullPoolLiteResearcherTasks,
    load_full_pool,
)
from .lexical_backend import SQLiteFTSLiteResearchBackend
from .upstream_backend import UpstreamHybridLiteResearchBackend
from .judge import (
    LITERESEARCHER_FORMAL_JUDGE_MODEL,
    LITERESEARCHER_FORMAL_JUDGE_MODELS,
    LiteResearchJudgeInfrastructureError,
    LiteResearchJudgeResult,
    NormalizedExactLiteResearchJudge,
    UpstreamCompatibleLLMJudge,
    UPSTREAM_LLM_JUDGE_CONTRACT,
)

__all__ = [
    "FrozenLiteResearchBackend",
    "LITERESEARCHER_DATA_REVISION",
    "LITERESEARCHER_FULLPOOL_SURFACE",
    "LITERESEARCHER_FORMAL_JUDGE_MODEL",
    "LITERESEARCHER_FORMAL_JUDGE_MODELS",
    "LITERESEARCHER_UPSTREAM_COMMIT",
    "LITERESEARCHER_SURFACE",
    "LiteResearchBackendError",
    "LiteResearchRequestError",
    "LiteResearcherCoverage",
    "LiteResearcherTask",
    "LiteResearcherWrapper",
    "FullPoolLiteResearcherTask",
    "FullPoolLiteResearcherTasks",
    "SQLiteFTSLiteResearchBackend",
    "UpstreamHybridLiteResearchBackend",
    "LiteResearchJudgeInfrastructureError",
    "LiteResearchJudgeResult",
    "NormalizedExactLiteResearchJudge",
    "UpstreamCompatibleLLMJudge",
    "UPSTREAM_LLM_JUDGE_CONTRACT",
    "load_full_pool",
    "load_coverage_manifest",
]
