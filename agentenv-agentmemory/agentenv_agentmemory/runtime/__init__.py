"""Generic AgentMemoryGym domain runtime.

The v3 runtime is intentionally parallel to the legacy WebShop v2 path.  New
domains use these interfaces; WebShop stays on its frozen implementation until
byte-for-byte compatibility fixtures allow a controlled migration.
"""

from .domain import (
    DomainContract,
    DomainDriver,
    DomainFactory,
    DomainTransition,
    MEMORY_ACTION_DESCRIPTIONS,
    contract_digest,
    render_system_prompt,
)
from .memory import MemoryRewardPolicy, MemoryToolRuntime
from .registry import DomainRegistry
from .wrapper import DomainEnvWrapper, MemoryAugmentedDriver

__all__ = [
    "DomainContract",
    "DomainDriver",
    "DomainEnvWrapper",
    "DomainFactory",
    "DomainRegistry",
    "DomainTransition",
    "MemoryAugmentedDriver",
    "MEMORY_ACTION_DESCRIPTIONS",
    "MemoryRewardPolicy",
    "MemoryToolRuntime",
    "contract_digest",
    "render_system_prompt",
]
