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
    contract_digest,
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
    "MemoryRewardPolicy",
    "MemoryToolRuntime",
    "contract_digest",
]
