from __future__ import annotations

from collections.abc import Callable

from .domain import DomainFactory


FactoryBuilder = Callable[[], DomainFactory]


class DomainRegistry:
    """Fail-closed registry whose builders may lazily import heavy domains."""

    def __init__(self) -> None:
        self._builders: dict[str, FactoryBuilder] = {}

    def register(self, surface: str, builder: FactoryBuilder) -> None:
        normalized = _normalize_surface(surface)
        if normalized in self._builders:
            raise ValueError(f"surface already registered: {normalized}")
        self._builders[normalized] = builder

    def build(self, surface: str) -> DomainFactory:
        normalized = _normalize_surface(surface)
        try:
            factory = self._builders[normalized]()
        except KeyError as exc:
            available = ", ".join(sorted(self._builders)) or "<none>"
            raise RuntimeError(
                f"unknown AgentMemoryGym v3 surface {normalized!r}; available: {available}"
            ) from exc
        if factory.surface != normalized:
            raise RuntimeError(
                "domain factory surface mismatch: "
                f"registered={normalized!r} factory={factory.surface!r}"
            )
        return factory

    def surfaces(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))


def _normalize_surface(surface: str) -> str:
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("surface must be a non-empty string")
    return surface.strip()
