from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping

from .catalog import SettingCatalog


class SettingSource(IntEnum):
    BUILTIN = 10
    PROFILE = 20
    MODEL = 30
    USER = 40
    ENV = 50
    BUNDLE = 60
    LEGACY_CLI = 70
    CLI_SET = 80
    CONSTRAINT = 90


@dataclass(frozen=True)
class SourceValue:
    source: SettingSource
    value: Any


@dataclass(frozen=True)
class ProvenanceRecord:
    source: SettingSource
    display_value: Any
    shadowed: tuple[SourceValue, ...]
    requested_value: Any | None = None
    reason: str | None = None


class ResolvedSettings:
    def __init__(
        self, values: dict[str, Any], provenance: dict[str, ProvenanceRecord]
    ):
        self._values = MappingProxyType(dict(values))
        self.provenance = MappingProxyType(dict(provenance))

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        if not redact:
            return dict(self._values)
        return {
            name: record.display_value
            for name, record in self.provenance.items()
        }

    def with_constraints(
        self, constraints: Mapping[str, tuple[Any, str]]
    ) -> ResolvedSettings:
        """Return a snapshot with changed values attributed to constraints."""

        values = dict(self._values)
        provenance = dict(self.provenance)
        for name, (effective_value, reason) in constraints.items():
            if name not in values:
                raise ValueError(f"unknown resolved setting: {name}")
            requested_value = values[name]
            if effective_value == requested_value:
                continue
            previous = provenance[name]
            values[name] = effective_value
            provenance[name] = ProvenanceRecord(
                SettingSource.CONSTRAINT,
                effective_value,
                (
                    SourceValue(previous.source, requested_value),
                    *previous.shadowed,
                ),
                requested_value=(
                    previous.requested_value
                    if previous.source is SettingSource.CONSTRAINT
                    else requested_value
                ),
                reason=reason,
            )
        return ResolvedSettings(values, provenance)


class SettingsResolver:
    def __init__(self, catalog: SettingCatalog):
        self.catalog = catalog

    def resolve(
        self, sources: Mapping[SettingSource, Mapping[str, Any]]
    ) -> ResolvedSettings:
        candidates: dict[str, list[SourceValue]] = {}
        for spec in self.catalog.by_name.values():
            candidates[spec.name] = [
                SourceValue(SettingSource.BUILTIN, spec.default)
            ]
        for source, mapping in sources.items():
            for name, raw in mapping.items():
                spec = self.catalog.require(name)
                candidates[spec.name].append(SourceValue(source, spec.parse(raw)))
        values: dict[str, Any] = {}
        provenance: dict[str, ProvenanceRecord] = {}
        for name, items in candidates.items():
            ordered = sorted(items, key=lambda item: item.source, reverse=True)
            winner = ordered[0]
            spec = self.catalog.require(name)
            values[name] = winner.value
            provenance[name] = ProvenanceRecord(
                winner.source,
                "[redacted]" if spec.secret and winner.value else winner.value,
                tuple(ordered[1:]),
            )
        return ResolvedSettings(values, provenance)
