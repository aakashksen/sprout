"""Identification contract shared by the chatbot, the app, and the real identifier."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

Status = Literal["answered", "unsure", "not_plant", "invalid"]
STATUSES: tuple[str, ...] = ("answered", "unsure", "not_plant", "invalid")


@dataclass(frozen=True)
class Candidate:
    label: str
    confidence: float


@dataclass(frozen=True)
class Identification:
    status: Status
    top: Candidate | None = None
    alternatives: list[Candidate] = field(default_factory=list)
    guidance: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> Identification:
        d = json.loads(text)
        top = Candidate(**d["top"]) if d["top"] else None
        return cls(d["status"], top, [Candidate(**a) for a in d["alternatives"]], d["guidance"])


class IdentifierProtocol(Protocol):
    def identify(self, image_bytes: bytes) -> Identification: ...


class FakeIdentifier:
    """Returns the given results in order, repeating the last one. Never valid in a real run."""

    def __init__(self, results: list[Identification]) -> None:
        if not results:
            raise ValueError("FakeIdentifier needs at least one result")
        self._results = list(results)
        self._i = 0

    def identify(self, image_bytes: bytes) -> Identification:
        r = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return r
