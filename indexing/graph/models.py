from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Triple:
    subject: str
    subject_label: str
    predicate: str
    object: str
    object_label: str
    properties: dict = field(default_factory=dict)

