from __future__ import annotations

import re


def extract_feature_flags_from_csharp(source_code: str) -> list[str]:
    if not source_code.strip():
        return []

    flags: set[str] = set()
    receivers: set[str] = set()

    for match in re.finditer(r"\bI?FeatureFlags\s+(\w+)\b", source_code):
        receivers.add(match.group(1))

    for receiver in receivers:
        pattern = rf"\b{re.escape(receiver)}\.(\w+)\b"
        for match in re.finditer(pattern, source_code):
            flags.add(match.group(1))

    for match in re.finditer(r"\bIFeatureFlags\.(\w+)\b", source_code):
        flags.add(match.group(1))

    interface_match = re.search(r"interface\s+IFeatureFlags\b([\s\S]*?)\}", source_code)
    if interface_match:
        for prop_match in re.finditer(
            r"\b(?:bool|Boolean)\??\s+(\w+)\s*\{",
            interface_match.group(1),
        ):
            flags.add(prop_match.group(1))

    return sorted(flags)

