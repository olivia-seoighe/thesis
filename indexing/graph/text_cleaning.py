from __future__ import annotations

import re


def clean_graph_text(value: str) -> str:
    cleaned = value.replace("**", "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\(L\d+(?:-\d+)?\)", "", cleaned).strip()
    cleaned = cleaned.rstrip(".,;:")
    return cleaned

