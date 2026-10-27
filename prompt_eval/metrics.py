"""Cheap, deterministic metrics computed from each model response."""

from __future__ import annotations

import re

# Split on sentence-ending punctuation followed by whitespace (a decimal like 3.5 is not
# split), or on newlines so that bullet lists count as several "sentences".
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


def count_words(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return len([p for p in _SENT_SPLIT.split(text) if p and p.strip()])


def cost_usd(input_tokens: int, output_tokens: int, price_in_per_m: float, price_out_per_m: float) -> float:
    """Reasoning tokens are included in output_tokens and billed at the output rate."""
    return (input_tokens * price_in_per_m + output_tokens * price_out_per_m) / 1_000_000
