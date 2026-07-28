from __future__ import annotations

import re
from decimal import Decimal

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*$", re.IGNORECASE)
_MULTIPLIERS = {
    None: 1,
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}


def parse_size(value: str) -> int:
    match = _SIZE_RE.match(value)
    if match is None:
        raise ValueError(f"invalid size: {value!r}")
    number, suffix = match.groups()
    multiplier = _MULTIPLIERS[suffix.lower() if suffix else None]
    return int(Decimal(number) * multiplier)


def format_gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"
