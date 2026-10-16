"""Shared validation for experiment and LLM configuration values."""

import math


def require_integer(name: str, value: object, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def require_number(
    name: str, value: object, *, minimum: float, exclusive: bool = False
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (exclusive and value == minimum):
        operator = ">" if exclusive else ">="
        raise ValueError(f"{name} must be {operator} {minimum}")
