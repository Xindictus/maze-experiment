from pydantic import ValidationInfo
from urllib.parse import urlparse


def must_be_power_of_two(v: int, info: ValidationInfo) -> int:
    if v & (v - 1) != 0:
        raise ValueError(
            f"[{info.field_name}]: Must be a power of 2 (got {v})"
        )
    return v


def must_be_valid_url(v: str, info: ValidationInfo) -> str:
    result = urlparse(v)

    if not all([result.scheme, result.netloc]):
        raise ValueError(
            f"[{info.field_name}]: Must be a valid URL (got {v})"
        )

    return v
