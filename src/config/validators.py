from pydantic import ValidationInfo


def must_be_power_of_two(v: int, info: ValidationInfo) -> int:
    if v & (v - 1) != 0:
        raise ValueError(
            f"[{info.field_name}]: Must be a power of 2 (got {v})"
        )
    return v
