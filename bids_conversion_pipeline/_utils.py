from typing import Literal


def _get_constant(
    object: dict[str, list[str]] | dict[str, int | float],
    dataset: Literal["mph", "naag"],
    cohort: Literal["kids", "adults"],
) -> list[str] | dict[str, int]:
    constant = object[dataset]
    if dataset == "mph":
        constant = constant[cohort]

    return constant
