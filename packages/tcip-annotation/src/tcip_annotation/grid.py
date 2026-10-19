"""Spreadsheet-style names for reference-grid cells, and the lookup of a named cell's rect."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    """One grid cell: its name plus its half-open native-pixel rect.

    The rect is ``x0 <= x < x1``, ``y0 <= y < y1``.
    ``col``/``row`` are 0-based grid coordinates; ``name`` is ``column_label(col)`` + 1-based row.
    """

    name: str
    col: int
    row: int
    x0: int
    y0: int
    x1: int
    y1: int


def column_label(index: int) -> str:
    """Return the spreadsheet-style label for 0-based column ``index``: A-Z, then AA, AB, ....

    Bijective base-26. A negative ``index`` raises ``ValueError``.
    """
    if index < 0:
        raise ValueError(f"Column index must be non-negative, got {index}")
    label = ""
    n = index + 1
    while n:
        n, rem = divmod(n - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def column_index(label: str) -> int:
    """Return the 0-based column index of a spreadsheet-style ``label``.

    A label that is empty or not all letters A-Z raises ``ValueError``.
    """
    if not label or not all("A" <= ch <= "Z" for ch in label):
        raise ValueError(
            f"Invalid column label: {label!r}. Expected letters A-Z."
        )
    n = 0
    for ch in label:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def cell_fields(cell: Cell) -> tuple[str, float, float, float, float]:
    """Return ``(name, x0, y0, x1, y1)`` from one grid cell, its rect as floats."""
    return (
        cell.name,
        float(cell.x0),
        float(cell.y0),
        float(cell.x1),
        float(cell.y1),
    )


def grid_to_rect(
    cell: str, cells: list[Cell]
) -> tuple[float, float, float, float]:
    """Resolve a grid cell reference (e.g. 'B3') to the named cell's half-open native-pixel rect.

    The rect is ``(x0, y0, x1, y1)``. ``cells`` is the caller's own cell list. Matching is
    case-insensitive and whitespace-stripped. A reference that is not a cell name at all raises
    ValueError naming the expected format; one that names no cell in this grid raises ValueError
    naming the grid's valid range.
    """
    wanted = cell.strip().upper()
    if re.fullmatch(r"[A-Z]+[0-9]+", wanted) is None:
        raise ValueError(
            f"Invalid cell reference: {cell!r}. Expected format like 'B3'."
        )
    if not cells:
        raise ValueError(
            "cells is empty: there is no grid to resolve a cell name against"
        )

    max_col = max_row = -1
    for c in cells:
        name, x0, y0, x1, y1 = cell_fields(c)
        if name.strip().upper() == wanted:
            return x0, y0, x1, y1
        parsed = re.fullmatch(r"([A-Z]+)([0-9]+)", name.strip().upper())
        if parsed is not None:
            max_col = max(max_col, column_index(parsed.group(1)))
            max_row = max(max_row, int(parsed.group(2)))
    hint = (
        f" Use A1 through {column_label(max_col)}{max_row}."
        if max_col >= 0
        else ""
    )
    raise ValueError(f"Cell '{wanted}' is not in this grid.{hint}")
