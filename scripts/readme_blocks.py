"""Rewrite generated blocks of README.md so its figures cannot drift from the reports.

A block sits between `<!-- NAME:START ... -->` and `<!-- NAME:END -->`. Everything inside
is owned by the script that names it and is replaced on every run, so hand-edits there are
lost; edit the generator instead. Prose outside a block is never touched, which is why it
should carry no figure that changes when the snapshot refreshes.
"""

from __future__ import annotations

from pathlib import Path

README_PATH = Path("README.md")


def sync_readme_block(name: str, lines: list[str], path: Path = README_PATH) -> bool:
    """Replace the body of block `name` with `lines`. False when the markers are missing."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    start, end = text.find(f"<!-- {name}:START"), text.find(f"<!-- {name}:END -->")
    if start == -1 or end == -1:
        return False
    marker_end = text.find("-->", start) + len("-->")
    block = "\n".join(["", *lines, ""])
    path.write_text(text[:marker_end] + block + text[end:], encoding="utf-8")
    return True


def p_text(value: float) -> str:
    """A p-value for prose. Exact zero comes from a statistic that overflowed, not from data."""
    return "p ≈ 0" if value == 0 else f"p = {value:.1e}" if value < 1e-3 else f"p = {value:.3f}"
