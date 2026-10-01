import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from readme_blocks import p_text, sync_readme_block  # noqa: E402

README = """intro stays
<!-- FINDINGS:START (generated) -->
old line
<!-- FINDINGS:END -->
outro stays
"""


def test_block_body_is_replaced_and_everything_outside_is_kept(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(README)
    assert sync_readme_block("FINDINGS", ["new line"], path)
    text = path.read_text()
    assert "new line" in text and "old line" not in text
    assert text.startswith("intro stays\n<!-- FINDINGS:START (generated) -->")
    assert text.endswith("<!-- FINDINGS:END -->\noutro stays\n")


def test_rewriting_twice_is_idempotent(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(README)
    sync_readme_block("FINDINGS", ["a", "", "b"], path)
    once = path.read_text()
    sync_readme_block("FINDINGS", ["a", "", "b"], path)
    assert path.read_text() == once


def test_missing_markers_leave_the_file_untouched(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(README)
    assert not sync_readme_block("CLASSIFICATION", ["x"], path)
    assert path.read_text() == README


def test_tiny_p_values_are_not_printed_as_zero():
    assert p_text(3.2e-22) == "p = 3.2e-22"
    assert p_text(0.0251) == "p = 0.025"
    assert p_text(0.0) == "p ≈ 0"
