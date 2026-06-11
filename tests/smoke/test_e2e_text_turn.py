"""
placeholder
"""

import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.skip(
    reason="no --text-only path on this branch: pipeline consumes audio via STT only, and "
    "building a text-only transcript-in/text-out mode is out of scope for this task."
)
def test_e2e_text_turn():  # pragma: no cover - intentionally skipped
    raise AssertionError("unreachable: skipped above")
