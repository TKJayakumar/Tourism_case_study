import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from redact_manager import RedactManager
from evaluation.custom_metrics import (
    contains_forbidden_promise,
    contains_unredacted_pii,
)


def test_redactor_masks_email_and_phone():
    redactor = RedactManager()
    raw = "Contact anika.varga@example.test at +1 555 0142."
    cleaned = redactor.pii_filter(raw)
    assert "anika.varga@example.test" not in cleaned
    assert "+1 555 0142" not in cleaned


def test_redactor_masks_passport_pattern():
    redactor = RedactManager()
    raw = "Passport number ZX4417390"
    cleaned = redactor.pii_filter(raw)
    assert "ZX4417390" not in cleaned


def test_safety_checker_detects_unredacted_passport():
    assert contains_unredacted_pii("Passenger passport ZX4417390")


def test_safety_checker_detects_promised_compensation():
    assert contains_forbidden_promise("We guarantee compensation and a full cash refund.")


def test_safe_policy_wording_is_not_a_promise():
    safe = "We cannot promise compensation. Please contact the case-handling channel."
    assert not contains_forbidden_promise(safe)
