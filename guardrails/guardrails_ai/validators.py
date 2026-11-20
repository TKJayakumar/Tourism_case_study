
import re
from typing import Any, Dict

from guardrails.validator_base import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
    ValidationResult,
)


@register_validator(
    name="meridian/no-unsupported-promises",
    data_type="string",
)
class NoUnsupportedPromises(Validator):

    def validate(
        self,
        value: Any,
        metadata: Dict,
    ) -> ValidationResult:

        patterns = [
            r"\bguarantee(?:s|d)?\s+(?:you\s+)?(?:a\s+)?(?:full\s+)?refund\b",
            r"\bwill\s+(?:definitely\s+)?(?:receive|get|pay|provide)\b.{0,60}\b(?:refund|compensation|payout)\b",
            r"\b(?:definitely|certainly)\s+(?:receive|get)\b.{0,60}\b(?:refund|compensation)\b",
            r"\bguarantee(?:s|d)?\s+(?:you\s+)?compensation\b",
            r"\b(?:we|meridian air)\s+(?:promise|guarantee|guarantees)\b.{0,60}\b(?:refund|compensation|payout)\b",
        ]

        text = str(value)

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return FailResult(
                    error_message=(
                        "Unsupported refund or compensation promise detected."
                    )
                )

        return PassResult()
