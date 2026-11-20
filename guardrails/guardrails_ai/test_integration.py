
import sys
from pathlib import Path

from guardrails.validator_base import FailResult

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "guardrails" / "guardrails_ai"))

from validators import NoUnsupportedPromises


validator = NoUnsupportedPromises()

test_answers = [
    "Meridian Air guarantees you a full refund.",
    "Your cabin baggage allowance is 10 kg.",
]

fallback = (
    "I can't confirm refund or compensation eligibility from the "
    "available policy information. Please refer this case to "
    "Meridian Air's case-handling team for review."
)

print("\nGUARDRAILS AI - INTEGRATION TEST")
print("-" * 60)

for answer in test_answers:
    result = validator.validate(answer, metadata={})

    if isinstance(result, FailResult):
        final_answer = fallback
        print("\nValidation: BLOCKED")
    else:
        final_answer = answer
        print("\nValidation: APPROVED")

    print("Generated answer:", answer)
    print("Final answer:", final_answer)

print("\n" + "-" * 60)
