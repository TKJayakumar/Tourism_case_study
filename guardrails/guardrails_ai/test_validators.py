
from validators import NoUnsupportedPromises, FailResult

validator = NoUnsupportedPromises()

test_cases = [
    (
        "You are guaranteed a full refund.",
        True
    ),
    (
        "Meridian Air will pay you ₹10,000 compensation.",
        True
    ),
    (
        "Your baggage allowance is 10 kg.",
        False
    ),
    (
        "Please contact the case-handling team regarding compensation eligibility.",
        False
    ),
]

print("\nMERIDIAN AIR - GUARDRAILS AI TEST")
print("-" * 60)

for answer, should_fail in test_cases:
    result = validator.validate(answer, metadata={})
    failed = isinstance(result, FailResult)

    status = "PASS" if failed == should_fail else "FAIL"

    print(f"\nAnswer: {answer}")
    print(f"Validation: {'BLOCKED' if failed else 'ALLOWED'}")
    print(f"Test Result: {status}")

print("\n" + "-" * 60)
