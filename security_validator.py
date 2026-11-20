
import re


class SecurityValidator:

    def __init__(self):

        self.max_input_length = 2000

        self.unsafe_output_patterns = [

            r"\b(?:we|meridian air)\s+(?:will|can)\s+"
            r"(?:provide|pay|give|issue|guarantee)\s+"
            r"(?:you\s+)?(?:a\s+)?(?:full\s+)?"
            r"(?:cash\s+)?refund\b",

            r"\b(?:you are|you're)\s+guaranteed\s+"
            r"(?:to receive\s+)?(?:a\s+)?"
            r"(?:full\s+)?refund\b",

            r"\b(?:USD\s*)?\$?\d[\d,]*\s+"
            r"compensation\s+(?:is\s+)?guaranteed\b",

            r"\b(?:we|meridian air)\s+(?:will|can)\s+"
            r"(?:pay|provide|offer)\s+"
            r"(?:you\s+)?(?:USD\s*)?\$?\d[\d,]*\s+"
            r"compensation\b"
        ]

    # INPUT VALIDATION

    def validate_input(self, question):

        if not isinstance(question, str):

            return {
                "is_valid": False,
                "reason": "Invalid input type."
            }

        question = question.strip()

        if not question:

            return {
                "is_valid": False,
                "reason": "Question cannot be empty."
            }

        if len(question) > self.max_input_length:

            return {
                "is_valid": False,
                "reason": "Question exceeds the maximum allowed length."
            }

        return {
            "is_valid": True,
            "reason": None
        }

    # OUTPUT VALIDATION

    def validate_output(self, answer):

        if not isinstance(answer, str) or not answer.strip():

            return {
                "is_valid": False,
                "reason": "Empty or invalid response."
            }

        for pattern in self.unsafe_output_patterns:

            if re.search(pattern, answer, re.IGNORECASE):

                return {
                    "is_valid": False,
                    "reason": "Potentially unauthorized refund or compensation promise."
                }

        return {
            "is_valid": True,
            "reason": None
        }
