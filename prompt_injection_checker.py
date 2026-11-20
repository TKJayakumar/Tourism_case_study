
import re


class PromptInjectionChecker:

    def __init__(self):

        
        self.suspicious_patterns = [
            r"ignore\s+(all\s+)?(your\s+)?(previous|earlier|above)\s+instructions",
            r"disregard\s+(your\s+)?(safety rules|instructions)",
            r"ignore\s+your\s+system\s+prompt",
            r"override\s+.*instructions",
            r"reveal\s+.*(password|passport|personal information)",
            r"do\s+not\s+mention\s+.*",
            r"NOTE\s+FOR\s+(THE\s+)?(AI\s+)?ASSISTANT",
            r"you\s+are\s+now\s+.*",
            r"act\s+as\s+.*instead",
            r"bypass\s+.*(security|rules|restrictions)"
        ]


    def check_text(self, text):

        if not text:
            return {
                "is_suspicious": False,
                "matched_patterns": []
            }

        matched_patterns = []

        for pattern in self.suspicious_patterns:

            if re.search(pattern, text, re.IGNORECASE):
                matched_patterns.append(pattern)

        return {
            "is_suspicious": len(matched_patterns) > 0,
            "matched_patterns": matched_patterns
        }
