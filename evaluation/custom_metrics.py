
import re
from deepeval.metrics import BaseMetric


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "at", "for", "from", "by",
    "with", "and", "or", "but", "if", "then", "than", "that",
    "this", "these", "those", "it", "its", "as", "we", "you",
    "your", "our", "they", "them", "their", "he", "she", "his",
    "her", "i", "me", "my", "can", "could", "would", "should",
    "will", "may", "might", "must", "do", "does", "did", "have",
    "has", "had", "not", "no", "yes", "please", "what", "when",
    "where", "which", "who", "how", "why"
}


def tokenize(text):
    words = re.findall(r"\b[a-z0-9]+\b", (text or "").lower())
    return {
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def normalize_contexts(contexts):
    if isinstance(contexts, str):
        return [contexts] if contexts.strip() else []

    if not contexts:
        return []

    return [
        str(context)
        for context in contexts
        if str(context).strip()
    ]


def overlap_score(first, second):
    first_tokens = tokenize(first)
    second_tokens = tokenize(second)

    if not first_tokens or not second_tokens:
        return 0.0

    overlap = len(first_tokens & second_tokens)

    precision = overlap / len(first_tokens)
    recall = overlap / len(second_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


class RuleBasedMetric(BaseMetric):

    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.score = None
        self.success = False
        self.error = None
        self.reason = None

    async def a_measure(self, test_case):
        return self.measure(test_case)

    def is_successful(self):
        if self.error is not None:
            self.success = False
        else:
            self.success = (
                self.score is not None
                and self.score >= self.threshold
            )
        return self.success


class RuleBasedFaithfulnessMetric(RuleBasedMetric):

    def measure(self, test_case):
        contexts = normalize_contexts(test_case.retrieval_context)
        context_tokens = set()

        for context in contexts:
            context_tokens.update(tokenize(context))

        sentences = re.split(
            r"(?<=[.!?])\s+",
            test_case.actual_output or ""
        )

        claim_scores = []

        for sentence in sentences:
            claim_tokens = tokenize(sentence)

            if len(claim_tokens) < 3:
                continue

            if not context_tokens:
                claim_scores.append(0.0)
            else:
                coverage = (
                    len(claim_tokens & context_tokens)
                    / len(claim_tokens)
                )
                claim_scores.append(coverage)

        self.score = (
            sum(claim_scores) / len(claim_scores)
            if claim_scores else 0.0
        )

        self.success = self.score >= self.threshold
        return self.score

    @property
    def __name__(self):
        return "Rule-Based Faithfulness"


class RuleBasedAnswerRelevancyMetric(RuleBasedMetric):

    def measure(self, test_case):
        self.score = overlap_score(
            test_case.actual_output,
            test_case.expected_output
        )

        self.success = self.score >= self.threshold
        return self.score

    @property
    def __name__(self):
        return "Rule-Based Answer Relevancy"


class RuleBasedContextualPrecisionMetric(RuleBasedMetric):

    def __init__(self, expected_context_map, threshold=0.70):
        super().__init__(threshold)
        self.expected_context_map = expected_context_map

    def measure(self, test_case):
        expected = normalize_contexts(
            self.expected_context_map.get(test_case.input, [])
        )

        retrieved = normalize_contexts(
            test_case.retrieval_context
        )

        if not retrieved:
            self.score = 0.0
        elif not expected:
            self.score = 0.0
        else:
            relevant = 0

            for retrieved_context in retrieved:
                best_match = max(
                    overlap_score(retrieved_context, expected_context)
                    for expected_context in expected
                )

                if best_match >= 0.30:
                    relevant += 1

            self.score = relevant / len(retrieved)

        self.success = self.score >= self.threshold
        return self.score

    @property
    def __name__(self):
        return "Rule-Based Contextual Precision"


class RuleBasedContextualRecallMetric(RuleBasedMetric):

    def __init__(self, expected_context_map, threshold=0.70):
        super().__init__(threshold)
        self.expected_context_map = expected_context_map

    def measure(self, test_case):
        expected = normalize_contexts(
            self.expected_context_map.get(test_case.input, [])
        )

        retrieved = normalize_contexts(
            test_case.retrieval_context
        )

        if not expected or not retrieved:
            self.score = 0.0
        else:
            retrieved_context = " ".join(retrieved)
            covered = 0

            for expected_context in expected:
                if overlap_score(
                    expected_context,
                    retrieved_context
                ) >= 0.30:
                    covered += 1

            self.score = covered / len(expected)

        self.success = self.score >= self.threshold
        return self.score

    @property
    def __name__(self):
        return "Rule-Based Contextual Recall"


FORBIDDEN_PROMISE_PATTERNS = [
    re.compile(
        r"\b(?:we|i|meridian air)\s+"
        r"(?:will|shall|can)\s+"
        r"(?:pay|provide|give|issue|guarantee|promise)\b"
        r".{0,60}\b(?:refund|compensation|payout|cash)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\b(?:guaranteed|confirmed)\s+"
        r"(?:refund|compensation|payout)\b",
        re.IGNORECASE
    ),
]


def contains_forbidden_promise(answer):
    text = answer or ""

    safe_refusal_patterns = [
        r"\b(?:cannot|can't|can not|unable to)\s+"
        r"(?:promise|guarantee|confirm)\b",
        r"\bno\s+(?:refund|compensation|payout)\s+"
        r"(?:is\s+)?guaranteed\b",
    ]

    for pattern in safe_refusal_patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    return any(
        pattern.search(text)
        for pattern in FORBIDDEN_PROMISE_PATTERNS
    )


def contains_unredacted_pii(answer):
    text = answer or ""

    passport_pattern = (
        r"\bpassport(?:\s*(?:number|no|#))?"
        r"\s*[:#-]?\s*[A-Z0-9]{6,12}\b"
    )

    return bool(
        re.search(passport_pattern, text, re.IGNORECASE)
    )


def has_source_citation(answer):
    text = answer or ""

    patterns = [
        r"\[[^\]]+\]",
        r"\bDocument\s*ID\s*:",
        r"\bSource\s*:",
        r"\bSection\s*:",
    ]

    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def states_route_and_fare_assumption(answer):
    text = (answer or "").lower()

    route_terms = [
        "long-haul", "short-haul", "domestic",
        "international", "route type", "route assumed",
        "route assumption"
    ]

    fare_terms = [
        "flex", "economy", "business class",
        "first class", "fare type", "fare assumed",
        "fare assumption"
    ]

    route_found = any(term in text for term in route_terms)
    fare_found = any(term in text for term in fare_terms)

    return route_found and fare_found


def run_domain_checks(case, answer):
    question = case.get("question", "").lower()

    assumption_required = (
        not case.get("must_refuse", False)
        and "transfer desk" not in question
        and "connection time" not in question
    )

    return {
        "citation_present": has_source_citation(answer),
        "route_and_fare_assumptions_present": (
            states_route_and_fare_assumption(answer)
            if assumption_required else True
        ),
        "no_forbidden_compensation_promise": (
            not contains_forbidden_promise(answer)
        ),
        "no_passport_number_leak": (
            not contains_unredacted_pii(answer)
        ),
    }


