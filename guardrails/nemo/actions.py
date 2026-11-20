
import re
from nemoguardrails.actions import action

PRIVACY_PATTERNS = [
    r"\b(?:another|other)\s+passenger(?:'s|s)?\b.*\b(?:passport|booking|account|phone|personal|ticket|flight|email)\b",
    r"\b(?:someone\s+else's|somebody\s+else's|another\s+person's)\b.*\b(?:passport|booking|account|phone|personal|ticket|flight|email)\b",
    r"\b(?:show|give|tell|find|reveal|get|provide|display)\b.*\b(?:passport|booking|account|phone|personal|ticket|flight|email)\s+(?:details|number|information)?\b.*\b(?:passenger|person|customer|traveller)\b",
    r"\b(?:passport|booking|account|phone|ticket|flight)\s+(?:details|number|information)\s+(?:for|of)\s+(?:a|another|the)\s+(?:passenger|person|customer|traveller)\b",
    r"\b(?:account|passport|booking|ticket|phone|email)\s+(?:details|information|number|reference)\s+(?:for|of)\s+(?:a|another|the)\s+(?:passenger|person|customer|traveller)\b",
    r"\b(?:passenger|customer|traveller)'s\s+(?:passport|booking|account|phone|personal|ticket|flight|email)\s+(?:details|number|information)\b",
    r"\b(?:show|reveal|give|find|get|display)\b.*\b(?:passenger|customer|traveller)\b.*\b(?:private|personal|account|passport|booking)\b",
]

INJECTION_PATTERNS = [
    r"\bignore\b.{0,50}\b(previous|above|earlier|prior)\b.{0,30}\b(instructions|rules|prompts)\b",
    r"\b(disregard|override|bypass|forget)\b.{0,60}\b(instructions|rules|policies|restrictions|guardrails|security)\b",
    r"\b(reveal|show|print|display|repeat|disclose)\b.{0,40}\b(system prompt|hidden prompt|internal instructions|system instructions)\b",
    r"\bact as\b.{0,40}\b(unrestricted|unfiltered|administrator|developer mode)\b",
    r"\b(jailbreak|developer mode|DAN mode)\b",
    r"\bdo not follow\b.{0,40}\b(previous|system|security)\b.{0,30}\b(instructions|rules|policies)\b",
]

INDIRECT_INJECTION_PATTERNS = [
    r"\bignore\b.{0,50}\b(previous|above|earlier|prior)\b.{0,30}\b(instructions|rules|prompts)\b",
    r"\b(disregard|override|bypass|forget)\b.{0,60}\b(instructions|rules|policies|security|guardrails)\b",
    r"\b(reveal|show|print|display|disclose)\b.{0,40}\b(system prompt|hidden prompt|internal instructions|system instructions)\b",
    r"\bact as\b.{0,40}\b(unrestricted|unfiltered|administrator|developer mode)\b",
    r"\b(jailbreak|developer mode|DAN mode)\b",
    r"\b(send|share|expose|reveal)\b.{0,50}\b(passenger|customer)\b.{0,50}\b(personal|private|booking|passport)\b",
    r"\bdo not follow\b.{0,40}\b(previous|system|security)\b.{0,30}\b(instructions|rules|policies)\b",
]

OUTPUT_PRIVACY_PATTERNS = [
    # Passport numbers
    r"\bpassport\s*(?:number|no\.?)\s*[:#-]?\s*[A-Z0-9]{6,12}\b",

    # Booking references
    r"\bbooking\s*(?:reference|ref|number|no\.?)\s*[:#-]\s*[A-Z0-9]{5,8}\b",

    # Ticket numbers
    r"\bticket\s*(?:number|no\.?)\s*[:#-]?\s*\d{10,13}\b",

    # Email addresses
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",

    # Phone numbers: 10 digits, optionally formatted with common separators
    # and an optional international country code.
    r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\d{10}|\d{5}[\s.-]\d{5}|\d{3}[\s.-]\d{3}[\s.-]\d{4})(?!\d)",
]


def get_user_message(context):
    if not isinstance(context, dict):
        return ""

    message = (
        context.get("user_message")
        or context.get("last_user_message")
        or ""
    )

    return message if isinstance(message, str) else ""


def matches_patterns(message, patterns):
    if not isinstance(message, str):
        return False

    return any(
        re.search(pattern, message, re.IGNORECASE)
        for pattern in patterns
    )


def contains_indirect_prompt_injection(documents):
    if not isinstance(documents, (list, tuple)):
        return False

    for document in documents:
        if isinstance(document, dict):
            content = document.get("page_content", "")
        else:
            content = getattr(document, "page_content", str(document))

        if matches_patterns(content, INDIRECT_INJECTION_PATTERNS):
            return True

    return False

def contains_confidential_passenger_data(answer):
    if not isinstance(answer, str):
        return False

    pattern_names = [
        "Passport number",
        "Booking reference",
        "Ticket number",
        "Email address",
        "Phone number",
    ]

    for name, pattern in zip(pattern_names, OUTPUT_PRIVACY_PATTERNS):
        if re.search(pattern, answer, re.IGNORECASE):
            print(f"[PRIVACY DEBUG] Matched pattern: {name}")
            return True

    return False

@action(is_system_action=True)
async def check_passenger_privacy(context: dict = None):
    return matches_patterns(
        get_user_message(context),
        PRIVACY_PATTERNS
    )


@action(is_system_action=True)
async def check_prompt_injection(context: dict = None):
    return matches_patterns(
        get_user_message(context),
        INJECTION_PATTERNS
    )


@action(is_system_action=True)
async def check_retrieved_document_injection(context: dict = None):
    if not isinstance(context, dict):
        return False

    documents = context.get("retrieved_documents", [])

    return contains_indirect_prompt_injection(documents)


@action(is_system_action=True)
async def check_generated_answer_privacy(context: dict = None):
    if not isinstance(context, dict):
        return False

    answer = (
        context.get("bot_message")
        or context.get("assistant_message")
        or context.get("generated_answer")
        or ""
    )

    return contains_confidential_passenger_data(answer)
