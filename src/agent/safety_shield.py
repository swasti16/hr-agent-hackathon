import re
from dataclasses import dataclass
from difflib import SequenceMatcher


# patterns to detect PII (India-aware) and redact it from the query.
PII_PATTERNS = {
    "email":   r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone":   r"\b[6-9]\d{9}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan":     r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
}

# phrases that indicate an injection attack
INJECTION_PHRASES = ["ignore previous", "ignore all instructions", "disregard",
                     "forget your instructions", "you are now a",
                     "you are now an", "act as a ", "act as an ",
                     "jailbreak", "bypass", "pretend you are",
                     "new instructions:", "system prompt", "####",

                     # Typo-resilient (common misspellings)
                     "forget yr", "forget your", "forg3t", "ignor3",

                     # Semantic / role-hijack patterns
                     "assume the identity", "assume the role",
                     "output your complete", "output your config",
                     "print your", "reveal your instructions",
                     "update your internal", "update your state",
                     "clear your instruction", "clear your memory",

                     # Encoded attack signals
                     "base64", "execute it immediately", "decode and run",

                     # Stress-test / auditor framing
                     "stress-test your system", "verify compliance",
                     "security auditor"]


@dataclass
class ShieldResult:
    is_safe: bool
    threat_type: str | None   # "PII" | "INJECTION" | None
    detail: str | None        # what was flagged
    sanitized_query: str      # original or redacted


FUZZY_THRESHOLD = 0.82  # tweak if too many false positives


def _fuzzy_injection_check(query: str) -> str | None:
    """Slide a window over query words, fuzzy-match against known phrases."""
    words = query.lower().split()
    for phrase in INJECTION_PHRASES:
        phrase_words = phrase.split()
        window_size = len(phrase_words)
        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i:i + window_size])
            ratio = SequenceMatcher(None, window, phrase).ratio()
            if ratio >= FUZZY_THRESHOLD:
                return phrase
    return None


def run_shield(query: str) -> ShieldResult:
    """Single entry point. Returns ShieldResult."""

    # Injection check (fast, no LLM)
    # REPLACE the injection check section with:
    lower = query.lower()

    # Exact match first (fast)
    flagged_phrase = next((p for p in INJECTION_PHRASES if p in lower), None)

    # Fuzzy match if exact misses
    if not flagged_phrase:
        flagged_phrase = _fuzzy_injection_check(query)

    if flagged_phrase:
        return ShieldResult(
            is_safe=False,
            threat_type="INJECTION",
            detail=f"Flagged phrase: '{flagged_phrase}'",
            sanitized_query=query,
        )

    # PII check + redact
    sanitized = query
    found_pii = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, sanitized)
        if matches:
            found_pii.append(pii_type.upper())
            sanitized = re.sub(pattern, f"[{pii_type.upper()} REDACTED]", sanitized)

    if found_pii:
        return ShieldResult(
            is_safe=False,
            threat_type="PII",
            detail=f"Detected: {', '.join(found_pii)}",
            sanitized_query=sanitized,
        )

    return ShieldResult(is_safe=True, threat_type=None, detail=None, sanitized_query=query)
