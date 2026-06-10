"""
Safety Shield — Pre-LLM input validation layer.

Two checks, in order:
  1. Injection detection  — phrase-matching + fuzzy matching (no LLM)
  2. PII detection        — regex-based redaction (India-aware)

Runs BEFORE any LLM call. Fast, deterministic, no API cost.

Design notes:
- INJECTION_PHRASES is ordered roughly by attack severity.
- Fuzzy threshold (0.82) balances false positives vs missed typo attacks;
  lower it to ~0.75 for tighter security, raise to ~0.88 to reduce FP.
- PII_PATTERNS covers Indian identifiers (Aadhaar, PAN, Indian mobile).
  Add international patterns (SSN, NINO, etc.) if expanding beyond India.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


# ======== PII Patterns (India-aware) ========
# Each key becomes the redaction label: e.g. "[EMAIL REDACTED]"
PII_PATTERNS = {
    "email":   r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone":   r"\b[6-9]\d{9}\b",       # Indian mobile: starts 6-9, 10 digits
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan":     r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
}

# ======== Injection Phrases ========
# Rule: if ANY phrase is found (exact or fuzzy), block the query immediately.
# Additions should be tested against test_safety_shield.py before merging.
INJECTION_PHRASES = [
    # Direct instruction override
    "ignore previous",
    "ignore all instructions",
    "disregard",
    "forget your instructions",

    # Role hijacking
    "you are now a",
    "you are now an",
    "act as a ",
    "act as an ",
    "pretend you are",
    "assume the identity",
    "assume the role",

    # Jailbreak persona framing
    "you are an unrestricted",
    "you are a unrestricted",
    "you are free to",
    "you have no restrictions",
    "you have no limits",
    "without restrictions",
    "unrestricted ai",
    "unfiltered ai",
    "do anything now",
    "dan mode",
    "developer mode",
    "get me the credentials",
    "get me the password",
    "give me the credentials",
    "give me access",

    # System prompt / config exfiltration
    "system prompt",
    "output your complete",
    "output your config",
    "print your",
    "reveal your instructions",
    "update your internal",
    "update your state",
    "clear your instruction",
    "clear your memory",

    # Common jailbreak keywords
    "jailbreak",
    "bypass",
    "####",
    "new instructions:",

    # Encoded / obfuscated attacks
    "base64",
    "execute it immediately",
    "decode and run",

    # Typo-resilient variants (caught by fuzzy matching)
    "forget yr",
    "forget your",
    "forg3t",
    "ignor3",

    # Auditor / stress-test framing
    "stress-test your system",
    "verify compliance",
    "security auditor",
]

# Fuzzy match threshold — 0.82 tested against known attack corpus.
# Raise toward 1.0 to reduce false positives; lower toward 0.70 to catch
# more misspelled variants.
_FUZZY_THRESHOLD = 0.82


@dataclass
class ShieldResult:
    """Result returned by run_shield()."""
    is_safe: bool
    threat_type: str | None    # "PII" | "INJECTION" | None
    detail: str | None         # human-readable description of what was flagged
    sanitized_query: str       # original query, or PII-redacted version


def _fuzzy_injection_check(query: str) -> str | None:
    """
    Slide a sliding window over query tokens and fuzzy-match each window
    against known injection phrases.

    Returns the matched phrase string, or None if no match found.

    Complexity: O(words × phrases × phrase_length) — acceptable for
    typical chat queries (<200 words). Not suitable for bulk processing.
    """
    words = query.lower().split()
    for phrase in INJECTION_PHRASES:
        phrase_words = phrase.split()
        window_size = len(phrase_words)
        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i: i + window_size])
            ratio = SequenceMatcher(None, window, phrase).ratio()
            if ratio >= _FUZZY_THRESHOLD:
                return phrase
    return None


def run_shield(query: str) -> ShieldResult:
    """
    Run the full safety check on a raw user query.

    Injection check runs first (fast, no LLM).
    PII check runs second and redacts matches before returning.

    Args:
        query: Raw user input as received from the UI.

    Returns:
        ShieldResult — callers must check is_safe before proceeding.
        If is_safe is False, do NOT forward the query to the LLM.
        If threat_type is "PII", sanitized_query has PII redacted and
        MAY be forwarded (HR agent blocks it, but it's available if
        a future caller wants to continue with redacted input).
    """
    lower = query.lower()

    # ======== Step 1: Injection — exact match (fast path) ========
    flagged_phrase = next(
        (p for p in INJECTION_PHRASES if p in lower), None
    )

    # ======== Step 2: Injection — fuzzy match (catches typos) ========
    if not flagged_phrase:
        flagged_phrase = _fuzzy_injection_check(query)

    if flagged_phrase:
        return ShieldResult(
            is_safe=False,
            threat_type="INJECTION",
            detail=f"Flagged phrase: '{flagged_phrase}'",
            sanitized_query=query,  # not redacted — query is blocked entirely
        )

    # ======== STEP 3: PII — regex detection + redaction ========
    sanitized = query
    found_pii: list[str] = []

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, sanitized)
        if matches:
            found_pii.append(pii_type.upper())
            sanitized = re.sub(
                pattern,
                f"[{pii_type.upper()} REDACTED]",
                sanitized,
            )

    if found_pii:
        return ShieldResult(
            is_safe=True,          # ← changed: allow through after redaction
            threat_type="PII",
            detail=f"Detected and redacted: {', '.join(found_pii)}",
            sanitized_query=sanitized,
        )

    return ShieldResult(
        is_safe=True,
        threat_type=None,
        detail=None,
        sanitized_query=query,
    )
