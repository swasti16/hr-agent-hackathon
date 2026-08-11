"""
Intent Classifier — LLM-based query intent detection, query resolution,
and topic-switch detection in a single call.

Classifies user queries into one or more supported HR intents, rewrites
follow-up questions into standalone queries for retrieval, and signals
whether the current query continues the prior topic or starts a new one.

Consolidated into one call (previously would have needed a separate
embedding-similarity gate for topic-switch — see git history, that
approach was validated against real query pairs and rejected: short
elliptical follow-ups like "What about paternity?" don't carry enough
signal for cosine similarity to separate same-topic from switched-topic
reliably). LLM-based judgment handles short-text reference resolution
better than embeddings do.

Supported intents:
    LEAVE_QUERY    — leave, holidays, sick days, maternity, paternity
    DISCIPLINARY   — violations, warnings, termination, misconduct
    RESIGNATION    — notice period, resignation process, exit
    SHIFT_QUERY    — shift timings, night shift, weekend shift allowances
    GREETING       — greetings, farewells, small talk
    OUT_OF_SCOPE   — anything unrelated to HR policies
"""

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm_factory import get_intent_llm
import logging
import json
from typing import TypedDict, List

logger = logging.getLogger(__name__)

# NOTE: If you add a new intent here, also add it to the VALID_INTENTS set
# below and update classify_intent()'s fallback logic.
INTENT_PROMPT = PromptTemplate.from_template("""
You are an HR assistant query analyzer. Given conversation history and the
current query, produce a JSON object with FOUR fields.

<INTENTS>
- LEAVE_QUERY: leave, holidays, sick days, maternity, paternity, comp-off, CL
- DISCIPLINARY: violations, warnings, termination, misconduct
- RESIGNATION: notice period, resignation process, exit
- SHIFT_QUERY: shift timings, night shift, weekend shift allowances
- GREETING: strictly greetings, farewells, small talk
- OUT_OF_SCOPE: anything not related to HR policies
</INTENTS>

<FIELD_INSTRUCTIONS>
1. "intents": array of one or more labels from <INTENTS>. If OUT_OF_SCOPE
   applies, return ONLY ["OUT_OF_SCOPE"].

2. "resolved_query": Rewrite the CURRENT QUERY into a fully standalone
   question by resolving pronouns, ellipsis, and implicit references
   using the conversation history. This will be used for document
   retrieval, so it must contain the actual topic keywords.
   - If the current query is already standalone (no pronouns/ellipsis
     referring to history), return it unchanged.
   - If there is no history, return the current query unchanged.
   Examples:
     History: "user: What is the maternity leave policy?"
     Query: "What about paternity?"
     resolved_query: "What is the paternity leave policy?"

     History: "user: What is the notice period?"
     Query: "Can I take leaves during it?"
     resolved_query: "Can I take leaves during the notice period?"

3. "topic_continues": true if the current query is a follow-up to the
   SAME subject as the most recent history turn (even if asking about a
   related-but-different policy, e.g. maternity -> paternity counts as
   continuing). false if the current query starts an unrelated subject,
   or if there is no history yet.

4. "reason": one line explaining the resolved_query and topic_continues
   decisions.
</FIELD_INSTRUCTIONS>

Conversation History: {history}
Current Query: {query}

Respond ONLY with valid JSON, no extra text, no markdown fences:
{{
  "intents": ["..."],
  "resolved_query": "...",
  "topic_continues": true/false,
  "reason": "..."
}}
""")

# Allowlist of valid intent labels. Any LLM output not in this set is dropped.
_VALID_INTENTS = {
    "LEAVE_QUERY",
    "DISCIPLINARY",
    "RESIGNATION",
    "OUT_OF_SCOPE",
    "SHIFT_QUERY",
    "GREETING",
}


class IntentResult(TypedDict):
    intents: List[str]
    resolved_query: str
    topic_continues: bool
    reason: str


def _fallback_result(query: str, raw_output: str = "") -> IntentResult:
    """
    Conservative fallback when JSON parsing fails: treat as out-of-scope,
    keep the query unresolved (retrieval will just use the raw query,
    same behavior as before this change), and assume topic continues
    (safer than dropping potentially-relevant history on a parse error).
    """
    logger.warning(
        "[IntentClassifier] JSON parse failed. Raw output: %r", raw_output
    )
    return {
        "intents": ["OUT_OF_SCOPE"],
        "resolved_query": query,
        "topic_continues": True,
        "reason": "Classification failed — fallback applied",
    }


def classify_intent(query: str, history_str: str = "") -> IntentResult:
    """
    Classify query intent, resolve it into a standalone query for
    retrieval, and detect whether the topic continues from history.

    Args:
        query:       Sanitized user query (PII already redacted by shield).
        history_str: Conversation history as plain text. Pass "" if none.

    Returns:
        IntentResult dict with keys: intents, resolved_query,
        topic_continues, reason. Falls back to _fallback_result() on
        JSON parse failure — never raises.
    """
    llm = get_intent_llm()
    chain = INTENT_PROMPT | llm | StrOutputParser()

    result = chain.invoke({"query": query, "history": history_str})

    try:
        cleaned = result.strip()
        cleaned = (
            cleaned.removesuffix("```")
            .removeprefix("```json")
            .removeprefix("```")
            .strip()
        )
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return _fallback_result(query, result)

    raw_intents = parsed.get("intents", [])
    valid_intents = [
        i.strip().upper() for i in raw_intents
        if isinstance(i, str) and i.strip().upper() in _VALID_INTENTS
    ]
    if not valid_intents:
        valid_intents = ["OUT_OF_SCOPE"]

    resolved_query = parsed.get("resolved_query") or query
    topic_continues = bool(parsed.get("topic_continues", True))
    reason = parsed.get("reason", "")

    logger.info(
        "[IntentClassifier] intents=%s resolved_query=%r topic_continues=%s",
        valid_intents, resolved_query, topic_continues,
    )

    return {
        "intents": valid_intents,
        "resolved_query": resolved_query,
        "topic_continues": topic_continues,
        "reason": reason,
    }
