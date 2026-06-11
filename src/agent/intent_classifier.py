"""
Intent Classifier — LLM-based query intent detection.

Classifies user queries into one or more of the supported HR intents.
Called after the Safety Shield has cleared the query.

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
from src.utils.llm_factory import get_llm
import re
from typing import List

# NOTE: If you add a new intent here, also add it to the VALID_INTENTS set
# below and update classify_intent()'s fallback logic.
INTENT_PROMPT = PromptTemplate.from_template("""
You are an HR assistant intent classifier.
Classify the user query with history into ONE OR MORE of these intents:
- LEAVE_QUERY: questions about leave, holidays, sick days, maternity, paternity, compensatory leave, comp-off, CL
- DISCIPLINARY: questions about violations, warnings, termination, misconduct
- RESIGNATION: questions about notice period, resignation process, exit
- SHIFT_QUERY: questions about shift timings, night shift, weekend shift allowances
- GREETING: strictly greetings, farewells, and small talk (e.g. "hello", "how are you?", "goodbye")
- OUT_OF_SCOPE: anything not related to HR policies

Rules:
- Return comma-separated labels if multiple intents apply
- If OUT_OF_SCOPE, return only OUT_OF_SCOPE
- No explanation, only labels

Conversation History: {history}
Query: {query}
Respond with intent labels only. No examples, no extra text.
Intents:""")

# Allowlist of valid intent labels. Any LLM output not in this set is dropped.
# If all labels are invalid, falls back to OUT_OF_SCOPE.
_VALID_INTENTS = {
    "LEAVE_QUERY",
    "DISCIPLINARY",
    "RESIGNATION",
    "OUT_OF_SCOPE",
    "SHIFT_QUERY",
    "GREETING",
}


def classify_intent(query: str, history_str: str = "") -> List[str]:
    """
    Classify query into one or more intent labels using the LLM.

    Args:
        query:       Sanitized user query (PII already redacted by shield).
        history_str: Conversation history as plain text for context.
                     Pass empty string if no history.

    Returns:
        List of valid intent label strings. Never empty — falls back to
        ["OUT_OF_SCOPE"] if LLM returns unrecognised labels.

    Note:
        history_str was previously typed as Optional[str] but the prompt
        always expects a string. Callers should pass "" not None.
        Fixed: changed default and guard to use "" consistently.
    """
    llm = get_llm()
    chain = INTENT_PROMPT | llm | StrOutputParser()

    # history_str guaranteed str here; None guard kept for safety
    result = chain.invoke({"query": query, "history": history_str})

    # Strip backticks and whitespace; LLM sometimes wraps output in ` `
    intents = [i.strip().strip("`").upper() for i in re.split(r"[,|]", result)]

    valid_intents = [i for i in intents if i in _VALID_INTENTS]

    # Debug trace — remove or guard with a DEBUG flag before production
    print(f"[IntentClassifier] Raw: {result!r}  →  Parsed: {valid_intents}")

    # Fallback: if LLM returned nothing recognisable, treat as out-of-scope
    return valid_intents or ["OUT_OF_SCOPE"]
