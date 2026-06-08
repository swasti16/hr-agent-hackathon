from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm_factory import get_llm

from typing import List

INTENT_PROMPT = PromptTemplate.from_template("""
You are an HR assistant intent classifier.
Classify the user query with history into ONE OR MORE of these intents:
- LEAVE_QUERY: questions about leave, holidays, sick days, maternity, paternity
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
Intents:""")


def classify_intent(query: str, history_str: str = None) -> List[str]:
    llm = get_llm()
    chain = INTENT_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"query": query, "history": history_str})
    intents = [i.strip().strip("`").upper() for i in result.split(",")]
    valid = {"LEAVE_QUERY", "DISCIPLINARY", "RESIGNATION", "OUT_OF_SCOPE",
             "SHIFT_QUERY", "GREETING"}
    print(f"Classified intents: {intents}")
    return [i for i in intents if i in valid] or ["OUT_OF_SCOPE"]
