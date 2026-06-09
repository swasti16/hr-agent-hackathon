"""
HR Agent — Main entry point.

Orchestrates the full pipeline:
  1. Safety Shield (injection + PII detection, no LLM)
  2. Intent Classification (LLM)
  3. RAG retrieval + Reasoning (LLM)

Returns a structured dict consumed by the Gradio UI.
"""

from src.agent.intent_classifier import classify_intent
from src.agent.reasoning_chain import reason_and_respond
from src.hr_rag_pipeline import HRRagPipeline
from src.agent.safety_shield import run_shield

# hr_agent.py  — add near top, after imports
_session_call_count = 0


def get_call_count():
    return _session_call_count


def increment_call_count():
    global _session_call_count
    _session_call_count += 1


class HRAgent:
    def __init__(self):
        # RAG pipeline is stateless between calls; one instance per agent.
        self.pipeline = HRRagPipeline()

    def ask(self, query: str, history_str: str = "") -> dict:
        """
        Process a user query through the full agent pipeline.

        Args:
            query:       Raw user input (may contain PII or injections).
            history_str: Serialized conversation history as plain text.
                         Format: "user: ...\nassistant: ..." (last N turns).

        Returns:
            dict with keys:
                answer         (str)  — final response shown to user
                intents        (list) — classified intent labels
                contexts       (list) — retrieved RAG chunks
                reasoning      (str)  — trace string for debug panel
                shield_triggered (bool) — True if safety shield fired
                threat_type    (str | None) — "PII" | "INJECTION" | None
        """
        # ======== Step 1: Safety Shield (fast, no LLM) ========
        shield = run_shield(query)
        if not shield.is_safe:
            return {
                "intents": ["BLOCKED"],
                "answer": (
                    f"⚠️ Query blocked by Safety Shield.\n"
                    f"**Reason:** {shield.threat_type} — {shield.detail}"
                ),
                "reasoning": {
                    "shield": "🔴 TRIGGERED",
                    "intents": ["BLOCKED"],
                    "classifier_reason": shield.detail,
                    "needs_personal_data": False,
                    "context_sufficient": False,
                    "chunks_retrieved": 0,
                },
                "contexts": [],
                "shield_triggered": True,
                "threat_type": shield.threat_type,
            }

        if not shield.sanitized_query.strip():
            return {
                "answer": "Please enter a question.",
                "intents": [],
                "contexts": [],
                "reasoning": {
                    "shield": "🟢 PASS",
                    "intents": [],
                    "classifier_reason": "Empty query after sanitization.",
                    "needs_personal_data": False,
                    "context_sufficient": False,
                    "chunks_retrieved": 0,
                },
                "shield_triggered": False,
                "threat_type": None,
            }

        # ======== Step 2: Pre-LLM trivial checks ========
        START_GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "howdy"}
        END_GREETINGS = {"bye", "goodbye", "good bye", "thanks", "thank you", "ok", "okay"}
        normalized = shield.sanitized_query.strip().lower().rstrip("!.,")
        answer_reason = None
        if normalized in START_GREETINGS:
            answer_reason = "Hello! I'm your HR assistant for ABC Corporation. How can I help you today?"
        elif normalized in END_GREETINGS:
            answer_reason = "Happy to help you!!"
        if answer_reason:
            return {
                "answer": answer_reason,
                "intents": ["GREETING"],
                "contexts": [],
                "reasoning": {
                    "shield": "🟢 PASS",
                    "intents": ["GREETING"],
                    "classifier_reason": "Rule-based greeting — no LLM needed.",
                    "needs_personal_data": False,
                    "context_sufficient": False,
                    "chunks_retrieved": 0,
                },
                "shield_triggered": False,
                "threat_type": None,
            }

        # ======== Step 3: Intent Classification ========
        # Use sanitized_query so redacted PII doesn't confuse the classifier.
        intents = classify_intent(shield.sanitized_query, history_str)

        if "OUT_OF_SCOPE" in intents and len(intents) == 1:
            return {
                "answer": "I can only answer HR policy questions for ABC Corporation.",
                "intents": intents,
                "contexts": [],
                "reasoning": {
                    "shield": "🟢 PASS",
                    "intents": intents,
                    "classifier_reason": "Query classified as out of scope.",
                    "needs_personal_data": False,
                    "context_sufficient": False,
                    "chunks_retrieved": 0,
                },
                "shield_triggered": False,
                "threat_type": None,
            }

        if "GREETING" in intents:
            return {
                "answer": "Hello! I'm your HR assistant for ABC Corporation. How can I help you today?",
                "intents": intents,
                "contexts": [],
                "reasoning": {
                    "shield": "🟢 PASS",
                    "intents": intents,
                    "classifier_reason": "Greeting detected — no policy lookup needed.",
                    "needs_personal_data": False,
                    "context_sufficient": False,
                    "chunks_retrieved": 0,
                },
                "shield_triggered": False,
                "threat_type": None,
            }

        # ======== Step 4: Build Retrieval Query ========
        # Enrich retrieval query with history for follow-ups.
        retrieval_query = f"{history_str}\n{shield.sanitized_query}"

        # ======== Step 5: RAG Retrieval ========
        rag_result = self.pipeline.ask(retrieval_query)

        # ======== Step 6: Reasoning + Response Generation ========
        response = reason_and_respond(
            query=shield.sanitized_query,
            intents=intents,
            context="\n".join(rag_result["contexts"]),
            history_str=history_str,
        )

        return {
            "answer": response["answer"],
            "intents": intents,
            "contexts": rag_result["contexts"],
            "shield_triggered": response.get("shield_triggered", False),
            "threat_type": response.get("threat_type", None),
            "reasoning": {
                "shield": "PASS",
                "intents": intents,
                "classifier_reason": response.get("reason", ""),
                "needs_personal_data": response.get("needs_personal_data", False),
                "context_sufficient": response.get("context_sufficient", False),
                "chunks_retrieved": len(rag_result["contexts"]),
            },
        }
