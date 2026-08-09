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
import threading

_stats_lock = threading.Lock()

session_stats = {
    "total_queries": 0,      # one per user message
    "injections_blocked": 0,
    "pii_blocked": 0,
    "oos_redirected": 0,
    "intent_counts": {}
}


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

        with _stats_lock:
            session_stats["total_queries"] += 1
            if shield.threat_type == "PII":
                session_stats["pii_blocked"] += 1
            if not shield.is_safe and shield.threat_type == "INJECTION":
                session_stats["injections_blocked"] += 1
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

        # ======== Step 3: Intent Classification + Query Resolution ========
        # Use sanitized_query so redacted PII doesn't confuse the classifier.
        # classify_intent() now also resolves follow-ups into standalone
        # queries and signals topic continuity — see intent_classifier.py.
        classification_result = classify_intent(shield.sanitized_query, history_str)
        intents = classification_result["intents"]
        resolved_query = classification_result["resolved_query"]
        topic_continues = classification_result["topic_continues"]

        with _stats_lock:
            for intent in intents:
                session_stats["intent_counts"][intent] = session_stats["intent_counts"].get(intent, 0) + 1
        if "OUT_OF_SCOPE" in intents and len(intents) == 1:
            with _stats_lock:
                session_stats["oos_redirected"] += 1
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

        # ======== Step 4: RAG Retrieval (resolved query, no history needed) ========
        # resolved_query already has pronouns/ellipsis resolved by the
        # classifier above, so retrieval doesn't need raw history at all.
        rag_result = self.pipeline.retrieve(resolved_query)

        # ======== Step 5: Reasoning + Response Generation ========
        # If topic_continues is False, drop history from generation too —
        # prevents stale context (e.g. previous answer's numbers) bleeding
        # into a response about an unrelated subject.
        generation_history = history_str if topic_continues else ""

        response = reason_and_respond(
            query=resolved_query,
            intents=intents,
            context="\n".join(rag_result["contexts"]),
            history_str=generation_history,
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
