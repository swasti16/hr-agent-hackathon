from src.agent.intent_classifier import classify_intent
from src.agent.reasoning_chain import reason_and_respond
from src.hr_rag_pipeline import HRRagPipeline
from src.agent.safety_shield import run_shield


class HRAgent:
    def __init__(self):
        self.pipeline = HRRagPipeline()

    def ask(self, query: str, history_str: str = "") -> dict:
        shield = run_shield(query)
        if not shield.is_safe:
            return {
                "intents": ["BLOCKED"],
                "answer": f"⚠️ Query blocked by Safety Shield.\n**Reason:** {shield.threat_type} — {shield.detail}",
                "reasoning": "Safety Shield prevented processing.",
                "contexts": [],
                "shield_triggered": True,
                "threat_type": shield.threat_type
            }

        intents = classify_intent(shield.sanitized_query, history_str)

        if "OUT_OF_SCOPE" in intents:
            return {
                "answer": "I can only answer HR policy questions for ABC Corporation.",
                "intents": intents,
                "contexts": [],
                "reasoning": "Query classified as out of scope."
            }

        if "GREETING" in intents:
            return {
                "answer": "Hello! I'm your HR assistant for ABC Corporation. How can I help you today?",
                "intents": intents,
                "contexts": [],
                "reasoning": "Greeting detected."
            }

        retrieval_query = query
        if history_str:
            # Take last assistant message to give follow-up context
            last_lines = history_str.strip().split("\n")
            retrieval_query = f"{last_lines[-1]}\n{query}"

        rag_result = self.pipeline.ask(retrieval_query)
        response = reason_and_respond(shield.sanitized_query, intents, "\n".join(rag_result["contexts"]), history_str)

        return {
            "answer": response["answer"],
            "intents": intents,
            "contexts": rag_result["contexts"],
            "reasoning": f"Intents detected: {intents}"
        }
