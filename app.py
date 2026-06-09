import gradio as gr
import threading
import argparse
from src.agent.hr_agent import HRAgent
import time
import sys
import os
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"


def _parse_args():
    parser = argparse.ArgumentParser(prog='HR Assistant',
                                     description='Answers questions about HR policies.')
    parser.add_argument('--force_reindex', '-f',action='store_true',
                        help='Force reload and re-indexing of HR documents')
    args, _ = parser.parse_known_args()
    return args


args = _parse_args()

agent = HRAgent()
is_ready = False


def initialize():
    global is_ready
    agent.pipeline.load_and_index(force_reindex=args.force_reindex)
    is_ready = True


threading.Thread(target=initialize, daemon=True).start()

custom_css = """
.chat-container {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 15px;
    background: #fafafa;
}
.fixed-height-debug {
    height: 460px;
    overflow-y: auto;
    padding: 12px;
    border-radius: 6px;
}
"""


def normalize_history(gradio_history: list) -> str:
    """Extract last 6 messages from Gradio history as clean string."""
    lines = []
    for msg in gradio_history[-6:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        # Skip any message that contains a shield-triggered response
        violation_str = (
            "SHIELD TRIGGERED",
            "blocked by Safety Shield",
            "ResponsibleAIPolicyViolation"
            )
        if any(i in content for i in violation_str):
            continue
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def chat(message: str, history_str: str) -> tuple[str, str, str | None, list]:
    t0 = time.time()
    print("[CHAT] Start", flush=True)

    result = agent.ask(message, history_str)
    print(f"[CHAT] agent.ask done in {time.time()-t0:.2f}s")

    if result.get("shield_triggered"):
        threat = result.get("threat_type", "UNKNOWN")
        shield_badge = f"**🔴 SHIELD TRIGGERED — {threat}**\n\n"
    else:
        shield_badge = "**🟢 Shield: PASS**\n\n"

    intents_str = " | ".join(result["intents"])
    answer = f"{shield_badge}**Intent:** `{intents_str}`\n\n{result['answer']}"

    # Format structured reasoning chain for debug panel
    r = result.get("reasoning", {})
    if isinstance(r, dict):
        npd = "true" if r.get("needs_personal_data") else "false"
        cs = "true" if r.get("context_sufficient")  else "false"
        shield_status = "🔴 TRIGGERED" if result.get("shield_triggered") else "🟢 PASS"
        chunks_text = (
            f"**🧠 Reasoning Chain**\n\n"
            f"```\n"
            f"├── Shield          : {shield_status}\n"
            f"├── Intent          : {' | '.join(r.get('intents', []))}\n"
            f"├── Classifier      : {r.get('classifier_reason', '')}\n"
            f"├── needs_personal_data : {npd}\n"
            f"├── context_sufficient  : {cs}\n"
            f"└── Retrieved       : {r.get('chunks_retrieved', 0)} chunks\n"
            f"```\n\n---\n"
        )
    else:
        chunks_text = f"**Reasoning:** {r}\n\n---\n"

    for i, chunk in enumerate(result["contexts"], 1):
        chunks_text += f"**Chunk {i}:**\n{chunk}\n\n---\n"

    return answer, chunks_text, result.get("threat_type"), result.get("intents", [])


def respond(message: str, history: list = []) -> tuple[list, str, str]:
    if not message.strip():
        return history, "", ""

    if message.strip().lower() == "/clear":
        return [], "", "*History cleared.*"

    if not is_ready:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant",
                        "content": "Still loading HR documents... please try again in a moment."})
        return history, "", ""

    # Normalize BEFORE appending current message (history = past only)
    history_str = normalize_history(history)

    try:
        bot_response, chunks, threat_type, intents = chat(message, history_str)
    except Exception as e:
        err_str = str(e)
        if "content_filter" in err_str or "ResponsibleAIPolicyViolation" in err_str:
            bot_response = "⚠️ Query blocked by content safety filter."
            temp_history = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": bot_response}
            ]
            return temp_history, "", "**🔴 Blocked by Azure content filter.**"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Sorry, I encountered an error. Please try again."})
        return history, "", f"**Error:** {err_str}"

    print("bot response", bot_response)
    print("chunks", chunks)
    print("threat_type", threat_type)
    print("intents", intents)
    if threat_type == "INJECTION":
        history = []
        temp_history = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_response}
        ]
        return temp_history, "", chunks
    elif "OUT_OF_SCOPE" in intents:
        # Show in UI but don't persist — rebuild history without this exchange
        temp_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": bot_response}
        ]
        return temp_history, "", chunks
    else:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        return history, "", chunks


with gr.Blocks(title="HR Assistant", css=custom_css) as demo:
    gr.Markdown("# 🤖 HR Assistant — ABC Corporation")
    gr.Markdown("Ask me anything about HR policies, leave, and conduct.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=520,
                show_label=False,
                avatar_images=("👤", "🤖")
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask an HR question...",
                    show_label=False,
                    scale=5,
                    container=False
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Column(scale=1, min_width=320):
            with gr.Accordion("🔍 Debug Panel (Click to toggle)", open=True):
                chunks_display = gr.Markdown(
                    value="*Ask a question to see retrieved chunks*",
                    label="Retrieved Chunks",
                    elem_classes="fixed-height-debug"
                )

    send_btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, chunks_display]
    )
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, chunks_display]
    )

    gr.Examples(
        examples=[
            "How many annual leave days do I get?",
            "How much is my night shift allowance?",
            "What is the notice period for resignation?",
            "What happens after a disciplinary violation?"
        ],
        inputs=msg,
        label="Example Questions"
    )

if __name__ == "__main__":
    demo.launch(share=True)
